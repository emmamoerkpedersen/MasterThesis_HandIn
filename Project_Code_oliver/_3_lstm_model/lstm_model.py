"""
LSTM-based autoencoder model for anomaly detection in time series data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Conv1D, Dropout, Bidirectional,
    LayerNormalization, Add, Attention, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import importlib.util
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS

class ConvLSTMAutoencoder:
    """
    Convolutional LSTM Autoencoder for anomaly detection in time series data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the autoencoder model with enhanced architecture options.
        
        Args:
            config: Dictionary containing model configuration parameters
                - sequence_length: Length of input sequences
                - n_features: Number of features in the input data
                - lstm_units: List of units in LSTM layers
                - dropout_rate: Dropout rate for regularization
                - learning_rate: Learning rate for optimizer
                - use_attention: Whether to use attention mechanisms
                - use_residual: Whether to use residual connections
                - use_layer_norm: Whether to use layer normalization
        """
        self.config = config
        self.sequence_length = config.get('sequence_length', 96)
        
        # Make sure n_features matches our feature engineering
        # Count features: Original + 5 engineered features (removed offset_detector)
        self.n_features = 6  # Updated to match our refined feature engineering
        
        self.lstm_units = config.get('lstm_units', [64, 32])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.01)
        
        # Architecture enhancement flags
        self.use_attention = config.get('use_attention', False)
        self.use_residual = config.get('use_residual', False)
        self.use_layer_norm = config.get('use_layer_norm', False)
        
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        # Add phase tracking
        self.training_phase = 'initial'  # 'initial' or 'fine_tuning'
        
    def _build_model(self) -> Model:
        """Build an enhanced LSTM autoencoder."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Convolutional feature extraction
        conv_filters = self.config.get('conv_filters', 32)
        kernel_size = self.config.get('kernel_size', 3)
        x = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
        x = Dropout(self.dropout_rate)(x)
        
        # LSTM encoder layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            # Support for bidirectional LSTMs
            if self.config.get('use_bidirectional', False):
                x = Bidirectional(LSTM(units, 
                return_sequences=return_sequences,
                activation='tanh',
                    recurrent_activation='sigmoid'))(x)
            else:
                x = LSTM(units, 
                    return_sequences=return_sequences,
                    activation='tanh',
                    recurrent_activation='sigmoid')(x)
            
            x = Dropout(self.dropout_rate)(x)
        
        # Bottleneck
        encoded = x
        
        # Decoder
        x = RepeatVector(self.sequence_length)(encoded)
        
        # LSTM decoder layers
        for units in reversed(self.lstm_units):
            x = LSTM(units, 
                    return_sequences=True,
                    activation='tanh',
                    recurrent_activation='sigmoid')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        decoded = TimeDistributed(Dense(self.n_features))(x)
        
        # Customize optimizer based on config
        optimizer_name = self.config.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.learning_rate)
        
        # Customize loss function
        loss_function = self.config.get('loss_function', 'mse')
        
        model = Model(inputs, decoded)
        model.compile(
            optimizer=optimizer,
            loss=loss_function
        )
        
        return model
    
    def fit(self, X, epochs=100, batch_size=32, validation_data=None, callbacks=None, verbose=1):
        """Train the model."""
        # Ensure X is the right shape
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, timesteps, features), got shape {X.shape}")
        
        # Fit scaler if not already fitted
        if not self.is_fitted:
            # Reshape to 2D for scaling
            X_2d = X.reshape(-1, X.shape[-1])
            self.scaler.fit(X_2d)
            self.is_fitted = True
        
        return self.model.fit(
            X, X,  # Use same data for input and output (autoencoder)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_data) if validation_data is not None else None,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def predict(self, X, batch_size=None, verbose=1):
        """
        Generate reconstructions for input data.
        
        Args:
            X: Input data
            batch_size: Optional batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Reconstructed data
        """
        if batch_size:
            return self.model.predict(X, batch_size=batch_size, verbose=verbose)
        else:
            return self.model.predict(X, verbose=verbose)
    
    def compute_reconstruction_error(self, X, X_pred):
        """
        Compute reconstruction error between original and reconstructed data.
        
        Args:
            X: Original data
            X_pred: Reconstructed data
            
        Returns:
            Reconstruction error (MSE) for each sample
        """
        return np.mean(np.square(X - X_pred), axis=(1, 2))
    
    def save(self, filepath):
        """Save the model to a file."""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load the model from a file."""
        self.model = tf.keras.models.load_model(filepath)

    def fine_tune(self, 
                 clean_data: np.ndarray,
                 anomalous_data: np.ndarray,
                 validation_data: np.ndarray = None,
                 fine_tuning_epochs: int = 10,
                 fine_tuning_batch_size: int = 32,
                 fine_tuning_lr: float = None) -> Dict:
        """Fine-tune the model to recognize anomalies."""
        
        # Compile model with new learning rate if specified
        if fine_tuning_lr is not None:
            self.model.compile(
                optimizer=Adam(learning_rate=fine_tuning_lr),
                loss='mse'
            )
        
        # Train to reconstruct clean data from anomalous input
        history = self.model.fit(
            anomalous_data,  # Input: anomalous data
            clean_data,      # Target: clean data
            epochs=fine_tuning_epochs,
            batch_size=fine_tuning_batch_size,
            validation_data=(validation_data, validation_data) if validation_data is not None else None,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss' if validation_data is not None else 'loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        return {'history': history.history}


def prepare_sequences_with_features(data: pd.DataFrame, sequence_length: int, feature_cols: List[str] = None, verbose: bool = False) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Prepare sequences for LSTM model with refined feature engineering.
    
    Args:
        data: DataFrame containing time series data
        sequence_length: Length of sequences to create
        feature_cols: List of column names to use as features
        verbose: Whether to print debugging information
    """
    if feature_cols is None:
        feature_cols = data.columns.tolist()
    
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()
    
    # Calculate basic features
    primary_col = feature_cols[0]  # Use first feature as primary
    
    # 1. Rate of change features (important for spike detection)
    enhanced_data['roc_1h'] = enhanced_data[primary_col].diff(4)  # 4 * 15min = 1 hour
    enhanced_data['roc_3h'] = enhanced_data[primary_col].diff(12)  # 12 * 15min = 3 hours
    
    # 2. Rolling statistics (help with detecting flatlines and offsets)
    enhanced_data['rolling_mean_6h'] = enhanced_data[primary_col].rolling(window=24).mean()
    enhanced_data['rolling_std_6h'] = enhanced_data[primary_col].rolling(window=24).std()
    
    # 3. Smoothness measure (helps with noise detection)
    enhanced_data['smoothness'] = abs(enhanced_data[primary_col] - enhanced_data[primary_col].rolling(3).mean())
    
    # Fill missing values created by calculations
    enhanced_data = enhanced_data.bfill().ffill()
    
    # Add verbosity control to debug prints
    if verbose:
        print(f"Feature engineering data sample:")
        print(enhanced_data.head(3))
        print(f"Enhanced data range: [{enhanced_data[primary_col].min():.2f}, {enhanced_data[primary_col].max():.2f}]")
    
    # Create sequences
    all_features = list(enhanced_data.columns)  # Use all features including engineered ones
    sequences = []
    timestamps = []
    
    for i in range(len(enhanced_data) - sequence_length + 1):
        sequences.append(enhanced_data[all_features].iloc[i:i+sequence_length].values)
        timestamps.append(enhanced_data.index[i+sequence_length-1])
    
    # Only print summary if verbose
    if verbose:
        print(f"Created {len(sequences)} sequences, shape: {np.array(sequences).shape}")
        print(f"First sequence data range: [{np.min(sequences[0]):.6f}, {np.max(sequences[0]):.6f}]")
    
    return np.array(sequences), pd.DatetimeIndex(timestamps)

def train_autoencoder(model, X_train, X_val=None, config=None, verbose=1):
    """Train the autoencoder model."""
    print(f"\nStarting model training for {config.get('epochs', 100)} epochs...")
    sys.stdout.flush()
    
    try:
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,  # Add verbose=1 to see when early stopping triggers
                mode='min',
                restore_best_weights=True  # Make sure this is True
            )
        ]
        
        # Add model checkpoint (optional enhancement)
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        callbacks.append(model_checkpoint)
        
        # Train the model
        history = model.fit(
            X_train,
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            validation_data=X_val,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\nModel training complete!")
        return model, {'history': history.history}
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        raise


def train_two_phase_autoencoder(
    train_data: Union[np.ndarray, Dict],
    validation_data: Union[np.ndarray, Dict] = None,
    config: Dict = None,
    verbose: int = 1
) -> Tuple[ConvLSTMAutoencoder, Dict]:
    """Train LSTM autoencoder using two-phase approach."""
    
    print("\nPhase 1: Training on clean data...")
    
    # Initialize model
    model = ConvLSTMAutoencoder(config)
    
    # Extract data from dictionaries if needed
    X_train = _extract_data_from_dict(train_data, config) if isinstance(train_data, dict) else train_data
    X_val = _extract_data_from_dict(validation_data, config) if isinstance(validation_data, dict) and validation_data is not None else validation_data
    
    # Fit the scaler on training data
    if len(X_train.shape) == 3:  # If 3D array (samples, timesteps, features)
        # Reshape to 2D for scaling
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        model.scaler.fit(X_train_2d)
    else:
        model.scaler.fit(X_train)
    
    model.is_fitted = True
    
    # Add this: Create callbacks for both phases
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
    
    callbacks = [early_stopping, checkpoint]
    
    # After initializing the model, add:
    print(f"\nModel structure:")
    print(f"  Input shape: ({config['sequence_length']}, {model.n_features})")
    print(f"  LSTM units: {config.get('lstm_units', [64, 32])}")
    print(f"  Dropout rate: {config.get('dropout_rate', 0.3)}")
    model.model.summary()
    
    # Phase 1: Train on clean data
    model.training_phase = 'initial'
    initial_training_results = _iterative_phase_one(
        model=model,
        data=X_train,
        validation_data=X_val,
        config=config,
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("\nPhase 2: Fine-tuning on synthetic anomalies...")
    model.training_phase = 'fine_tuning'
    fine_tuning_results = _phase_two_fine_tuning(
        model=model,
        data=X_train,
        validation_data=X_val,
        config=config,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Combine results
    training_results = {
        'initial_training': initial_training_results,
        'fine_tuning': fine_tuning_results
    }
    
    # After phase 1 & 2 training, add this:
    
    # Evaluate on each station/year combination
    print("Evaluating model on data with synthetic errors...")
    station_results = {}
    
    # Iterate through each station in training data
    for station_key, station_data in train_data.items():
        if 'vst_raw' in station_data:
            try:
                # Prepare data for evaluation
                feature_cols = config.get('feature_cols', ['Value'])
                eval_df = station_data['vst_raw'][feature_cols].copy()
                
                # Prepare sequences
                eval_sequences, timestamps = prepare_sequences_with_features(
                    eval_df, config['sequence_length'], feature_cols
                )
                
                # Compute reconstructions
                reconstructions = model.predict(eval_sequences)
                
                # Compute reconstruction errors
                errors = model.compute_reconstruction_error(eval_sequences, reconstructions)
                
                # Compute threshold (e.g., percentile-based)
                threshold = np.percentile(errors, 95)
                
                # Flag anomalies
                anomaly_flags = (errors > threshold).astype(int)
                
                # Store results for this station
                station_results[station_key] = {
                    'reconstruction_errors': errors,
                    'anomaly_flags': anomaly_flags,
                    'timestamps': timestamps,
                    'threshold': threshold,
                    'reconstructions': reconstructions
                }
                
                print(f"Added evaluation results for {station_key}")
            except Exception as e:
                print(f"Error evaluating {station_key}: {e}")
                continue
    
    # Add station results to the overall results
    training_results.update(station_results)
    
    # Add this after initial training phase
    print("\nPhase 1 Training Results:")
    if isinstance(initial_training_results, dict) and 'training_history' in initial_training_results:
        for i, hist in enumerate(initial_training_results['training_history']):
            if 'loss' in hist:
                print(f"  Iteration {i+1}: Initial loss: {hist['loss'][0]:.6f}, Final loss: {hist['loss'][-1]:.6f}")
                if 'val_loss' in hist:
                    print(f"    Validation - Initial: {hist['val_loss'][0]:.6f}, Final: {hist['val_loss'][-1]:.6f}")
    
    # Add this after fine-tuning phase
    print("\nPhase 2 Fine-tuning Results:")
    if isinstance(fine_tuning_results, dict) and 'loss' in fine_tuning_results:
        print(f"  Initial loss: {fine_tuning_results['loss'][0]:.6f}, Final loss: {fine_tuning_results['loss'][-1]:.6f}")
        if 'val_loss' in fine_tuning_results:
            print(f"  Validation - Initial: {fine_tuning_results['val_loss'][0]:.6f}, Final: {fine_tuning_results['val_loss'][-1]:.6f}")
    
    return model, training_results


def _extract_data_from_dict(data_dict: Dict, config: Dict) -> np.ndarray:
    """Extract model-ready data from dictionary structure."""
    if data_dict is None:
        return None
        
    # Extract feature columns from first station
    first_station = list(data_dict.values())[0]
    if 'vst_raw' in first_station and first_station['vst_raw'] is not None:
        # Get original data
        feature_cols = config.get('feature_cols', ['Value'])
        df = first_station['vst_raw'][feature_cols].copy()
        
        # Prepare sequences
        sequences, _ = prepare_sequences_with_features(
            df, 
            config['sequence_length'],
            feature_cols
        )
        
        # After preparing sequences
        print(f"Training data range before scaling: [{np.min(sequences):.2f}, {np.max(sequences):.2f}]")
        
        return sequences
    else:
        raise ValueError("Could not extract features from data dictionary")


def _iterative_phase_one(
    model: ConvLSTMAutoencoder,
    data: np.ndarray,
    validation_data: Optional[np.ndarray],
    config: Dict,
    callbacks: Optional[List] = None,
    verbose: int = 1
) -> Tuple[ConvLSTMAutoencoder, Dict]:
    """First phase of training using iterative refinement."""
    
    current_data = data.copy()
    if validation_data is not None:
        current_val_data = validation_data.copy()
    
    results = []
    all_diagnostics = []  # Store diagnostics from each iteration
    
    for iteration in range(config.get('iterative_training', {}).get('max_iterations', 1)):
        print(f"\nIteration {iteration + 1}")
        
        # Transform data using fitted scaler - reshape for scaler operations
        if len(current_data.shape) == 3:  # If 3D array (samples, timesteps, features)
            # Store original shape
            orig_shape = current_data.shape
            # Reshape to 2D for scaling
            reshaped_data = current_data.reshape(-1, current_data.shape[-1])
            # Scale the data
            scaled_reshaped = model.scaler.transform(reshaped_data)
            # Reshape back to 3D
            scaled_data = scaled_reshaped.reshape(orig_shape)
        else:
            scaled_data = model.scaler.transform(current_data)
        
        # Same for validation data if available
        if validation_data is not None:
            if len(current_val_data.shape) == 3:  # If 3D array
                orig_val_shape = current_val_data.shape
                reshaped_val = current_val_data.reshape(-1, current_val_data.shape[-1])
                scaled_reshaped_val = model.scaler.transform(reshaped_val)
                scaled_val_data = scaled_reshaped_val.reshape(orig_val_shape)
            else:
                scaled_val_data = model.scaler.transform(current_val_data)
        else:
            scaled_val_data = None
        
        # Train the model
        history = model.fit(
            scaled_data,
            validation_data=scaled_val_data,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )
        
        results.append(history.history)
        
        # Break if not using iterative training
        if not config.get('iterative_training', {}).get('enabled', False):
            break
            
        # Update data for next iteration (if needed) and collect diagnostics
        current_data, current_val_data, iteration_diagnostics = _refine_training_data(
            model, current_data, current_val_data, config
        )
        all_diagnostics.append(iteration_diagnostics)
        
        # Visualize the potential anomalies that were removed
        if iteration_diagnostics['anomaly_count'] > 0:
            print("\nVisualizing potential anomalies that will be removed:")
            # Save path for the visualization
            save_dir = Path(config.get('output_dir', '.')) / 'diagnostics' / 'anomaly_visualizations'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"iteration_{iteration+1}_anomalies.png"
            
            # Call visualization function with more context
            visualize_refined_data_anomalies(
                data=scaled_data,  # Scaled data for model operations
                diagnostics=iteration_diagnostics,
                original_data=current_data,  # Unscaled data for display
                model=model,  # Pass model for reconstructions and inverse scaling
                max_plots=5,  # Show up to 5 potential anomalies
                context_sequences=2,  # Show 2 sequences before and after
                save_path=str(save_path),
                y_label="Water Level (mm)"  # Appropriate units
            )
    
    # Add diagnostics to results
    combined_results = {
        'training_history': results,
        'refinement_diagnostics': all_diagnostics
    }
    
    return model, combined_results

def _phase_two_fine_tuning(
    model: ConvLSTMAutoencoder,
    data: np.ndarray,
    validation_data: Optional[np.ndarray],
    config: Dict,
    callbacks: Optional[List] = None,
    verbose: int = 1
) -> Tuple[ConvLSTMAutoencoder, Dict]:
    """Second phase of training (fine-tuning)."""
    
    # Same reshaping logic for scaling
    if len(data.shape) == 3:  # If 3D array (samples, timesteps, features)
        # Store original shape
        orig_shape = data.shape
        # Reshape to 2D for scaling
        reshaped_data = data.reshape(-1, data.shape[-1])
        # Scale the data
        scaled_reshaped = model.scaler.transform(reshaped_data)
        # Reshape back to 3D
        scaled_data = scaled_reshaped.reshape(orig_shape)
    else:
        scaled_data = model.scaler.transform(data)
    
    # Same for validation data if available
    if validation_data is not None:
        if len(validation_data.shape) == 3:  # If 3D array
            orig_val_shape = validation_data.shape
            reshaped_val = validation_data.reshape(-1, validation_data.shape[-1])
            scaled_reshaped_val = model.scaler.transform(reshaped_val)
            scaled_val_data = scaled_reshaped_val.reshape(orig_val_shape)
        else:
            scaled_val_data = model.scaler.transform(validation_data)
    else:
        scaled_val_data = None
    
    # Set model to fine-tuning phase
    model.training_phase = 'fine_tuning'
    
    # Compile with new learning rate if specified
    if 'fine_tuning_lr' in config:
        model.model.compile(
            optimizer=Adam(learning_rate=config['fine_tuning_lr']),
            loss='mse'
        )
    
    # Create early stopping callback if not provided
    if callbacks is None:
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data is not None else 'loss',
            patience=3,
            restore_best_weights=True
        )
        callbacks = [early_stopping]
    
    # Train the model with same input/output (autoencoder style)
    history = model.model.fit(
        scaled_data, scaled_data,  # Same data for input and output
        epochs=config.get('fine_tuning_epochs', 10),
        batch_size=config.get('fine_tuning_batch_size', 32),
        validation_data=(scaled_val_data, scaled_val_data) if scaled_val_data is not None else None,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Return the model and training history
    return model, history.history


def evaluate_with_synthetic(model: ConvLSTMAutoencoder, test_data: Dict[str, Dict], ground_truth: Dict[str, np.ndarray], config: Dict) -> Dict:
    """Evaluate the model on test data with synthetic errors."""
    results = {}
    
    for station_name, station_data in test_data.items():
        if 'vst_raw_modified' in station_data and station_data['vst_raw_modified'] is not None:
            # Get both original and modified data
            original_data = station_data['vst_raw'][config['feature_cols']]
            modified_data = station_data['vst_raw_modified'][config['feature_cols']]
            
            # Ensure both datasets cover the same time period
            common_index = original_data.index.intersection(modified_data.index)
            original_data = original_data.loc[common_index]
            modified_data = modified_data.loc[common_index]
            
            # Scale both datasets
            scaled_original = pd.DataFrame(
                model.scaler.transform(original_data),
                index=original_data.index,
                columns=original_data.columns
            )
            scaled_modified = pd.DataFrame(
                model.scaler.transform(modified_data),
                index=modified_data.index,
                columns=modified_data.columns
            )
            
            # Create sequences using the same timestamps
            original_sequences, timestamps = prepare_sequences_with_features(
                scaled_original,
                    config['sequence_length'],
                    config['feature_cols']
                )
            modified_sequences, _ = prepare_sequences_with_features(
                scaled_modified,
                    config['sequence_length'],
                    config['feature_cols']
                )
            
            # Verify sequences have the same shape
            assert original_sequences.shape == modified_sequences.shape, \
                f"Sequence shapes don't match: {original_sequences.shape} vs {modified_sequences.shape}"
            
            # Generate reconstructions using modified data
            reconstructions = model.predict(modified_sequences)
            
            # Compare reconstructions against original sequences to detect anomalies
            errors = model.compute_reconstruction_error(original_sequences, reconstructions)
            
            # Compute adaptive thresholds
            thresholds = compute_adaptive_threshold(errors, timestamps)
            
            # Flag anomalies
            anomaly_flags = errors > thresholds
            
            # Calculate confidence scores (normalized errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            confidence_scores = (errors - min_error) / (max_error - min_error)
            
            # Inverse transform the reconstructions to original scale
            n_samples, seq_len, n_features = reconstructions.shape
            reshaped_recon = reconstructions.reshape(n_samples * seq_len, n_features)
            inverse_transformed = model.scaler.inverse_transform(reshaped_recon)
            reconstructed_original = inverse_transformed.reshape(n_samples, seq_len, n_features)
            
            # Store results
            results[station_name] = {
                'timestamps': timestamps,
                'reconstruction_errors': errors,
                'anomaly_flags': anomaly_flags,
                'confidence_scores': confidence_scores,
                'thresholds': thresholds,  # Now an array
                'reconstructions': reconstructed_original
            }
            
            # If ground truth is available, calculate metrics
            if station_name in ground_truth:
                try:
                    # Get ground truth and convert to numpy array if needed
                    gt_data = ground_truth[station_name]
                    
                    # Debug: Print information about ground truth
                    print(f"Ground truth type: {type(gt_data)}")
                    if hasattr(gt_data, 'shape'):
                        print(f"Ground truth shape: {gt_data.shape}")
                    
                    # Convert to numpy array if it's a pandas object
                    if isinstance(gt_data, (pd.DataFrame, pd.Series)):
                        gt_array = gt_data.values
                    else:
                        gt_array = np.array(gt_data)
                    
                    # Ensure it's a 1D array
                    gt_array = gt_array.flatten()
                    
                    # Now align with our predictions
                    aligned_gt = np.zeros(len(timestamps))
                    
                    # Map timestamps to indices in original data, simplify approach 
                    modified_timestamps = modified_data.index
                    for i, ts in enumerate(timestamps):
                        # Find the index of this timestamp in the original data
                        if ts in modified_timestamps:
                            ts_idx = modified_timestamps.get_loc(ts)
                            if ts_idx < len(gt_array):
                                aligned_gt[i] = 1 if gt_array[ts_idx] == 1 else 0
                    
                    # Calculate metrics
                    binary_flags = anomaly_flags.astype(int)
                    
                    precision = precision_score(aligned_gt, binary_flags, zero_division=0)
                    recall = recall_score(aligned_gt, binary_flags, zero_division=0)
                    f1 = f1_score(aligned_gt, binary_flags, zero_division=0)
                    cm = confusion_matrix(aligned_gt, binary_flags)
                    
                    # Store metrics in results
                    results[station_name]['evaluation_metrics'] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'confusion_matrix': cm,
                        'true_positives': cm[1, 1] if cm.shape == (2, 2) else 0,
                        'false_positives': cm[0, 1] if cm.shape == (2, 2) else 0,
                        'false_negatives': cm[1, 0] if cm.shape == (2, 2) else 0,
                        'true_negatives': cm[0, 0] if cm.shape == (2, 2) else 0,
                        'aligned_ground_truth': aligned_gt
                    }
                    
                    # Calculate average confidence on true anomalies
                    true_anomaly_indices = np.where(aligned_gt == 1)[0]
                    if len(true_anomaly_indices) > 0:
                        avg_confidence = np.mean(confidence_scores[true_anomaly_indices])
                        results[station_name]['evaluation_metrics']['avg_confidence'] = avg_confidence
                
                except Exception as e:
                    print(f"Error processing ground truth for {station_name}: {str(e)}")
                    # Continue without metrics rather than failing the entire process
                    results[station_name]['evaluation_metrics'] = {
                        'error': str(e),
                        'message': 'Failed to calculate metrics due to ground truth format issues'
                    }
                
    return results


def compute_adaptive_threshold(errors, timestamps=None, base_percentile=95):
    """Compute adaptive threshold for anomaly detection."""
    # Basic global threshold - increase from 95th to 97th percentile
    global_threshold = np.percentile(errors, base_percentile)
    
    # If no timestamps provided, use the global threshold
    if timestamps is None:
        return np.full_like(errors, global_threshold)
    
    # Create series for easier manipulation
    error_series = pd.Series(errors, index=timestamps)
    
    # 1. Time-based window calculations (adapts to local patterns)
    window_size = 96 * 7  # 7 days with 15-min intervals
    try:
        # Calculate rolling statistics (centered to avoid look-ahead bias)
        rolling_median = error_series.rolling(window=window_size, center=True, min_periods=window_size//4).median()
        rolling_std = error_series.rolling(window=window_size, center=True, min_periods=window_size//4).std()
        
        # Fill missing values at edges
        rolling_median = rolling_median.fillna(np.median(errors))
        rolling_std = rolling_std.fillna(np.std(errors))
        
        # Adaptive threshold: base (median + 3.5*std) - increased multiplier from 3.0 to 3.5
        adaptive_threshold = rolling_median + (3.5 * rolling_std)
        
        # Ensure threshold doesn't go below a minimum value (increased from 0.7 to 0.8)
        adaptive_threshold = np.maximum(adaptive_threshold.values, global_threshold * 0.8)
        
        # 2. Add seasonal sensitivity if timestamps have dates
        if hasattr(timestamps, 'month'):
            # Create seasonal adjustment factors - make less aggressive
            seasonal_factor = np.ones(len(timestamps))
            
            # Slightly more sensitive in spring/summer when water levels change more
            # Changed from 0.95 to 0.98 (even less aggressive)
            spring_summer = (timestamps.month >= 3) & (timestamps.month <= 8)
            seasonal_factor[spring_summer] = 0.98
            
            # Apply seasonal adjustment
            adaptive_threshold = adaptive_threshold * seasonal_factor
    except Exception as e:
        print(f"Error computing adaptive threshold: {e}")
        return np.full_like(errors, global_threshold)
    
    # Make threshold more sensitive to sudden changes - but less aggressively
    if timestamps is not None:
        # Create a series with error data
        error_series = pd.Series(errors, index=timestamps)
        
        # Calculate rate of change in error
        error_roc = error_series.diff().abs()
        
        # Make this adjustment more conservative - reduced impact further
        # Changed from 1.2 to 1.1 and from 0.3 to 0.2
        roc_factor = np.minimum(error_roc / (error_roc.median() * 3), 1.1)
        adaptive_threshold = adaptive_threshold / (1 + roc_factor * 0.2)
    
    # Add final step: smooth anomaly threshold with rolling min
    if timestamps is not None:
        # Increased window size from 3 to 5 to require anomalies to persist longer
        window_size = 5
        min_window = pd.Series(adaptive_threshold, index=timestamps).rolling(window=window_size, center=True).min()
        
        # Increased from 0.85 to 0.9 to be less aggressive
        adaptive_threshold = np.maximum(min_window.fillna(adaptive_threshold).values, adaptive_threshold * 0.9)
    
    # Add a minimum absolute threshold to avoid flagging tiny deviations (new)
    min_absolute_threshold = 0.002  # Minimum threshold value
    adaptive_threshold = np.maximum(adaptive_threshold, min_absolute_threshold)
    
    return adaptive_threshold


def evaluate_with_sliding_window(model, test_data, ground_truth, config, split_datasets, window_months=1):
    """Evaluate model on test data using sliding window approach."""
    results = {}
    
    for station_key, station_data in test_data.items():
        try:
            print(f"\nEvaluating {station_key}")
            # Get raw and modified data for this station
            if 'vst_raw' not in station_data:
                print(f"Missing vst_raw for {station_key}")
                continue
                
            if 'vst_raw_modified' not in station_data:
                print(f"Missing vst_raw_modified for {station_key}")
                continue
                
            original_data = station_data['vst_raw']  # Store the complete DataFrame
            modified_data = station_data['vst_raw_modified']  # Store the complete DataFrame
            
            print(f"Data shapes: original={original_data.shape}, modified={modified_data.shape}")
            
            # Check data ranges but only once per station
            print(f"Original data range: [{original_data['Value'].min():.2f}, {original_data['Value'].max():.2f}]")
            print(f"Modified data range: [{modified_data['Value'].min():.2f}, {modified_data['Value'].max():.2f}]")
            
            # Convert DataFrames to have a 'Value' column if needed
            if isinstance(original_data, pd.DataFrame) and 'Value' not in original_data.columns:
                # Rename first column to 'Value'
                original_data = original_data.rename(columns={original_data.columns[0]: 'Value'})
            
            if isinstance(modified_data, pd.DataFrame) and 'Value' not in modified_data.columns:
                # Rename first column to 'Value'
                modified_data = modified_data.rename(columns={modified_data.columns[0]: 'Value'})
            
            # Extract feature columns for model processing
            feature_cols = config['feature_cols']
            original_features = original_data[feature_cols]
            modified_features = modified_data[feature_cols]
            
            # Create sliding windows
            window_size = config['sequence_length']
            stride = window_size // 4  # 75% overlap between windows
            
            # First, let's do a detailed debug of a small section
            debug_sample_size = 3  # Number of windows to debug in detail
            print(f"\n===== DETAILED RECONSTRUCTION DEBUG ({debug_sample_size} samples) =====")
            
            for debug_idx in range(0, min(debug_sample_size * window_size, len(modified_features) - window_size), window_size):
                debug_window = modified_features.iloc[debug_idx:debug_idx+window_size]
                debug_seq, _ = prepare_sequences_with_features(debug_window, config['sequence_length'], feature_cols, verbose=False)
                
                # Print raw data values (first feature only)
                print(f"\nSample window at index {debug_idx}:")
                print(f"Raw values (first 5): {debug_window.iloc[:5][feature_cols[0]].values}")
                
                # Get stats before any transformations
                print(f"Original sequence shape: {debug_seq.shape}")
                print(f"Original range: [{np.min(debug_seq):.4f}, {np.max(debug_seq):.4f}]")
                
                # Predict with this sequence
                debug_recon = model.predict(debug_seq, verbose=0)
                
                # Print raw reconstruction values
                print(f"Raw reconstruction shape: {debug_recon.shape}")
                print(f"Raw reconstruction range: [{np.min(debug_recon):.6f}, {np.max(debug_recon):.6f}]")
                
                # Inverse transform manually
                debug_recon_2d = debug_recon.reshape(-1, debug_recon.shape[-1])
                debug_inversed = model.scaler.inverse_transform(debug_recon_2d).reshape(debug_recon.shape)
                
                # Print inverse-transformed values
                print(f"Inverse-transformed range: [{np.min(debug_inversed):.4f}, {np.max(debug_inversed):.4f}]")
                print(f"First 5 reconstructed values: {debug_inversed[0, :5, 0]}")
                
                # Directly compare some values
                for i in range(5):
                    orig_val = debug_window.iloc[i][feature_cols[0]]
                    recon_val = debug_inversed[0, i, 0]
                    print(f"Position {i}: Original={orig_val:.4f}, Reconstructed={recon_val:.4f}, Diff={orig_val-recon_val:.4f}")
            
            print("\n===== END DEBUG SECTION =====\n")
            
            # Now continue with the actual evaluation with modified approach
            
            # Collect full reconstructions for the entire series
            all_input_values = []
            all_recon_values = []
            all_timestamps = []
            all_errors = []
            
            stride = max(1, window_size // 8)  # Use smaller stride for better coverage
            print(f"Processing with stride {stride}...")
            
            for i in range(0, len(modified_features) - window_size + 1, stride):
                if i % (stride * 100) == 0:
                    print(f"  Progress: {i}/{len(modified_features) - window_size + 1}")
                    
                # Get window
                window = modified_features.iloc[i:i+window_size]
                
                # Process the window
                seq, _ = prepare_sequences_with_features(window, config['sequence_length'], feature_cols, verbose=False)
                
                # Get reconstruction
                recon = model.predict(seq, verbose=0)
                
                # Calculate error using correct scaling (transform input to match reconstruction space)
                error = model.compute_reconstruction_error(seq, recon)[0]
                
                # Inverse transform recon
                recon_2d = recon.reshape(-1, recon.shape[-1])
                inversed_recon = model.scaler.inverse_transform(recon_2d).reshape(recon.shape)
                
                # Get the center point only (to avoid duplicate predictions for the same point)
                center_idx = window_size // 2
                center_timestamp = window.index[center_idx]
                
                # Store values
                all_timestamps.append(center_timestamp)
                all_recon_values.append(inversed_recon[0, center_idx, 0])  # Center point, first feature
                all_input_values.append(window.iloc[center_idx][feature_cols[0]])
                all_errors.append(error)
            
            # Convert to arrays
            all_timestamps = pd.DatetimeIndex(all_timestamps)
            all_recon_values = np.array(all_recon_values)
            all_input_values = np.array(all_input_values)
            all_errors = np.array(all_errors)
            
            # Compute threshold based on errors
            threshold = np.percentile(all_errors, 95)
            anomaly_flags = (all_errors > threshold).astype(int)
            
            # Detailed output summary
            print(f"\nReconstruction summary:")
            print(f"Generated {len(all_recon_values)} reconstruction points")
            print(f"Input value range: [{np.min(all_input_values):.2f}, {np.max(all_input_values):.2f}]")
            print(f"Reconstruction range: [{np.min(all_recon_values):.2f}, {np.max(all_recon_values):.2f}]")
            print(f"Mean absolute error: {np.mean(np.abs(all_input_values - all_recon_values)):.4f}")
            print(f"Error range: [{np.min(all_errors):.6f}, {np.max(all_errors):.6f}]")
            print(f"Found {sum(anomaly_flags)} potential anomalies ({sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)")
            
            # Store results
            results[station_key] = {
                'reconstruction_errors': all_errors,
                'anomaly_flags': anomaly_flags,
                'timestamps': all_timestamps,
                'threshold': threshold,
                'reconstructions': all_recon_values,  # 1D array of reconstructed values
                'original_data': original_data,
                'modified_data': modified_data,
                'ground_truth': ground_truth.get(station_key, None)
            }
            
        except Exception as e:
            print(f"Error evaluating {station_key}: {e}")
            import traceback
            print(traceback.format_exc())
    
    return results


def _refine_training_data(
    model: ConvLSTMAutoencoder,
    data: np.ndarray,
    validation_data: Optional[np.ndarray],
    config: Dict
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Refine training data by identifying and removing potential anomalies.
    Now also returns diagnostic information about removed data.
    """
    # Get refinement parameters
    refinement_params = config.get('iterative_training', {}).get('refinement_params', {})
    confidence_threshold = refinement_params.get('confidence_threshold_std', 2.0)
    min_data_fraction = refinement_params.get('min_data_fraction', 0.7)
    
    # Prepare data for reconstruction error calculation
    if len(data.shape) == 3:  # If 3D array (samples, timesteps, features)
        # Store original shape
        orig_shape = data.shape
        # Reshape to 2D for scaling
        reshaped_data = data.reshape(-1, data.shape[-1])
        # Scale the data
        scaled_reshaped = model.scaler.transform(reshaped_data)
        # Reshape back to 3D
        scaled_data = scaled_reshaped.reshape(orig_shape)
    else:
        scaled_data = model.scaler.transform(data)
    
    # Generate reconstructions
    reconstructions = model.predict(scaled_data)
    
    # Calculate reconstruction errors
    errors = model.compute_reconstruction_error(scaled_data, reconstructions)
    
    # Calculate threshold for outlier identification
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    threshold = error_mean + (confidence_threshold * error_std)
    
    # Identify "normal" data points
    normal_indices = errors < threshold
    anomaly_indices = ~normal_indices  # Indices of potential anomalies
    
    # Save diagnostic information
    diagnostics = {
        'threshold': threshold,
        'mean_error': error_mean,
        'std_error': error_std,
        'total_samples': len(data),
        'anomaly_count': np.sum(anomaly_indices),
        'anomaly_percentage': 100 * np.sum(anomaly_indices) / len(data),
        'top_errors': sorted(errors, reverse=True)[:10],  # Top 10 highest errors
        'anomaly_indices': np.where(anomaly_indices)[0].tolist(),  # Save indices for potential visualization
    }
    
    # Ensure we keep at least min_data_fraction of the data
    if np.mean(normal_indices) < min_data_fraction:
        # If we would remove too much data, adjust the threshold to keep min_data_fraction
        sorted_errors = np.sort(errors)
        cutoff_idx = int(len(errors) * min_data_fraction)
        threshold = sorted_errors[cutoff_idx]
        normal_indices = errors < threshold
        anomaly_indices = ~normal_indices
        
        # Update diagnostics after threshold adjustment
        diagnostics.update({
            'adjusted_threshold': threshold,
            'adjusted_anomaly_count': np.sum(anomaly_indices),
            'adjusted_anomaly_percentage': 100 * np.sum(anomaly_indices) / len(data)
        })
    
    # Create refined datasets
    refined_data = data[normal_indices]
    
    # Refine validation data if provided using the same model
    refined_val_data = None
    if validation_data is not None:
        if len(validation_data.shape) == 3:  # If 3D array
            # Store original shape
            orig_val_shape = validation_data.shape
            # Reshape to 2D for scaling
            reshaped_val = validation_data.reshape(-1, validation_data.shape[-1])
            # Scale the data
            scaled_reshaped_val = model.scaler.transform(reshaped_val)
            # Reshape back to 3D
            scaled_val_data = scaled_reshaped_val.reshape(orig_val_shape)
        else:
            scaled_val_data = model.scaler.transform(validation_data)
        
        # Generate reconstructions
        val_reconstructions = model.predict(scaled_val_data)
        
        # Calculate reconstruction errors
        val_errors = model.compute_reconstruction_error(scaled_val_data, val_reconstructions)
        
        # Use the same threshold approach
        val_normal_indices = val_errors < threshold
        
        # Ensure we keep enough validation data
        if np.mean(val_normal_indices) < min_data_fraction:
            sorted_val_errors = np.sort(val_errors)
            val_cutoff_idx = int(len(val_errors) * min_data_fraction)
            val_threshold = sorted_val_errors[val_cutoff_idx]
            val_normal_indices = val_errors < val_threshold
        
        refined_val_data = validation_data[val_normal_indices]
    
    print(f"Refined training data: {len(refined_data)}/{len(data)} samples kept ({100 * len(refined_data)/len(data):.1f}%)")
    if validation_data is not None:
        print(f"Refined validation data: {len(refined_val_data)}/{len(validation_data)} samples kept ({100 * len(refined_val_data)/len(validation_data):.1f}%)")
    
    # Print more detailed diagnostics
    print(f"Anomaly detection threshold: {threshold:.6f}")
    print(f"Identified {diagnostics['anomaly_count']} potential anomalies ({diagnostics['anomaly_percentage']:.2f}%)")
    print(f"Top 5 reconstruction errors: {diagnostics['top_errors'][:5]}")
    
    return refined_data, refined_val_data, diagnostics


def visualize_refined_data_anomalies(
    data: np.ndarray, 
    diagnostics: Dict,
    original_data: Optional[np.ndarray] = None,
    model: Optional[ConvLSTMAutoencoder] = None,
    timestamps=None,
    max_plots: int = 3,
    feature_idx: int = 0,
    context_sequences: int = 2,  # Number of sequences to show before/after
    save_path: Optional[str] = None,
    y_label: str = "Water Level (mm)"
):
    """
    Visualize the data points that were flagged as potential anomalies during refinement.
    
    Args:
        data: The original 3D data array (samples, timesteps, features)
        diagnostics: Diagnostics dict from _refine_training_data
        original_data: Optional unscaled data for plotting in original units
        model: Optional model with scaler for inverse transform
        timestamps: Optional timestamps for x-axis (if None, will use indices)
        max_plots: Maximum number of anomalous sequences to plot
        feature_idx: Index of the feature to plot (default is 0, the raw value)
        context_sequences: Number of sequences to show before and after anomaly
        save_path: Optional path to save the plot
        y_label: Label for y-axis (with units)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Get anomaly indices and error values
    anomaly_indices = diagnostics['anomaly_indices']
    
    if len(anomaly_indices) == 0:
        print("No anomalies detected for visualization")
        return
    
    # Limit the number of plots
    plot_indices = anomaly_indices[:min(max_plots, len(anomaly_indices))]
    
    # Create figure with subplots - one per anomalous sequence
    n_plots = len(plot_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots), sharex=False)
    
    # Handle case of single plot
    if n_plots == 1:
        axes = [axes]
    
    for i, idx in enumerate(plot_indices):
        ax = axes[i]
        
        # Determine context window
        start_idx = max(0, idx - context_sequences)
        end_idx = min(len(data), idx + context_sequences + 1)
        context_indices = list(range(start_idx, end_idx))
        
        # Flag for showing reconstructions
        show_reconstruction = model is not None
        
        # Create a continuous array of all sequences in context
        if show_reconstruction:
            # Get reconstructions for all sequences in context
            context_data = data[context_indices]
            reconstructions = model.predict(context_data)
            
            # Inverse transform if model is provided
            if original_data is not None:
                # Use original data
                sequences_to_plot = original_data[context_indices]
                orig_shape = reconstructions.shape
                reshaped_recon = reconstructions.reshape(-1, reconstructions.shape[-1])
                recon_orig = model.scaler.inverse_transform(reshaped_recon).reshape(orig_shape)
            else:
                # Try to inverse transform if we have the model
                sequences_to_plot = data[context_indices]
                try:
                    # For both data and reconstructions
                    orig_shape = sequences_to_plot.shape
                    reshaped_data = sequences_to_plot.reshape(-1, sequences_to_plot.shape[-1])
                    sequences_to_plot = model.scaler.inverse_transform(reshaped_data).reshape(orig_shape)
                    
                    orig_shape = reconstructions.shape
                    reshaped_recon = reconstructions.reshape(-1, reconstructions.shape[-1])
                    recon_orig = model.scaler.inverse_transform(reshaped_recon).reshape(orig_shape)
                except Exception as e:
                    print(f"Warning: Could not inverse transform data: {e}")
                    sequences_to_plot = data[context_indices]
                    recon_orig = reconstructions
        else:
            # Just use the provided data without reconstructions
            if original_data is not None:
                sequences_to_plot = original_data[context_indices]
            else:
                sequences_to_plot = data[context_indices]
                if model is not None:
                    try:
                        # Try to inverse transform
                        orig_shape = sequences_to_plot.shape
                        reshaped_data = sequences_to_plot.reshape(-1, sequences_to_plot.shape[-1])
                        sequences_to_plot = model.scaler.inverse_transform(reshaped_data).reshape(orig_shape)
                    except Exception as e:
                        print(f"Warning: Could not inverse transform data: {e}")
            
        # Plot each sequence in the context
        for j, seq_idx in enumerate(context_indices):
            sequence = sequences_to_plot[j, :, feature_idx]
            x = range(len(sequence)) if timestamps is None else timestamps[seq_idx]
            
            # Make the anomalous sequence stand out
            if seq_idx == idx:
                ax.plot(x, sequence, 'r-', linewidth=2, label=f'Anomalous Sequence {seq_idx}')
                
                # Add reconstruction if available
                if show_reconstruction:
                    recon_sequence = recon_orig[j, :, feature_idx]
                    ax.plot(x, recon_sequence, 'g--', alpha=0.7, linewidth=2, label='Reconstruction')
                    
                    # Add error area between original and reconstruction
                    ax.fill_between(x, sequence, recon_sequence, color='r', alpha=0.2, label='Error Area')
                
                # Show error value as a line
                error_value = next((err for i, err in enumerate(diagnostics['top_errors']) 
                                  if diagnostics['anomaly_indices'][i] == seq_idx), None)
                
                # Highlight the anomalous sequence with a red box
                y_min, y_max = ax.get_ylim()
                rect = Rectangle((min(x), y_min), max(x) - min(x), y_max - y_min,
                                fill=False, edgecolor='r', linestyle='-', linewidth=2, alpha=0.5)
                ax.add_patch(rect)
            else:
                # Plot context sequences in blue with less emphasis
                alpha = 0.5
                label = f'Context Sequence {seq_idx}' if j == 0 else None  # Only label one context sequence
                ax.plot(x, sequence, 'b-', alpha=alpha, label=label)
                
                # Add reconstruction for context if available
                if show_reconstruction:
                    recon_sequence = recon_orig[j, :, feature_idx]
                    # Only label one reconstruction
                    recon_label = 'Context Reconstruction' if j == 0 else None
                    ax.plot(x, recon_sequence, 'g--', alpha=alpha*0.7, label=recon_label)
        
        # Add a vertical line to separate each sequence
        for j in range(len(context_indices)-1):
            seq_len = sequences_to_plot.shape[1]
            ax.axvline(x=seq_len * (j+1), color='k', linestyle='--', alpha=0.3)
        
        # Add error information to the title
        error_value = next((err for i, err in enumerate(diagnostics['top_errors']) 
                          if diagnostics['anomaly_indices'][i] == idx), None)
        title = f"Flagged Anomaly Sequence {idx}"
        if error_value:
            title += f" (Error: {error_value:.6f}, Threshold: {diagnostics['threshold']:.6f})"
        ax.set_title(title)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        
        # Label axes
        ax.set_ylabel(y_label)
        if i == n_plots - 1:  # Last subplot
            ax.set_xlabel('Time' if timestamps is not None else 'Sequence Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    #plt.show()