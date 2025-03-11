"""
Simplified LSTM autoencoder implementation using PyTorch.

PROBLEM:
You've raised an excellent point! Different water monitoring stations likely have unique characteristics that would benefit from specialized models. Let's explore the options and develop a strategy:

RECOMMENDED SOLUTION:
I recommend implementing a tiered approach:
Start with a Global Model: Train on all available stations to learn general water level patterns
Add Station-Specific Fine-Tuning: For stations with sufficient data, fine-tune the model
Implement Dynamic Model Selection: Build a registry system that automatically selects the most appropriate model for each station
Cache and Update Strategy: Store model predictions for each station and only retrain when performance degrades
This gives you the best of both worlds - shared learning across stations while still accommodating station-specific patterns. It also scales well as you add more stations.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_curve

class LSTMAutoencoder(nn.Module):
    """Simple LSTM Autoencoder using PyTorch."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sequence_len: int, dropout: float = 0.2):
        """
        Initialize the LSTM autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            sequence_len: Length of input sequences
            dropout: Dropout rate for regularization
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.dropout = dropout
        
        # Encoder (simpler approach)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.encoder_dropout = nn.Dropout(dropout)
        
        # Decoder (simpler approach)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.decoder_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass through the autoencoder."""
        batch_size = x.size(0)
        
        # Encoder: input -> hidden state
        _, (hidden, cell) = self.encoder_lstm(x)
        hidden = self.encoder_dropout(hidden)
        
        # Create decoder input sequence
        # We need to expand the hidden state to match sequence length
        decoder_input = hidden.repeat(1, 1, 1)  # Shape: [1, batch_size, hidden_dim]
        
        # Create a sequence of the same hidden state
        decoder_input = decoder_input.repeat(self.sequence_len, 1, 1)  # Shape: [seq_len, batch_size, hidden_dim]
        decoder_input = decoder_input.permute(1, 0, 2)  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Decoder: hidden state -> output sequence
        decoder_output, _ = self.decoder_lstm(decoder_input)
        decoder_output = self.decoder_dropout(decoder_output)
        output = self.output_layer(decoder_output)
        
        return output

class AutoencoderWrapper:
    """Wrapper class to provide a unified interface for the PyTorch autoencoder."""
    
    def __init__(self, config: Dict):
        """
        Initialize the autoencoder wrapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sequence_length = config.get('sequence_length', 96)
        self.feature_cols = config.get('feature_cols', ['Value'])
        self.n_features = len(self.feature_cols)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Create the PyTorch model
        self.model = LSTMAutoencoder(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            sequence_len=self.sequence_length,
            dropout=self.dropout_rate
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Data preprocessing
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def fit(self, X, epochs=100, batch_size=32, validation_data=None, callbacks=None, verbose=1, patience=10):
        """
        Train the autoencoder with early stopping support.
        
        Args:
            X: Input data (3D numpy array: samples, timesteps, features)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional validation data
            callbacks: Not used (kept for API compatibility)
            verbose: Verbosity level
            patience: Number of epochs to wait for validation improvement before stopping
            
        Returns:
            Dictionary with training history
        """
        # Convert data to PyTorch tensors
        X_tensor = self._prepare_input(X, debug=True)
        
        # Print debug info
        print(f"X_tensor shape: {X_tensor.shape}")
        
        # Create validation tensor if provided
        if validation_data is not None:
            val_tensor = self._prepare_input(validation_data, debug=True)
            # Set up early stopping
            early_stopping_enabled = True
            early_stopping_counter = 0
            best_val_loss = float('inf')
            best_model_state = None
            print(f"Early stopping enabled with patience {patience}")
        else:
            early_stopping_enabled = False
            print("Early stopping disabled (no validation data)")
        
        # Training history
        history = {'loss': [], 'val_loss': []}
        
        # Create progress bar for epochs
        try:
            from tqdm import tqdm
            epoch_iterator = tqdm(range(epochs), desc="Training")
        except ImportError:
            print(f"Training for {epochs} epochs:")
            epoch_iterator = range(epochs)
        
        # Training loop
        self.model.train()
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            batches_processed = 0
            
            # Process data in batches
            for i in range(0, len(X_tensor), batch_size):
                batches_processed += 1
                batch_end = min(i + batch_size, len(X_tensor))
                inputs = X_tensor[i:batch_end]
                
                # Debug batch shape on first iteration
                if epoch == 0 and i == 0:
                    print(f"First batch shape: {inputs.shape}")
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Debug output shape on first iteration
                if epoch == 0 and i == 0:
                    print(f"First batch output shape: {outputs.shape}")
                
                # Calculate loss
                loss = self.criterion(outputs, inputs)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * (batch_end - i)
            
            # Compute average loss
            avg_loss = epoch_loss / len(X_tensor)
            history['loss'].append(avg_loss)
            
            # Validation and early stopping
            if val_tensor is not None:
                val_loss = self._validate(val_tensor, batch_size)
                history['val_loss'].append(val_loss)
                
                # Early stopping check
                if early_stopping_enabled:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Save model state
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        early_stopping_counter = 0
                        
                        # Update progress message
                        if not isinstance(epoch_iterator, range):
                            epoch_iterator.set_description(f"Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f} (improved)")
                    else:
                        early_stopping_counter += 1
                        
                        # Update progress message
                        if not isinstance(epoch_iterator, range):
                            epoch_iterator.set_description(f"Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f} (no improvement: {early_stopping_counter}/{patience})")
                            
                        # Check if we should stop
                        if early_stopping_counter >= patience:
                            if verbose > 0:
                                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                                print(f"Best validation loss: {best_val_loss:.6f}")
                            break
                
                # Print progress consistently for each epoch
                if isinstance(epoch_iterator, range) and verbose > 0:
                    if early_stopping_enabled and early_stopping_counter > 0:
                        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, val_loss={val_loss:.6f} (no improvement: {early_stopping_counter}/{patience})")
                    else:
                        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, val_loss={val_loss:.6f}")
            elif isinstance(epoch_iterator, range) and verbose > 0:
                print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
        
        # Restore best model if early stopping was used
        if early_stopping_enabled and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored model weights from best epoch with validation loss: {best_val_loss:.6f}")
        
        # Final summary
        print("\nTraining completed:")
        print(f"Initial loss: {history['loss'][0]:.6f}, Final loss: {history['loss'][-1]:.6f}")
        if val_tensor is not None:
            print(f"Initial val_loss: {history['val_loss'][0]:.6f}, Final val_loss: {best_val_loss:.6f}")    
        
        return {'history': history, 'best_val_loss': best_val_loss if early_stopping_enabled else None}
    
    def predict(self, X, batch_size=32, verbose=0):
        """
        Generate reconstructions for input data.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Reconstructed data
        """
        # Prepare input
        X_tensor = self._prepare_input(X, debug=False)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Process in batches
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_end = min(i + batch_size, len(X_tensor))
                inputs = X_tensor[i:batch_end]
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        # Combine batches and convert to numpy
        predictions = np.vstack(predictions)
        
        # Convert back to original scale if needed
        if self.is_fitted:
            # Reshape to 2D for inverse transform
            orig_shape = predictions.shape
            predictions_2d = predictions.reshape(-1, predictions.shape[-1])
            predictions_unscaled = self.scaler.inverse_transform(predictions_2d)
            predictions = predictions_unscaled.reshape(orig_shape)
        
        return predictions
    
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
    
    def _prepare_input(self, X, debug=False):
        """Prepare input data for the PyTorch model."""
        # Scale data if scaler is not fitted
        if not self.is_fitted:
            # Reshape to 2D for scaling
            X_2d = X.reshape(-1, X.shape[-1])
            self.scaler.fit(X_2d)
            self.is_fitted = True
            # Print only on first fit
            print(f"Fitted scaler on data with range [{X_2d.min():.4f}, {X_2d.max():.4f}]")
        
        # Apply scaling
        if len(X.shape) == 3:  # 3D array (samples, timesteps, features)
            orig_shape = X.shape
            X_2d = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_2d)
            X_scaled = X_scaled.reshape(orig_shape)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Only print shape info if debug mode is enabled
        if debug:
            print(f"Input shape: {X_scaled.shape}, Tensor device: {self.model.device}")
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.model.device)
        
        return X_tensor
    
    def _validate(self, val_tensor, batch_size):
        """Evaluate model on validation data."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_tensor), batch_size):
                batch_end = min(i + batch_size, len(val_tensor))
                inputs = val_tensor[i:batch_end]
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                val_loss += loss.item() * (batch_end - i)
        
        avg_val_loss = val_loss / len(val_tensor)
        return avg_val_loss
    
    def save(self, filepath):
        """Save the model to a file."""
        # Save both PyTorch model and scaler
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=self.model.device)
        
        # Recreate model architecture
        self.config = checkpoint['config']
        self.sequence_length = self.config.get('sequence_length', 96)
        self.n_features = len(self.config.get('feature_cols', ['Value']))
        self.hidden_dim = self.config.get('hidden_dim', 32)
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Recreate model and optimizer
        self.model = LSTMAutoencoder(
            input_dim=self.n_features,
            hidden_dim=self.hidden_dim,
            sequence_len=self.sequence_length,
            dropout=self.dropout_rate
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Load saved state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        self.is_fitted = checkpoint['is_fitted']


# Simplified feature engineering
def prepare_sequences(data: pd.DataFrame, sequence_length: int, feature_cols: List[str] = None, verbose: bool = False) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Prepare sequences for LSTM model with minimal feature engineering.
    
    Args:
        data: DataFrame containing time series data
        sequence_length: Length of sequences to create
        feature_cols: List of column names to use as features
        verbose: Whether to print debugging information
    """
    if feature_cols is None:
        feature_cols = ['Value']
    
    # Create a copy to avoid modifying the original
    sequences = []
    timestamps = []
    
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[feature_cols].iloc[i:i+sequence_length].values)
        timestamps.append(data.index[i+sequence_length-1])
    
    # Print summary if verbose
    if verbose:
        print(f"Created {len(sequences)} sequences, shape: {np.array(sequences).shape}")
        print(f"First sequence range: [{np.min(sequences[0]):.4f}, {np.max(sequences[0]):.4f}]")
    
    return np.array(sequences), pd.DatetimeIndex(timestamps)


# Main training function
def train_autoencoder(
    train_data: Dict, 
    validation_data: Optional[Dict] = None, 
    config: Dict = None,
    base_model: AutoencoderWrapper = None
) -> Tuple[AutoencoderWrapper, Dict]:
    """
    Train an LSTM autoencoder on the provided data.
    
    Args:
        train_data: Dictionary of training data
        validation_data: Dictionary of validation data (optional)
        config: Configuration parameters (optional)
        base_model: Optional base model to initialize from (for fine-tuning)
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Use default configuration if none provided
    if config is None:
        config = {
            'sequence_length': 96,
            'hidden_dim': 32,
            'feature_cols': ['Value'],
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'dropout_rate': 0.2
        }
    
    # Extract sequences from training data
    X_train, feature_cols = extract_training_sequences(train_data, config)
    
    if X_train is None or len(X_train) == 0:
        raise ValueError("No valid training sequences extracted")
    
    # Extract validation sequences if provided
    X_val = None
    if validation_data is not None and len(validation_data) > 0:
        X_val, _ = extract_training_sequences(validation_data, config)
    
    # Create model (or use base model for fine-tuning)
    if base_model is not None:
        print("Initializing model from base model for fine-tuning")
        # Create new model with same architecture
        model = AutoencoderWrapper(config)
        
        # Copy weights from base model if compatible
        if model.model.input_dim == base_model.model.input_dim and \
           model.model.sequence_len == base_model.model.sequence_len:
            # Copy encoder weights
            model.model.encoder_lstm.load_state_dict(
                base_model.model.encoder_lstm.state_dict()
            )
            # Copy decoder weights
            model.model.decoder_lstm.load_state_dict(
                base_model.model.decoder_lstm.state_dict()
            )
            model.model.output_layer.load_state_dict(
                base_model.model.output_layer.state_dict()
            )
            
            # Copy scaler if available
            if base_model.is_fitted:
                model.scaler = base_model.scaler
                model.is_fitted = True
                
            print("Successfully transferred weights from base model")
        else:
            print("Warning: Base model architecture doesn't match, training from scratch")
    else:
        # Create new model
        model = AutoencoderWrapper(config)
    
    # Train the model
    history = model.fit(
        X=X_train,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        validation_data=X_val,
        patience=config.get('patience', 10)
    )
    
    return model, history


def extract_training_sequences(data_dict: Dict, config: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Extract model-ready training sequences from dictionary structure.
    
    Args:
        data_dict: Dictionary of station data
        config: Configuration parameters
        
    Returns:
        Tuple of (sequences, feature_columns)
    """
    if data_dict is None or len(data_dict) == 0:
        return None, None
    
    # Get the feature columns from config
    feature_cols = config.get('feature_cols', ['Value'])
    sequence_length = config.get('sequence_length', 96)
    
    # Extract and combine sequences from all stations
    all_sequences = []
    
    for station_key, station_data in data_dict.items():
        try:
            # Get the raw data from the station
            if 'vst_raw' in station_data and station_data['vst_raw'] is not None:
                df = station_data['vst_raw'][feature_cols].copy()
                
                # Prepare sequences with minimal processing
                sequences, _ = prepare_sequences(
                    df, 
                    sequence_length,
                    feature_cols,
                    verbose=False  # Set to True for debugging
                )
                
                if sequences is not None and len(sequences) > 0:
                    all_sequences.append(sequences)
            else:
                print(f"Warning: No raw data found for {station_key}")
        except Exception as e:
            print(f"Error extracting sequences from {station_key}: {e}")
            continue
    
    # Count unique stations (excluding years)
    unique_stations = set()
    for station_key in data_dict.keys():
        if '_' in station_key:
            station_id = station_key.split('_')[0]
            unique_stations.add(station_id)
        else:
            unique_stations.add(station_key)
    
    # Combine all sequences
    if all_sequences:
        combined_sequences = np.vstack(all_sequences)
        print(f"Extracted {len(combined_sequences)} sequences from {len(data_dict)} station-years ({len(unique_stations)} unique stations)")
        return combined_sequences, feature_cols
    else:
        print("Warning: No valid sequences extracted")
        return None, feature_cols


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
        
        # Prepare sequences with minimal processing
        sequences, _ = prepare_sequences(
            df, 
            config['sequence_length'],
            feature_cols,
            verbose=True
        )
        
        return sequences
    else:
        raise ValueError("Could not extract features from data dictionary")

def learn_optimal_threshold(
    model, 
    synthetic_data, 
    ground_truth, 
    config,
    split_datasets=None
):
    """Learn optimal detection threshold using synthetic anomalies."""
    
    normal_errors = []
    anomaly_errors = []
    
    # Process each station with synthetic anomalies
    for station_key, data in synthetic_data.items():
        # Skip if no ground truth available
        if station_key not in ground_truth:
            print(f"Skipping {station_key} - no ground truth available")
            continue
        
        # Get labels for this station
        labels = ground_truth[station_key]
        
        # Prepare proper test data structure - this was missing
        try:
            # Extract station and year from key
            station_id, year = station_key.split('_')
            
            # Skip if we can't find raw data
            if (split_datasets is None or 
                year not in split_datasets['windows'] or
                station_id not in split_datasets['windows'][year] or
                'vst_raw' not in split_datasets['windows'][year][station_id]):
                print(f"Skipping {station_key} - couldn't find raw data")
                continue
            
            # Create proper test data structure with modified data
            test_data = {
                station_key: {
                    'vst_raw_modified': data  # We only need the modified data now
                }
            }
            
            # Get reconstruction errors using realistic evaluation
            results = evaluate_realistic(
                model, test_data, {}, config, split_datasets, window_months=1
            )
            
            # Skip if station not in results
            if station_key not in results:
                print(f"Skipping {station_key} - evaluation produced no results")
                continue
            
            # Extract errors and timestamps
            errors = results[station_key]['reconstruction_errors']
            timestamps = results[station_key]['timestamps']
            
            print(f"Processed {station_key}: {len(errors)} data points")
            
            # Match errors with ground truth periods
            for i, ts in enumerate(timestamps):
                # Find if this point corresponds to an anomaly
                is_anomaly = False
                
                # Look for it in ground truth periods
                for period in labels.get('periods', []):
                    if period['start'] <= ts <= period['end']:
                        is_anomaly = True
                        break
                
                if is_anomaly:
                    anomaly_errors.append(errors[i])
                else:
                    normal_errors.append(errors[i])
        
        except Exception as e:
            print(f"Error processing {station_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not anomaly_errors or not normal_errors:
        print("Warning: Not enough labeled data to optimize threshold")
        return None
    
    # Calculate optimal threshold using F1 score
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_labels = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(anomaly_errors))
    ])
    
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    if len(thresholds) > 0:
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Learned optimal threshold: {optimal_threshold:.6f}")
        print(f"Expected F1 score: {f1_scores[optimal_idx]:.4f}")
        
        return {
            'threshold': optimal_threshold,
            'f1_score': f1_scores[optimal_idx],
            'precision': precision[optimal_idx],
            'recall': recall[optimal_idx],
            'normal_mean': np.mean(normal_errors),
            'normal_std': np.std(normal_errors),
            'anomaly_mean': np.mean(anomaly_errors),
            'anomaly_std': np.std(anomaly_errors)
        }
    else:
        return None 

def evaluate_realistic(
    model, 
    test_data, 
    ground_truth, 
    config, 
    split_datasets=None, 
    window_months=1
):
    """
    Realistic evaluation that doesn't compare against original data.
    
    This function provides a more honest assessment of model performance by only
    looking at the reconstruction error of potentially anomalous data, without
    using original "clean" data as a reference point.
    
    Args:
        model: Trained autoencoder model
        test_data: Dictionary of test data with station_year keys
        ground_truth: Dictionary of ground truth anomaly information
        config: Model configuration
        split_datasets: Optional split datasets information
        window_months: Number of months to use for window analysis
        
    Returns:
        Dictionary of evaluation results
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Tip: Install tqdm for better progress reporting")
        
    results = {}
    
    # Loop through each station-year
    for station_key, station_data in test_data.items():
        try:
            print(f"Evaluating {station_key}")
            
            # Get modified data (potentially anomalous)
            modified_data = station_data.get('vst_raw_modified')
            
            # Skip if missing data
            if modified_data is None or modified_data.empty:
                print(f"Missing vst_raw_modified for {station_key}")
                continue
            
            # Get window size from sequence length
            sequence_length = config.get('sequence_length', 96)
            stride = max(1, sequence_length // 8)  # Stride for efficiency
            
            # Feature columns
            feature_cols = config.get('feature_cols', ['Value'])
            
            # Prepare for sliding window
            all_timestamps = []
            all_reconstructions = []
            all_errors = []
            
            # Calculate total steps
            total_steps = len(modified_data) - sequence_length + 1
            
            # Create iterator with tqdm progress bar if available
            if use_tqdm:
                iterator = tqdm(
                    range(0, total_steps, stride),
                    desc=f"Processing windows",
                    unit="windows", 
                    total=total_steps//stride
                )
            else:
                iterator = range(0, total_steps, stride)
                print(f"Processing {total_steps//stride} windows...")
            
            # Process in stride steps
            for i in iterator:
                # Get window from MODIFIED data only
                mod_window = modified_data.iloc[i:i+sequence_length]
                
                # Create sequence and predict
                mod_seq, _ = prepare_sequences(mod_window, sequence_length, feature_cols)
                
                # Get reconstruction
                reconstructed = model.predict(mod_seq, verbose=0)
                
                # Calculate reconstruction error (how well the model can reproduce the input)
                # We're directly comparing the input to its reconstruction
                error = model.compute_reconstruction_error(mod_seq, reconstructed)
                
                # Store center point for visualization
                center_idx = sequence_length // 2
                all_timestamps.append(mod_window.index[center_idx])
                all_reconstructions.append(reconstructed[0, center_idx, 0])
                all_errors.append(error)
            
            # Convert to arrays
            all_timestamps = pd.DatetimeIndex(all_timestamps)
            all_reconstructions = np.array(all_reconstructions)
            all_errors = np.array(all_errors)
            
            # Compute threshold (can be learned or default percentile)
            if 'optimal_threshold' in config:
                threshold = config['optimal_threshold']
            else:
                threshold = np.percentile(all_errors, 95)  # 95th percentile as default
                
            anomaly_flags = (all_errors > threshold).astype(int)
            
            # Calculate z-scores for better interpretability
            mean_error = np.mean(all_errors)
            std_error = np.std(all_errors)
            z_scores = (all_errors - mean_error) / (std_error + 1e-10)  # Avoid division by zero
            
            # Store results
            results[station_key] = {
                'reconstruction_errors': all_errors,
                'anomaly_flags': anomaly_flags,
                'timestamps': all_timestamps,
                'threshold': threshold,
                'reconstructions': all_reconstructions,
                'modified_data': modified_data,
                'z_scores': z_scores,
                'ground_truth': ground_truth.get(station_key, None)
            }
            
            # Performance metrics against ground truth if available
            if station_key in ground_truth and ground_truth[station_key] is not None:
                try:
                    # Create binary ground truth array aligned with timestamps
                    gt_array = np.zeros(len(all_timestamps))
                    for period in ground_truth[station_key].get('periods', []):
                        # Mark points within anomaly periods as 1
                        mask = (all_timestamps >= period['start']) & (all_timestamps <= period['end'])
                        gt_array[mask] = 1
                    
                    # Calculate metrics
                    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                    
                    precision = precision_score(gt_array, anomaly_flags, zero_division=0)
                    recall = recall_score(gt_array, anomaly_flags, zero_division=0)
                    f1 = f1_score(gt_array, anomaly_flags, zero_division=0)
                    accuracy = accuracy_score(gt_array, anomaly_flags)
                    
                    # Add to results
                    results[station_key]['metrics'] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'accuracy': accuracy
                    }
                    
                    print(f"  Ground truth metrics:")
                    print(f"    Precision: {precision:.4f}")
                    print(f"    Recall: {recall:.4f}")
                    print(f"    F1 Score: {f1:.4f}")
                    print(f"    Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  Error calculating metrics: {e}")
            
            # Print summary
            print(f"\nEvaluation summary for {station_key}:")
            print(f"  Processed {len(all_timestamps)} windows")
            print(f"  Reconstruction error range: [{np.min(all_errors):.6f}, {np.max(all_errors):.6f}]")
            print(f"  Anomaly threshold: {threshold:.6f}")
            print(f"  Found {np.sum(anomaly_flags)} potential anomalies ({np.sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)")
            
        except Exception as e:
            print(f"Error evaluating {station_key}: {e}")
            import traceback
            traceback.print_exc()
    
    return results 