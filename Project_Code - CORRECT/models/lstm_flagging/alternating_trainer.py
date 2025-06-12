import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent  # models/lstm_flagging -> models -> Project_Code - CORRECT
sys.path.append(str(project_dir))

from models.lstm_traditional.objective_functions import get_objective_function
from .alternating_forecast_model import AlternatingForecastModel

class AlternatingTrainer:
    """
    Trainer for the AlternatingForecastModel that handles the sequential nature
    of the training process and maintains hidden states across batches.
    """
    def __init__(self, config, preprocessor):
        """
        Initialize the trainer and AlternatingForecastModel.
        
        Args:
            config: Dictionary containing model and training parameters
            preprocessor: Instance of DataPreprocessor
        """
        self.config = config
        self.device = torch.device('cpu')  # Use CPU to match main model setup
        self.preprocessor = preprocessor
        
        # Print device information
        print(f"\n🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        print("="*50)
        
        # Initialize feature columns from config
        self.feature_cols = config['feature_cols'].copy()
        
        # Add feature station columns dynamically
        if config.get('feature_stations'):
            for station in config['feature_stations']:
                for feature in station['features']:
                    feature_name = f"feature_station_{station['station_id']}_{feature}"
                    self.feature_cols.append(feature_name)
        
        print(f"Using features: {self.feature_cols}")
        
        # Initialize model (will be properly initialized after feature engineering)
        self.model = None
        
        # Get the loss function
        self.criterion = get_objective_function(config.get('objective_function'))
    
    def prepare_sequences(self, df, is_training=True):
        """
        Custom method to prepare sequences for the alternating model.
        Each feature is scaled individually using its own StandardScaler.
        Scalers are fitted only on training data and reused for validation/test.
        
        NaN handling:
        - Temperature: Forward fill, backward fill, then 30-day aggregation
        - Rainfall: Fill NaN with -1
        - vst_raw_feature: Fill NaN with -1
        - Feature station data: Fill NaN with -1
        - vst_raw (target): Keep NaN values (handled in loss calculation)
        
        Args:
            df: DataFrame with features and target
            is_training: Whether data is for training
            
        Returns:
            tuple: (x_tensor, y_tensor) containing input features and targets
        """
        print(f"\nPreparing sequences:")
        print(f"Input DataFrame shape: {df.shape}")
        
        # Get all feature columns except the target
        feature_cols = [col for col in df.columns if col not in self.config['output_features']]
        print(f"Using all available features: {len(feature_cols)} features")
        print(f"Features: {feature_cols}")
        
        # Extract features and target as DataFrames
        features_df = df[feature_cols]
        target_df = df[self.config['output_features']]
        
        print(f"Features DataFrame shape: {features_df.shape}")
        print(f"Target DataFrame shape: {target_df.shape}")
        
        # Check for NaN/inf values in features before scaling
        print("\nChecking for problematic values before scaling:")
        for col in feature_cols:
            nan_count = features_df[col].isna().sum()
            inf_count = np.isinf(features_df[col]).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  {col}: {nan_count} NaN, {inf_count} inf values")
                
                # Clean the feature data
                cleaned_values = features_df[col].copy()
                
                # Handle NaN values
                if nan_count > 0:
                    # For lagged features, use mean imputation for any remaining NaN
                    if 'lag_' in col:
                        fill_value = cleaned_values.mean()
                        if np.isnan(fill_value):  # If mean is also NaN, use 0
                            fill_value = 0
                        cleaned_values = cleaned_values.fillna(fill_value)
                        print(f"    Filled {col} NaN with mean: {fill_value:.3f}")
                    else:
                        # For other features, use standard handling
                        cleaned_values = cleaned_values.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Handle inf values
                if inf_count > 0:
                    finite_mean = cleaned_values[np.isfinite(cleaned_values)].mean()
                    if np.isnan(finite_mean):
                        finite_mean = 0
                    cleaned_values = cleaned_values.replace([np.inf, -np.inf], finite_mean)
                    print(f"    Replaced {col} inf values with: {finite_mean:.3f}")
                
                # Update the dataframe
                features_df.loc[:, col] = cleaned_values
        
        # Apply scaling
        if is_training:
            # Initialize dictionary to store feature scalers
            self.feature_scalers = {}
            
            # Scale each feature individually, but skip anomaly flags
            x_scaled = np.zeros_like(features_df.values)
            for i, feature in enumerate(feature_cols):
                # Skip scaling for anomaly flag columns - they should remain binary
                if 'anomaly_flag' in feature:
                    x_scaled[:, i] = features_df[feature].values  # Keep original binary values
                    print(f"Skipping scaling for anomaly flag: {feature}")
                else:
                    scaler = StandardScaler()
                    x_scaled[:, i] = scaler.fit_transform(features_df[feature].values.reshape(-1, 1)).flatten()
                    self.feature_scalers[feature] = scaler
            
            # Fit separate StandardScaler for target
            self.target_scaler = StandardScaler()
            # Handle NaN values in target
            valid_mask = ~np.isnan(target_df.values.flatten())
            self.target_scaler.fit(target_df.values[valid_mask].reshape(-1, 1))
            
            print(f"Fitted scalers on {np.sum(valid_mask)} valid target points")
            print(f"Scaled features shape: {x_scaled.shape}")
        else:
            # Use previously fitted scalers
            if not hasattr(self, 'feature_scalers') or not hasattr(self, 'target_scaler'):
                raise ValueError("Scalers not fitted. Call with is_training=True first.")
            
            # Scale each feature using its fitted scaler, but skip anomaly flags
            x_scaled = np.zeros_like(features_df.values)
            for i, feature in enumerate(feature_cols):
                # Skip scaling for anomaly flag columns - they should remain binary
                if 'anomaly_flag' in feature:
                    x_scaled[:, i] = features_df[feature].values  # Keep original binary values
                else:
                    x_scaled[:, i] = self.feature_scalers[feature].transform(
                        features_df[feature].values.reshape(-1, 1)
                    ).flatten()
            
            print(f"Scaled features shape: {x_scaled.shape}")
        
        # Scale the target
        y_values = target_df.values.flatten()
        y_scaled = np.full_like(y_values, np.nan, dtype=float)
        valid_mask = ~np.isnan(y_values)
        
        if np.any(valid_mask):
            if is_training:
                y_scaled[valid_mask] = self.target_scaler.transform(
                    y_values[valid_mask].reshape(-1, 1)
                ).flatten()
            else:
                y_scaled[valid_mask] = self.target_scaler.transform(
                    y_values[valid_mask].reshape(-1, 1)
                ).flatten()
        
        # Print NaN values in final tensors
        print("\nNaN values in final tensors:")
        print("--------------------------")
        for i, feature in enumerate(feature_cols):
            nan_count = np.isnan(x_scaled[:, i]).sum()
            if nan_count > 0:
                print(f"Feature {feature}: {nan_count} NaN values ({nan_count/len(x_scaled)*100:.2f}%)")
        
        nan_count = np.isnan(y_scaled).sum()
        if nan_count > 0:
            print(f"Target: {nan_count} NaN values ({nan_count/len(y_scaled)*100:.2f}%)")
        
        # Final safety check: replace any remaining NaN values in features with 0
        total_nan_features = np.isnan(x_scaled).sum()
        if total_nan_features > 0:
            print(f"WARNING: {total_nan_features} NaN values remain in features - replacing with 0")
            x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensors and add necessary dimensions for LSTM input
        # x shape should be [batch_size, seq_len, num_features]
        # y shape should be [batch_size, seq_len, 1]
        
        # Shape the input tensor correctly for our model
        x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0)  # Add batch dimension
        y_tensor = torch.FloatTensor(y_scaled).unsqueeze(0).unsqueeze(2)  # Add batch and feature dimensions
        
        print(f"Prepared tensors - x_shape: {x_tensor.shape}, y_shape: {y_tensor.shape}")
        
        return x_tensor, y_tensor
    
    def train(self, train_df, val_df, epochs, batch_size=None):
        """
        Train the model with the alternating pattern of original/predicted data.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            epochs: Number of epochs to train
            batch_size: Batch size for training (if None, use full dataset)
            
        Returns:
            history: Dictionary with training history
            val_predictions: Tensor with validation predictions
            val_targets: Tensor with validation targets
        """
        print("\nPreparing training and validation data...")
        
        x_train, y_train = self.prepare_sequences(train_df, is_training=True)
        x_val, y_val = self.prepare_sequences(val_df, is_training=False)
        
        print(f"\nTraining data shapes:")
        print(f"x_train: {x_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"x_val: {x_val.shape}")
        print(f"y_val: {y_val.shape}")
        
        # If in quick mode, automatically reduce epochs
        if self.config.get('quick_mode', False) and epochs > 10:
            original_epochs = epochs
            epochs = max(5, epochs // 3)
            print(f"\n*** QUICK MODE: Reducing epochs from {original_epochs} to {epochs} ***")
        
        # If no batch_size provided, train on chunks rather than full data
        if batch_size is None:
            # Use a reasonable chunk size to avoid memory issues (50,000 samples)
            batch_size = min(50000, x_train.shape[1] // 10)
            print(f"Using chunk size of {batch_size} samples")
        
        # Move data to device
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # Initialize training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_val_predictions = None
        
        # Total number of batches
        total_samples = x_train.shape[1]
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"\nTraining configuration:")
        print(f"Total samples: {total_samples}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches per epoch: {num_batches}")
        
        epoch_pbar = tqdm(range(epochs), desc="Training Progress")
        for epoch in epoch_pbar:
            self.model.train()
            
            total_train_loss = 0
            batch_losses = []
            
            # State resets between epochs
            hidden_state, cell_state = None, None
            
            for batch_idx in range(num_batches):
                # Extract batch
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, total_samples)
                actual_batch_size = batch_end - batch_start
                
                # Select the current batch from the sequence
                x_batch = x_train[:, batch_start:batch_end, :]
                y_batch = y_train[:, batch_start:batch_end, :]
                
                self.optimizer.zero_grad()
                
                # Forward pass with alternating pattern
                outputs, hidden_state, cell_state = self.model(
                    x_batch, 
                    hidden_state, 
                    cell_state,
                    use_predictions=True,
                    alternating_weeks=True
                )
                
                # Detach hidden states to prevent gradient computation through sequences
                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()
                
                # Calculate loss
                # Create mask for valid targets (non-NaN values)
                mask = ~torch.isnan(y_batch)
                
                # Add warm-up mask - don't calculate loss for warmup period
                warmup_length = self.config.get('warmup_length', 0)
                if warmup_length > 0:
                    warmup_mask = torch.ones_like(mask, dtype=torch.bool)
                    warmup_mask[:, :warmup_length, :] = False
                    # Combine warm-up mask with NaN mask
                    mask = mask & warmup_mask
                
                if mask.any():
                    # Use weighted loss if anomaly flags are available and enabled
                    if (self.config.get('use_weighted_loss', False) and 
                        self.config.get('anomaly_flag_column', 'anomaly_flag') in 
                        [col for col in self.all_feature_cols if 'anomaly_flag' in col]):
                        
                        # Extract anomaly flags from the batch
                        # Find the anomaly flag column index
                        anomaly_flag_idx = None
                        for i, col in enumerate(self.all_feature_cols):
                            if self.config.get('anomaly_flag_column', 'anomaly_flag') in col:
                                anomaly_flag_idx = i
                                break
                        
                        if anomaly_flag_idx is not None:
                            # Extract anomaly flags for this batch
                            anomaly_flags = x_batch[:, :, anomaly_flag_idx].unsqueeze(2)  # Match y_batch shape
                            
                            # Apply mask to anomaly flags too
                            masked_flags = anomaly_flags[mask]
                            
                            # Use the model's weighted loss function
                            loss = self.model.anomaly_aware_loss(
                                outputs[mask], 
                                y_batch[mask], 
                                masked_flags,
                                self.criterion
                            )
                        else:
                            # Fallback to standard loss if flag column not found
                            loss = self.criterion(outputs[mask], y_batch[mask])
                    else:
                        # Standard loss calculation
                        loss = self.criterion(outputs[mask], y_batch[mask])
                        
                    loss.backward()
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    self.optimizer.step()
                    
                    batch_loss = loss.item()
                    total_train_loss += batch_loss
                    batch_losses.append(batch_loss)
            
            # Calculate average training loss for the epoch
            avg_train_loss = total_train_loss / max(1, num_batches)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            
            with torch.no_grad():
                # Reset states
                hidden_state, cell_state = None, None
                
                # Get predictions
                val_outputs, _, _ = self.model(
                    x_val, 
                    hidden_state, 
                    cell_state,
                    use_predictions=False  # Use original data for validation
                )
                
                mask = ~torch.isnan(y_val)
                if mask.any():
                    val_loss = self.criterion(val_outputs[mask], y_val[mask]).item()
                else:
                    val_loss = float('inf')
                
                history['val_loss'].append(val_loss)
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.6f}",
                    'val_loss': f"{val_loss:.6f}",
                    'best_val_loss': f"{best_val_loss:.6f}"
                })
                
                # Print epoch results
                print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                if val_loss < best_val_loss:
                    print(f"New best validation loss: {best_val_loss:.6f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    best_val_predictions = val_outputs.detach().cpu()
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{self.config.get('patience', 5)}")
                    if patience_counter >= self.config.get('patience', 5):
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Clean up CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return history, best_val_predictions, y_val.cpu()
    
    def predict(self, test_df, num_steps=None, use_predictions=True):
        """
        Generate predictions using the trained model.
        
        Args:
            test_df: Test DataFrame
            num_steps: Number of steps to predict (if None, predict entire sequence)
            use_predictions: Whether to use model's own predictions as input
            
        Returns:
            predictions: Model predictions
            y_test: Original targets
        """
        # Prepare test data
        x_test, y_test = self.prepare_sequences(test_df, is_training=False)
        x_test = x_test.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Determine prediction length
        if num_steps is None:
            num_steps = x_test.shape[1]
        
        with torch.no_grad():
            # Initialize hidden and cell states
            hidden_state, cell_state = self.model.init_hidden(x_test.shape[0], self.device)
            
            # Generate predictions
            outputs, _, _ = self.model(
                x_test, 
                hidden_state, 
                cell_state,
                use_predictions=use_predictions
            )
            
            # Convert predictions back to original scale
            predictions_scaled = outputs.cpu().numpy()
            y_test_np = y_test.numpy()
            
            # Inverse transform predictions using our target_scaler
            predictions = self.target_scaler.inverse_transform(predictions_scaled)
            
            return predictions, predictions_scaled, y_test_np
    
    def load_data(self, project_root, station_id):
        """
        Custom data loading function for the alternating model that uses both primary station
        and configured feature stations.
        
        NaN handling:
        - Temperature: Forward fill, backward fill, then 30-day aggregation
        - Rainfall: Fill NaN with -1
        - vst_raw_feature: Fill NaN with -1
        - vst_raw (target): Keep NaN values (handled in loss calculation)
        
        Args:
            project_root: Path to project root
            station_id: Station ID to process
            
        Returns:
            train_data, val_data, test_data: DataFrames for training, validation, and testing
        """
        print(f"\nLoading data for station {station_id}...")
        
        # Load the preprocessed data
        data_dir = project_root / "data_utils" / "Sample data"
        data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
        #Print keys in the data dictionary
        #print(f"\nKeys in the data dictionary: {data.keys()}")
        #print(f"\nKeys in the station data: {data[station_id].keys()}")
        
        # Check if station_id exists in the data dictionary
        station_data = data.get(station_id)
        if not station_data:
            raise ValueError(f"Station ID {station_id} not found in the data.")

        # Initialize DataFrame with only the features specified in config
        df = pd.DataFrame(index=station_data[list(station_data.keys())[0]].index)
        
        # Add only the features specified in config['feature_cols']
        for feature in self.config['feature_cols']:
            if feature in station_data:
                df[feature] = station_data[feature][feature]
            else:
                raise ValueError(f"Feature {feature} not found in station data")
        
        # Add target column
        if self.config['output_features'][0] in station_data:
            df[self.config['output_features'][0]] = station_data[self.config['output_features'][0]][self.config['output_features'][0]]
        else:
            raise ValueError(f"Target feature {self.config['output_features'][0]} not found in station data")
        
        # Add feature station data if configured
        if self.config.get('feature_stations'):
            print("\nAdding feature station data:")
            for station in self.config['feature_stations']:
                feature_station_id = station['station_id']
                feature_station_data = data.get(feature_station_id)
                
                if not feature_station_data:
                    raise ValueError(f"Feature station ID {feature_station_id} not found in the data.")
                
                # Add each requested feature from the feature station
                for feature in station['features']:
                    if feature not in feature_station_data:
                        raise ValueError(f"Feature {feature} not found in station {feature_station_id}")
                        
                    feature_name = f"feature_station_{feature_station_id}_{feature}"
                    df[feature_name] = feature_station_data[feature][feature]
                    print(f"  - Added {feature} from station {feature_station_id}")
        
        # Print basic feature information
        print("\nInput Features Summary:")
        print("----------------------")
        print("Using features from:")
        print("  - Primary station:")
        for feature in self.config['feature_cols']:
            print(f"    * {feature}")
        
        if self.config.get('feature_stations'):
            print("  - Feature stations:")
            for station in self.config['feature_stations']:
                print(f"    * Station {station['station_id']}:")
                for feature in station['features']:
                    print(f"      - {feature}")
        
        print("\nFeature engineering settings:")
        print(f"  - Using time features: {self.config.get('use_time_features', False)}")
        print(f"  - Using cumulative features: {self.config.get('use_cumulative_features', False)}")
        print(f"  - Using lagged features: {self.config.get('use_lagged_features', False)}")
        print("----------------------\n")

        # Set date range
        start_date = pd.Timestamp('2010-01-04')
        end_date = pd.Timestamp('2025-01-07')
        
        # Cut dataframe to date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Handle NaN values for input features
        print("\nHandling NaN values:")
        print("  - Temperature: Forward fill, backward fill, then 30-day aggregation")
        df.loc[:, 'temperature'] = df['temperature'].ffill().bfill()
        df.loc[:, 'temperature'] = df['temperature'].rolling(window=30, min_periods=1).mean()
        
        print("  - Rainfall: Fill NaN with -1")
        df.loc[:, 'rainfall'] = df['rainfall'].fillna(-1)
        
        print("  - vst_raw_feature: Fill NaN with -1")
        df.loc[:, 'vst_raw_feature'] = df['vst_raw_feature'].fillna(-1)
        
        # Fill NaN values for feature station data
        if self.config.get('feature_stations'):
            print("  - Feature station data: Fill NaN with -1")
            for station in self.config['feature_stations']:
                for feature in station['features']:
                    col_name = f"feature_station_{station['station_id']}_{feature}"
                    if col_name in df.columns:
                        df.loc[:, col_name] = df[col_name].fillna(-1)
        
        print("  - Target (vst_raw): Keep NaN values (handled in loss calculation)")
        
        # Store all the feature columns to use (only basic features at this point)
        all_columns = [col for col in df.columns if col not in self.config['output_features']]
        self.all_feature_cols = all_columns
        
        print(f"\nBasic feature columns loaded ({len(self.all_feature_cols)}): {self.all_feature_cols}")
        
        # Note: Feature engineering (time, cumulative, lagged) will be done later after error injection
        
        # Initialize the model with basic input size (will be updated after feature engineering)
        input_size = len(self.all_feature_cols)
        print(f"Initializing model with basic input size: {input_size}")
        
        self.model = AlternatingForecastModel(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            output_size=1,
            dropout=self.config['dropout'],
            config=self.config
        ).to(self.device)
        
        # Initialize optimizer after model creation
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate'))

        # Split data based on years
        test_data = df[(df.index.year == 2024)]
        
        # Check if quick mode is enabled
        if self.config.get('quick_mode', False):
            print("\n*** QUICK MODE ENABLED ***")
            print("Using balanced dataset: 4-5 years training, 1 year validation")
            
            # Use full year 2023 for validation (12 months)
            val_data = df[df.index.year == 2023]  # Full 2023
            
            # Use 2018-2022 for training in quick mode (5 years)
            train_data = df[(df.index.year >= 2018) & (df.index.year <= 2022)]
            
            print("Quick mode data ranges:")
            print(f"  - Training: 2018-2022 (5 years)")
            print(f"  - Validation: 2023 (full year)")
            print(f"  - Test: 2024 (unchanged)")
        elif self.config.get('full_dataset_mode', False):
            print("\n*** FULL DATASET MODE ENABLED ***")
            print("Using maximum available data for better learning")
            
            # Use full year 2023 for validation (12 months)
            val_data = df[df.index.year == 2023]  # Full 2023
            
            # Use all available years before 2023 for training (2010-2022, ~13 years)
            train_data = df[df.index.year < 2023]
            
            print("Full dataset mode data ranges:")
            print(f"  - Training: 2010-2022 (~13 years)")
            print(f"  - Validation: 2023 (full year)")
            print(f"  - Test: 2024 (unchanged)")
        else:
            # Standard mode: use moderate data
            val_data = df[(df.index.year >= 2022) & (df.index.year <= 2023)]  # Validation: 2022-2023
            train_data = df[df.index.year < 2022]  # Training: before 2022
        
        # Print NaN values in each split after all handling
        print("\nNaN values in data splits (after handling):")
        print("----------------------------------------")
        for split_name, split_data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
            print(f"\n{split_name} data:")
            for col in split_data.columns:
                nan_count = split_data[col].isna().sum()
                if nan_count > 0:
                    print(f"  {col}: {nan_count} NaN values ({nan_count/len(split_data)*100:.2f}%)")
        
        print(f"\nSplit Summary:")
        print(f"Training period: {train_data.index.min().year} - {train_data.index.max().year}")
        print(f"Validation period: {val_data.index.min().year} - {val_data.index.max().year}")
        print(f"Test year: {test_data.index.min().year}")
        
        print(f'\nData shapes:')
        print(f'Total data: {df.shape}')
        print(f'Train data: {train_data.shape}')
        print(f'Validation data: {val_data.shape}')
        print(f'Test data: {test_data.shape}')
        
        return train_data, val_data, test_data
    
    def add_all_features(self, data):
        """
        Add ALL features to data after synthetic error injection.
        This ensures all features (time, cumulative, lagged) are created from the corrupted data.
        
        Args:
            data: DataFrame with corrupted data (basic features only)
            
        Returns:
            DataFrame with all features added
        """
        print(f"\n🔧 Adding ALL features after error injection...")
        
        # Create a copy to avoid modifying original data
        data_with_features = data.copy()
        
        # 1. Add time-based features if enabled
        if self.config.get('use_time_features', False):
            print("  - Adding time features...")
            data_with_features = self.preprocessor.feature_engineer._add_time_features(data_with_features)
        
        # 2. Add cumulative features if enabled
        if self.config.get('use_cumulative_features', False):
            print("  - Adding cumulative features...")
            data_with_features = self.preprocessor.feature_engineer._add_cumulative_features(data_with_features)
        
        # 3. Add lagged features if enabled (using corrupted data)
        if self.config.get('use_lagged_features', False) and 'lag_hours' in self.config:
            print(f"  - Adding lagged features: {self.config['lag_hours']} hours...")
            
            # 15-minute intervals: 4 timesteps per hour
            timesteps_per_hour = 4
            target_col = self.config['output_features'][0]
            
            # Use vst_raw_feature as source for lagged features (avoid look-ahead bias)
            source_col = 'vst_raw_feature' if 'vst_raw_feature' in data_with_features.columns else target_col
            source_data = data_with_features[source_col].copy()
            
            print(f"    Using {source_col} as source for lagged features")
            
            # Check for NaN/inf values in source and handle them
            nan_count = source_data.isna().sum()
            inf_count = np.isinf(source_data).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"    Warning: Source has {nan_count} NaN and {inf_count} inf values")
                # Clean the source data for lagged feature creation
                source_data = source_data.fillna(method='ffill').fillna(method='bfill')
                # Replace any remaining inf values with a reasonable value
                if np.isinf(source_data).any():
                    finite_mean = source_data[np.isfinite(source_data)].mean()
                    source_data = source_data.replace([np.inf, -np.inf], finite_mean)
            
            # Create lagged features
            for lag_hours in self.config['lag_hours']:
                lag_timesteps = lag_hours * timesteps_per_hour
                feature_name = f'{target_col}_lag_{lag_hours}h'
                
                # Create lagged feature
                lagged_values = source_data.shift(lag_timesteps)
                
                # Handle NaN values at the beginning more robustly
                if lagged_values.isna().any():
                    # First try forward fill
                    lagged_values = lagged_values.fillna(method='ffill')
                    
                    # If still NaN at the beginning, use backward fill
                    if lagged_values.isna().any():
                        lagged_values = lagged_values.fillna(method='bfill')
                        
                    # If still NaN, use the overall mean of the source data
                    if lagged_values.isna().any():
                        fill_value = source_data.mean()
                        lagged_values = lagged_values.fillna(fill_value)
                        print(f"    Filled remaining NaN in {feature_name} with mean: {fill_value:.3f}")
                
                data_with_features[feature_name] = lagged_values
                
                # Add to feature columns list if not already there
                if feature_name not in self.preprocessor.feature_engineer.feature_cols:
                    self.preprocessor.feature_engineer.feature_cols.append(feature_name)
        
        # Update all_feature_cols to include all features
        all_columns = [col for col in data_with_features.columns if col not in self.config['output_features']]
        self.all_feature_cols = all_columns
        
        # Verify no NaN values remain in any features
        total_nan = 0
        problematic_features = []
        for col in all_columns:
            col_nan = data_with_features[col].isna().sum()
            total_nan += col_nan
            if col_nan > 0:
                problematic_features.append(f"{col}({col_nan})")
        
        if problematic_features:
            print(f"    Warning: Features with NaN: {', '.join(problematic_features)}")
        
        print(f"  ✅ Feature engineering complete:")
        print(f"    - Time features: {self.config.get('use_time_features', False)}")
        print(f"    - Cumulative features: {self.config.get('use_cumulative_features', False)}")
        print(f"    - Lagged features: {self.config.get('use_lagged_features', False)}")
        print(f"    - Total features: {len(self.all_feature_cols)}")
        print(f"    - Total NaN values: {total_nan}")
        
        return data_with_features

    def reinitialize_model_for_anomaly_flags(self, sample_data):
        """
        Reinitialize the model with the correct input size after anomaly flags are added.
        
        Args:
            sample_data: Sample data with anomaly flags to determine correct input size
        """
        # Count feature columns (excluding target)
        feature_cols = [col for col in sample_data.columns if col not in self.config['output_features']]
        input_size = len(feature_cols)
        
        print(f"\n🔄 Reinitializing model for anomaly flags...")
        print(f"   New input size: {input_size} (includes anomaly flags)")
        print(f"   Features: {feature_cols}")
        
        # Update all_feature_cols
        self.all_feature_cols = feature_cols
        
        # Create new model with correct input size
        self.model = AlternatingForecastModel(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            output_size=1,
            dropout=self.config['dropout'],
            config=self.config
        ).to(self.device)
        
        # Reinitialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate'))
        
        print(f"   ✅ Model reinitialized successfully") 