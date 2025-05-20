import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

from _3_lstm_model.objective_functions import get_objective_function
from experiments.iterative_forecaster.alternating_forecast_model import AlternatingForecastModel

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = preprocessor
        
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
        
        # Apply scaling
        if is_training:
            # Initialize dictionary to store feature scalers
            self.feature_scalers = {}
            
            # Scale each feature individually
            x_scaled = np.zeros_like(features_df.values)
            for i, feature in enumerate(feature_cols):
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
            
            # Scale each feature using its fitted scaler
            x_scaled = np.zeros_like(features_df.values)
            for i, feature in enumerate(feature_cols):
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
            # Only enable debug mode for first batch of first epoch
            self.model.debug_mode = (epoch == 0 and batch_size == total_samples)
            
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
                    # Calculate loss on valid targets only
                    loss = self.criterion(outputs[mask], y_batch[mask])
                    loss.backward()
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    batch_loss = loss.item()
                    total_train_loss += batch_loss
                    batch_losses.append(batch_loss)
            
            # Calculate average training loss for the epoch
            avg_train_loss = total_train_loss / max(1, num_batches)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            # Enable debug mode for validation only in first epoch
            self.model.debug_mode = (epoch == 0)
            
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
        
        # Add time-based features if enabled
        if self.config.get('use_time_features', False):
            print("\nAdding time-based features...")
            df = self.preprocessor.feature_engineer._add_time_features(df)
            print(f"  - Added time features: {[col for col in df.columns if col in ['month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']]}")
        
        # Add cumulative features if enabled
        if self.config.get('use_cumulative_features', False):
            print("\nAdding cumulative features...")
            df = self.preprocessor.feature_engineer._add_cumulative_features(df)
            print(f"  - Added cumulative rainfall features")
        
        # Add lagged features if enabled
        if self.config.get('use_lagged_features', False) and 'lag_hours' in self.config:
            print("\nAdding lagged features...")
            df = self.preprocessor.feature_engineer.add_lagged_features(
                df,
                target_col=self.config['output_features'][0],
                lags=self.config['lag_hours']
            )
            print(f"  - Added lagged features for hours: {self.config['lag_hours']}")
        
        # Store all the feature columns to use
        all_columns = [col for col in df.columns if col not in self.config['output_features']]
        self.all_feature_cols = all_columns
        
        print(f"\nAll available feature columns ({len(self.all_feature_cols)}): {self.all_feature_cols}")
        
        # Initialize the model with the correct input size based on all features
        input_size = len(self.all_feature_cols)
        print(f"Initializing model with input size: {input_size}")
        
        self.model = AlternatingForecastModel(
            input_size=input_size,  # Use total number of features
            hidden_size=self.config['hidden_size'],
            output_size=1,  # Always predict 1 time step ahead for water level
            dropout=self.config['dropout'],
            config=self.config  # Pass config to the model
        ).to(self.device)
        
        # Initialize optimizer after model creation
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate'))
        
        print(f"\nTotal features after engineering: {len(df.columns)}")
        
        # Split data based on years
        test_data = df[(df.index.year == 2024)]
        
        # Check if quick mode is enabled
        if self.config.get('quick_mode', False):
            print("\n*** QUICK MODE ENABLED ***")
            print("Using reduced dataset: 3 years training, 1 year validation")
            
            # Use only 2021 for validation in quick mode
            val_data = df[(df.index.year == 2021)]
            
            # Use only 2018-2020 for training in quick mode (3 years)
            train_data = df[(df.index.year >= 2018) & (df.index.year <= 2020)]
            
            print("Quick mode data ranges:")
            print(f"  - Training: 2018-2020 (3 years)")
            print(f"  - Validation: 2021 (1 year)")
            print(f"  - Test: 2024 (unchanged)")
        else:
            # Standard mode: use more data
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