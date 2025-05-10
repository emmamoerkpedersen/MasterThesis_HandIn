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

from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
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
        
        # Filter to use only the 3 required features
        self.use_features = [
            'vst_raw_feature',  # Water level (as input feature)
            'temperature',      # Temperature data
            'rainfall'          # Rainfall from primary station
        ]
        
        # Ensure our feature columns are in the correct order (water level first)
        self.feature_cols = []
        for feature in self.use_features:
            if feature in preprocessor.feature_cols:
                self.feature_cols.append(feature)
            else:
                # Look for columns containing this feature
                for col in preprocessor.feature_cols:
                    if feature in col:
                        self.feature_cols.append(col)
                        break
        
        # Verify we have the features we need
        if len(self.feature_cols) < 3:
            raise ValueError(f"Could not find all required features. Found: {self.feature_cols}")
        
        print(f"Using features: {self.feature_cols}")
        
        # Will be set after loading and feature engineering
        self.all_feature_cols = self.feature_cols.copy()
        
        # Initialize model (will be properly initialized after feature engineering)
        self.model = None
        
        # Get the loss function
        self.criterion = get_objective_function(config.get('objective_function'))
    
    def prepare_sequences(self, df, is_training=True):
        """
        Custom method to prepare sequences for the alternating model.
        
        Args:
            df: DataFrame with features and target
            is_training: Whether data is for training
            
        Returns:
            tuple: (x_tensor, y_tensor) containing input features and targets
        """
        # Print feature columns for debugging
        print(f"\nPreparing sequences with features: {list(df.columns)}")
        
        # Extract features and target as DataFrames - ONLY use features explicitly listed in config
        if hasattr(self, 'all_feature_cols'):
            # Use only the features explicitly defined in config
            available_features = [col for col in self.all_feature_cols if col in df.columns]
            filtered_features = [col for col in available_features if col in self.config.get('feature_cols', [])]
            
            if len(filtered_features) == 0:
                # Fallback - look for columns containing the feature names
                for feature in self.config.get('feature_cols', []):
                    for col in available_features:
                        if feature in col:
                            filtered_features.append(col)
                            break
            
            features_df = df[filtered_features]
            print(f"Using {len(filtered_features)} features from config: {filtered_features}")
        else:
            # If all_feature_cols not available, use self.feature_cols (should be rare)
            features_df = df[self.feature_cols]
            print(f"Using {len(self.feature_cols)} features: {list(self.feature_cols)}")
        
        target_df = df[self.config['output_features']]
        
        # Apply scaling
        if is_training:
            # Fit StandardScaler for features
            feature_scaler = StandardScaler()
            x_scaled = feature_scaler.fit_transform(features_df)
            
            # Fit separate StandardScaler for target
            target_scaler = StandardScaler()
            # Handle NaN values in target
            valid_mask = ~np.isnan(target_df.values.flatten())
            target_scaler.fit(target_df.values[valid_mask].reshape(-1, 1))
            
            # Store scalers for later use
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            
            print(f"Fitted scalers on {np.sum(valid_mask)} valid target points")
            print(f"Target scaling - mean: {target_scaler.mean_[0]:.2f}, scale: {target_scaler.scale_[0]:.2f}")
        else:
            # Use previously fitted scalers
            if not hasattr(self, 'feature_scaler') or not hasattr(self, 'target_scaler'):
                raise ValueError("Scalers not fitted. Call with is_training=True first.")
            
            x_scaled = self.feature_scaler.transform(features_df)
        
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
        
        # Use our custom prepare_sequences method
        x_train, y_train = self.prepare_sequences(train_df, is_training=True)
        x_val, y_val = self.prepare_sequences(val_df, is_training=False)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Validation data shape: {x_val.shape}")
        
        # If in quick mode, automatically reduce epochs by half (minimum 5)
        if self.config.get('quick_mode', False) and epochs > 10:
            original_epochs = epochs
            epochs = max(5, epochs // 2)
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
        
        print(f"\nStarting training with {num_batches} batches per epoch")
        print(f"Total samples: {total_samples}, Batch size: {batch_size}")
        
        # Training loop with better progress tracking
        epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=0)
        for epoch in epoch_pbar:
            self.model.train()
            total_train_loss = 0
            batch_losses = []
            
            # Create batch-level progress bar
            train_bar = tqdm(
                range(num_batches), 
                desc=f"Epoch {epoch+1}/{epochs}", 
                position=1, 
                leave=False,
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'
            )
            
            # State resets between epochs
            hidden_states, cell_states = None, None
            
            for batch_idx in train_bar:
                # Extract batch
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, total_samples)
                actual_batch_size = batch_end - batch_start
                
                # Select the current batch from the sequence
                x_batch = x_train[:, batch_start:batch_end, :]
                y_batch = y_train[:, batch_start:batch_end, :]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with alternating pattern
                # Use predictions from previous timesteps in alternating weeks
                outputs, hidden_states, cell_states = self.model(
                    x_batch, 
                    hidden_states, 
                    cell_states,
                    use_predictions=True,
                    alternating_weeks=True
                )
                
                # Detach hidden states to prevent gradient computation through sequences
                hidden_states = [h.detach() for h in hidden_states]
                cell_states = [c.detach() for c in cell_states]
                
                # Calculate loss
                # Create mask for valid targets (non-NaN values)
                mask = ~torch.isnan(y_batch)
                
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
                    
                    # Update progress bar with loss information
                    if len(batch_losses) > 0:
                        avg_loss = sum(batch_losses[-10:]) / min(len(batch_losses), 10)  # Average of last 10 batches
                        train_bar.set_postfix({
                            'samples': f"{batch_end}/{total_samples}",
                            'loss': f"{batch_loss:.6f}",
                            'avg_loss': f"{avg_loss:.6f}"
                        })
            
            # Calculate average training loss for the epoch
            avg_train_loss = total_train_loss / max(1, num_batches)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                # Reset states
                hidden_states, cell_states = None, None
                
                print(f"\nValidating epoch {epoch+1}...")
                # Get predictions
                val_outputs, _, _ = self.model(
                    x_val, 
                    hidden_states, 
                    cell_states,
                    use_predictions=False  # Use original data for validation
                )
                
                # Calculate validation loss
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
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    best_val_predictions = val_outputs.detach().cpu()
                    print(f"New best validation loss: {best_val_loss:.6f}")
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
            hidden_states, cell_states = self.model.init_hidden(x_test.shape[0], self.device)
            
            # Generate predictions
            outputs, _, _ = self.model(
                x_test, 
                hidden_states, 
                cell_states,
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
        Custom data loading function for the alternating model that doesn't rely on feature stations.
        
        Args:
            project_root: Path to project root
            station_id: Station ID to process
            
        Returns:
            train_data, val_data, test_data: DataFrames for training, validation, and testing
        """
        print(f"\nLoading data for station {station_id} with simplified approach...")
        
        # Load the preprocessed data
        data_dir = project_root / "data_utils" / "Sample data"
        data = pd.read_pickle(data_dir / "preprocessed_data.pkl")

        # Check if station_id exists in the data dictionary
        station_data = data.get(station_id)
        if not station_data:
            raise ValueError(f"Station ID {station_id} not found in the data.")

        # Concatenate all station data columns for the main station
        df = pd.concat(station_data.values(), axis=1)
        
        # Print basic feature information
        print("\nInput Features Summary:")
        print("----------------------")
        print("Using only the primary station features:")
        for feature in self.feature_cols:
            print(f"  - {feature}")
        
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
        
        # Fill NaN values for basic features
        df.loc[:, 'temperature'] = df['temperature'].ffill().bfill()
        df.loc[:, 'rainfall'] = df['rainfall'].fillna(0)  # Changed from -1 to 0 for rainfall (more realistic)
        df.loc[:, 'vst_raw_feature'] = df['vst_raw_feature'].fillna(method='ffill')  # Changed to forward fill
        
        # Aggregate temperature to 30 days
        df.loc[:, 'temperature'] = df['temperature'].rolling(window=30, min_periods=1).mean()
        print(f"  - Aggregated temperature to 30 days")
        
        # Add feature engineering if enabled
        original_feature_count = len(df.columns)
        feature_count = original_feature_count
        
        if self.config.get('use_time_features', False):
            print("Adding time-based features...")
            
            # Add day of week (0-6)
            df['day_of_week'] = df.index.dayofweek
            
            # Add hour of day (0-23)
            df['hour_of_day'] = df.index.hour
            
            # Add month (1-12)
            df['month'] = df.index.month
            
            # Add quarter (1-4)
            df['quarter'] = df.index.quarter
            
            # Add sin and cos components for cyclical features (day, hour)
            df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
            df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            
            df['hour_sin'] = np.sin(df['hour_of_day'] * (2 * np.pi / 24))
            df['hour_cos'] = np.cos(df['hour_of_day'] * (2 * np.pi / 24))
            
            feature_count = len(df.columns)
            print(f"  - Added {feature_count - original_feature_count} time features")
        
        if self.config.get('use_cumulative_features', False):
            print("Adding cumulative features...")
            
            # Add 24-hour cumulative rainfall
            df['rainfall_24h'] = df['rainfall'].rolling(window=24*4, min_periods=1).sum()  # 4 readings per hour
            
            # Add 72-hour cumulative rainfall
            df['rainfall_72h'] = df['rainfall'].rolling(window=72*4, min_periods=1).sum()  
            
            # Add 7-day cumulative rainfall
            df['rainfall_7d'] = df['rainfall'].rolling(window=7*24*4, min_periods=1).sum()
            
            print(f"  - Added {len(df.columns) - feature_count} cumulative features")
            print("  - Avoided water level derived features to prevent error propagation")
            feature_count = len(df.columns)
        
        if self.config.get('use_lagged_features', False) and 'lag_hours' in self.config:
            print("Adding lagged features...")
            lag_hours = self.config['lag_hours']
            
            for lag in lag_hours:
                periods = lag * 4  # 4 readings per hour
                df[f'vst_raw_lag_{lag}h'] = df['vst_raw_feature'].shift(periods)
                
            print(f"  - Added {len(df.columns) - feature_count} lagged features")
        
        # Fill any NaN values created by feature engineering
        print("Filling NaN values created by feature engineering...")
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Store all the feature columns to use
        all_columns = [col for col in df.columns if col not in self.config['output_features']]
        
        # Filter out excluded features defined in config
        excluded_features = self.config.get('excluded_features', [])
        if excluded_features:
            excluded_cols = []
            for col in all_columns:
                if any(excluded in col for excluded in excluded_features):
                    excluded_cols.append(col)
            
            if excluded_cols:
                print(f"\nExcluding columns based on config.excluded_features: {excluded_cols}")
                all_columns = [col for col in all_columns if col not in excluded_cols]
        
        self.all_feature_cols = all_columns
        
        print(f"\nAll available feature columns ({len(self.all_feature_cols)}): {self.all_feature_cols}")
        print(f"Will use only features explicitly listed in config.feature_cols")
        
        # Initialize the model now that we know all the features
        # Filter to only use features from config.feature_cols
        filtered_features_count = len([col for col in self.all_feature_cols if col in self.config.get('feature_cols', [])])
        print(f"Filtered features count: {filtered_features_count}")
        
        self.model = AlternatingForecastModel(
            input_size=filtered_features_count,
            hidden_size=self.config['hidden_size'],
            output_size=1,  # Always predict 1 time step ahead for water level
            num_layers=self.config['num_layers'],
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