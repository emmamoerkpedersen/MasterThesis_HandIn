import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from _3_lstm_model.model import LSTMModel
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature columns list with base features
        self.feature_cols = config['feature_cols'].copy()
        
        # Add feature station columns dynamically
        for station in config['feature_stations']:
            for feature in station['features']:
                feature_name = f"feature_station_{station['station_id']}_{feature}"
                self.feature_cols.append(feature_name)
                
        self.output_features = config['output_features'][0]
    
    def load_and_split_data(self, project_root, station_id):
        """
        Load and Split data into features and target.
        Splits data chronologically with 2024 as test year.
        """
        data_dir = project_root / "data_utils" / "Sample data"
        data = pd.read_pickle(data_dir / "preprocessed_data.pkl")

        # Check if station_id exists in the data dictionary, if not return empty dict
        station_data = data.get(station_id)
        if not station_data:
            raise ValueError(f"Station ID {station_id} not found in the data.")

        # Concatenate all station data columns
        df = pd.concat(station_data.values(), axis=1)

        # Add feature station data
        for station in self.config['feature_stations']:
            feature_station_id = station['station_id']
            feature_station_data = data.get(feature_station_id)
            
            if not feature_station_data:
                raise ValueError(f"Feature station ID {feature_station_id} not found in the data.")
            
            # Add each requested feature from the feature station
            for feature in station['features']:
                if feature not in feature_station_data:
                    raise ValueError(f"Feature {feature} not found in station {feature_station_id}")
                    
                feature_data = feature_station_data[feature][feature].rename(
                    f"feature_station_{feature_station_id}_{feature}"
                )
                df = pd.concat([df, feature_data], axis=1)

        # Start_date is first rainfall not nan, End_date is last vst_raw not nan
        start_date = df['rainfall'].first_valid_index()
        end_date = df['vst_raw'].last_valid_index()
        # Cut dataframe
        data = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Fill temperature and rainfall Nan with bfill and ffill
        data.loc[:, 'temperature'] = data['temperature'].ffill().bfill()
        data.loc[:, 'rainfall'] = data['rainfall'].fillna(-1)
        data.loc[:, 'feature_station_21006845_vst_raw'] = data['feature_station_21006845_vst_raw'].fillna(-1)
        data.loc[:, 'feature_station_21006845_rainfall'] = data['feature_station_21006845_rainfall'].fillna(-1)
        data.loc[:, 'feature_station_21006847_vst_raw'] = data['feature_station_21006847_vst_raw'].fillna(-1)
        data.loc[:, 'feature_station_21006847_rainfall'] = data['feature_station_21006847_rainfall'].fillna(-1)
        print(f"  - Filled temperature and rainfall Nan with bfill and ffill")

        # Add cumulative rainfall and temperature features
        #data = self._add_cumulative_features(data)
        #print(f"  - Added cumulative rainfall and temperature features")
        
        # Add time-based features if enabled in config
        if self.config.get('use_time_features', False):
            data = self._add_time_features(data)
            print(f"  - Added cyclical time features")

        feature_cols = self.feature_cols
        target_feature = self.output_features
        all_features = list(set(feature_cols + [target_feature]))        
        #Filter data to only include the features and target feature
        data = data[all_features]

        # Get the date range of the data
        print(f"\nData ranges from {data.index.min()} to {data.index.max()}")
        
        # Split data based on years
        test_data = data[data.index.year == 2024]  # Test data is 2024
        val_data = data[(data.index.year >= 2022) & (data.index.year <= 2023)]  # Validation is 2022-2023
        train_data = data[data.index.year < 2022]  # Training is everything before 2022
        
        print(f"\nSplit Summary:")
        print(f"Training period: {train_data.index.min().year} - {train_data.index.max().year}")
        print(f"Validation period: 2022 - 2023")
        print(f"Test year: 2024")
        
        print(f"\nDetailed date ranges:")
        print(f"Train data: {train_data.index.min()} to {train_data.index.max()}")
        print(f"Validation data: {val_data.index.min()} to {val_data.index.max()}")
        print(f"Test data: {test_data.index.min()} to {test_data.index.max()}")
        
        print(f'\nData shapes:')
        print(f'Total data: {data.shape}')
        print(f'Train data: {train_data.shape}')
        print(f'Validation data: {val_data.shape}')
        print(f'Test data: {test_data.shape}')

        return train_data, val_data, test_data
    
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation. Scale data and create sequences.
        """
        # Add time-based features if enabled in config
        if self.config.get('use_time_features', False):
            data = self._add_time_features(data)
            
        # Get features and target
        feature_cols = self.feature_cols
        target_col = self.output_features   
        
        # Make sure all feature columns are in the data
        available_features = [col for col in feature_cols if col in data.columns]
        if len(available_features) != len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            print(f"Warning: Missing features in data: {missing}")
        
        # Use only available features
        features = pd.concat([data[col] for col in available_features], axis=1)
        target = pd.DataFrame(data[target_col])

        # Scale data
        scaled_features, scaled_target = self.scale_data(features, target)
        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)
        
        # Only print basic shape info for debugging
        print(f"{'Training' if is_training else 'Validation'} data: {X.shape[0]} sequences of length {X.shape[1]}")
        
        # Convert to tensors and move to device
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def scale_data(self, features, target):
        """
        Scale features and target using StandardScaler.
        """
        if not self.is_fitted:
            # Initialize scalers for each feature
            self.scalers = {
                'features': {},
                'target': StandardScaler()
            }
            
            # Fit target scaler
            self.scalers['target'].fit(target)
            
            # Fit feature scalers for each column
            for col in features.columns:
                self.scalers['features'][col] = StandardScaler()
                self.scalers['features'][col].fit(features[[col]])
                
            self.is_fitted = True
        else:
            # Check if we have new features that weren't present during initialization
            for col in features.columns:
                if col not in self.scalers['features']:
                    # Removed debug print
                    # print(f"New feature detected during scaling: {col}")
                    self.scalers['features'][col] = StandardScaler()
                    self.scalers['features'][col].fit(features[[col]])

        # Scale each feature separately
        scaled_features_list = []
        for col in features.columns:
            scaled_col = self.scalers['features'][col].transform(features[[col]])
            scaled_features_list.append(scaled_col)
        
        # Combine scaled features
        scaled_features = np.hstack(scaled_features_list)

        # Scale target and ensure correct shape
        scaled_target = self.scalers['target'].transform(target).flatten()
        
        return scaled_features, scaled_target

    def _create_sequences(self, features, targets):
         """
         Create sequences using the configured sequence length.
         """
         sequence_length = self.config.get('sequence_length', 5000)  # Get from config or default to 5000
         data_length = len(features)
         
         X, y = [], []
 
         # Create sequences based on configured sequence length
         for i in range(0, data_length, sequence_length):
             end_idx = min(i + sequence_length, data_length)
             feature_seq = features[i:end_idx]
             target_seq = targets[i:end_idx]
 
             # Pad sequences if needed
             if end_idx - i < sequence_length:
                 pad_length = sequence_length - (end_idx - i)
                 feature_seq = np.pad(feature_seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                 target_seq = np.pad(target_seq, (0, pad_length), mode='constant', constant_values=np.nan)
 
             X.append(feature_seq)
             y.append(target_seq)
 
         X = np.array(X)  # Shape: (num_sequences, sequence_length, num_features)
         y = np.array(y)[..., np.newaxis]  # Shape: (num_sequences, sequence_length, 1)
 
         return X, y

    def _add_time_features(self, data):
        """
        Add time-based features to better capture temporal patterns.
        Uses sin/cos encoding for cyclical features (month, day of year).
        """
        # Skip if the data already has time features
        if 'month_sin' in data.columns:
            return data
            
        # Extract datetime components
        data.loc[:, 'month'] = data.index.month
        data.loc[:, 'day'] = data.index.day
        data.loc[:, 'day_of_year'] = data.index.dayofyear
    
        
        # Create cyclical features for month (period = 12)
        data.loc[:, 'month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data.loc[:, 'month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Create cyclical features for day of year (period = 365.25)
        data.loc[:, 'day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
        data.loc[:, 'day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
        
        
        # Add these features to feature_cols (check to avoid duplicates)
        time_features = ['month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
        for feature in time_features:
            if feature not in self.feature_cols:
                self.feature_cols.append(feature)
        
        # Remove the raw time features that we don't want to use directly
        data = data.drop(['month', 'day', 'day_of_year'], axis=1, errors='ignore')
        
        return data

    def _add_cumulative_features(self, data):
        """
        Add cumulative rainfall and temperature features to better capture long-term patterns.
        
        Features added:
        - 1-month (30-day) cumulative rainfall
        - 3-month (90-day) cumulative rainfall
        - 6-month (180-day) cumulative rainfall
        - 30-day average temperature
        - 90-day average temperature
        - Temp-rain interaction (product of temp and rain)
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            DataFrame with added cumulative features
        """
        # First, convert rainfall -1 values (missing) to 0 for cumulative calculations
        rainfall_fixed = data['rainfall'].copy()
        rainfall_fixed[rainfall_fixed == -1] = 0
        
        # Similarly for feature stations
        feature_1_rain = data['feature_station_21006845_rainfall'].copy()
        feature_1_rain[feature_1_rain == -1] = 0
        
        feature_2_rain = data['feature_station_21006847_rainfall'].copy()
        feature_2_rain[feature_2_rain == -1] = 0
        
        # Calculate cumulative rainfall for different windows
        # Using rolling windows with different sizes
        data.loc[:, 'rainfall_30day'] = rainfall_fixed.rolling(window=30, min_periods=1).sum()
        data.loc[:, 'rainfall_90day'] = rainfall_fixed.rolling(window=90, min_periods=1).sum()
        data.loc[:, 'rainfall_180day'] = rainfall_fixed.rolling(window=180, min_periods=1).sum()
        
        # Calculate average temperature for different windows
        data.loc[:, 'temp_30day_avg'] = data['temperature'].rolling(window=30, min_periods=1).mean()
        data.loc[:, 'temp_90day_avg'] = data['temperature'].rolling(window=90, min_periods=1).mean()
        
        # Calculate rainfall-temperature interaction term
        # This can help capture effects where rainfall impact depends on temperature
        data.loc[:, 'temp_rain_interaction'] = data['temperature'] * rainfall_fixed
        
        # Calculate cumulative rainfall for feature stations as well
        data.loc[:, 'feature1_rain_30day'] = feature_1_rain.rolling(window=30, min_periods=1).sum()
        data.loc[:, 'feature2_rain_30day'] = feature_2_rain.rolling(window=30, min_periods=1).sum()
        
        # Calculate combined rainfall (average of all stations)
        combined_rain = (rainfall_fixed + feature_1_rain + feature_2_rain) / 3
        data.loc[:, 'combined_rain_30day'] = combined_rain.rolling(window=30, min_periods=1).sum()
        data.loc[:, 'combined_rain_90day'] = combined_rain.rolling(window=90, min_periods=1).sum()
        
        # Fill potential NaN values created by rolling operations with forward fill then backward fill
        cumulative_cols = [
            'rainfall_30day', 'rainfall_90day', 'rainfall_180day',
            'temp_30day_avg', 'temp_90day_avg', 'temp_rain_interaction',
            'feature1_rain_30day', 'feature2_rain_30day',
            'combined_rain_30day', 'combined_rain_90day'
        ]
        
        for col in cumulative_cols:
            data.loc[:, col] = data[col].ffill().bfill()
            
            # Add new columns to feature_cols list if they're not already there
            if col not in self.feature_cols:
                self.feature_cols.append(col)
        
        return data

class LSTM_Trainer:
    def __init__(self, config, preprocessor):
        """
        Initialize the trainer and LSTM model.
        
        Args:
            config: Dictionary containing model and training parameters
            preprocessor: Instance of DataPreprocessor
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = preprocessor  # Use preprocessor for data handling
        
        # Initialize history dictionary for tracking during training
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'smoothed_val_loss': []}
        
        # Get peak weight for the custom loss function (default to 2.0 if not specified)
        self.peak_weight = config.get('peak_weight', 2.0)
        
        # Get gradient clipping value (default to 1.0 if not specified)
        self.grad_clip_value = config.get('grad_clip_value', 1.0)

        # Initialize LSTM Model using parameters from config
        self.model = LSTMModel(
            input_size=len(preprocessor.feature_cols),
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            output_size=len(config['output_features']),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Choose loss function based on configuration
        if config.get('use_peak_weighted_loss', False):
            print(f"Using peak weighted loss (weight: {self.peak_weight})")
            self.criterion = self.peak_weighted_loss
        else:
            print("Using standard MSE loss")
            self.criterion = nn.MSELoss()
            
        # Print whether gradient clipping is enabled
        print(f"Using gradient clipping with max norm: {self.grad_clip_value}")
        print(f"Using learning rate scheduler with patience 5, factor 0.5")

    def peak_weighted_loss(self, outputs, targets):
        """
        Custom loss function that gives higher weight to errors during peak water levels
        and mid-range values where the model tends to struggle.
        """
        # Calculate normal MSE
        mse_loss = nn.functional.mse_loss(outputs, targets, reduction='none')
        
        # Create weights based on target values
        min_val = targets.min()
        max_val = targets.max()
        if max_val > min_val:  # Avoid division by zero
            normalized_targets = (targets - min_val) / (max_val - min_val)
            
            # Create tailored weighting function specifically for the water level data pattern
            # Higher weights for both peaks (>0.7) and troughs (<0.3)
            peak_weight = torch.pow(normalized_targets, 2) * self.peak_weight
            
            # Targeted boost for mid-range values (0.3-0.7) 
            # Using a Gaussian with max at 0.5 (mid-range)
            mid_boost = 2.0 * torch.exp(-40.0 * torch.pow(normalized_targets - 0.5, 2))
            
            # Combined weights with higher mid-range emphasis
            weights = 1.0 + peak_weight + mid_boost
        else:
            weights = torch.ones_like(targets)
        
        # Apply weights to the MSE loss
        weighted_loss = mse_loss * weights
        
        # Return mean of weighted loss
        return weighted_loss.mean()
    
    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation 
        """
        self.model.train() if training else self.model.eval()
        total_loss = 0
        
        # Lists to store validation predictions and targets
        all_predictions = []
        all_targets = []

        with torch.set_grad_enabled(training):
            for batch_X, batch_y in tqdm(data_loader, desc="Training" if training else "Validating", leave=False):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Create warm-up mask
                warmup_mask = torch.ones_like(batch_y, dtype=torch.bool)
                warmup_mask[:, :150, :] = False

                # Combine warm-up mask with NaN mask
                non_nan_mask = ~torch.isnan(batch_y)
                valid_mask = non_nan_mask & warmup_mask

                valid_outputs = outputs[valid_mask]
                valid_target = batch_y[valid_mask]
                if valid_target.size(0) == 0:
                    continue

                # Use the selected loss function (either MSE or peak_weighted_loss)
                loss = self.criterion(valid_outputs, valid_target)
                
                if training:
                    loss.backward()
                    # Apply gradient clipping to prevent exploding gradients
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                    self.optimizer.step()
                
                total_loss += loss.item()

                if not training:
                    all_predictions.append(outputs.cpu().detach())
                    all_targets.append(batch_y.cpu())

        if training:
            return total_loss / len(data_loader)
        else:
            # Concatenate all validation predictions and targets
            val_predictions = torch.cat(all_predictions, dim=0)
            val_targets = torch.cat(all_targets, dim=0)
            return total_loss / len(data_loader), val_predictions, val_targets

    def train(self, train_data, val_data, epochs, batch_size, patience, epoch_callback=None):
        """
        Train the LSTM model with improved efficiency.
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            epoch_callback: Optional callback function for each epoch (for hyperparameter tuning)
        
        Returns:
            Dictionary with training history and validation predictions and targets
        """
        # Prepare data
        print(f"Train data length: {len(train_data)}")
        print(f"Validation data length: {len(val_data)}")
        X_train, y_train = self.preprocessor.prepare_data(train_data, is_training=True)
        X_val, y_val = self.preprocessor.prepare_data(val_data, is_training=False)

        # Create data loaders with num_workers for parallel data loading
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            batch_size=batch_size, 
            shuffle=True,  # Enable shuffling for better training
            num_workers=0 if self.device.type == 'cuda' else 4  # Set to 0 since data is already on GPU
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0 if self.device.type == 'cuda' else 4  # Set to 0 since data is already on GPU
        )

        # Initialize early stopping
        best_val_loss = float('inf')
        smoothed_val_loss = float('inf')  # Initialize smoothed validation loss
        patience_counter = 0
        best_model_state = None
        
        # Reset history for new training run
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'smoothed_val_loss': []}
        
        # Exponential moving average weight for validation loss
        beta = 0.7  # Weight for previous smoothed value (higher = more smoothing)
        print(f"Using validation loss EMA smoothing with beta={beta}")

        # Training loop with progress bar
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss, val_predictions, val_targets = self._run_epoch(val_loader, training=False)

            # Calculate smoothed validation loss using exponential moving average
            if epoch == 0:
                smoothed_val_loss = val_loss  # Initialize with first value
            else:
                smoothed_val_loss = beta * smoothed_val_loss + (1 - beta) * val_loss
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['smoothed_val_loss'].append(smoothed_val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Step the learning rate scheduler based on validation loss
            # Use smoothed validation loss for scheduler
            self.scheduler.step(smoothed_val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Smoothed Val Loss: {smoothed_val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Call epoch callback if provided (for hyperparameter tuning)
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch, train_loss, val_loss)
                except Exception as e:
                    print(f"Callback raised an exception: {e}")
                    break

            # Early stopping based on smoothed validation loss
            if smoothed_val_loss < best_val_loss:
                best_val_loss = smoothed_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered! No improvement in smoothed validation loss for {patience} epochs.")
                    break

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.history, val_predictions, val_targets

    def predict(self, data):
        """
        Make predictions on new data with proper sequence handling.
        """
        self.model.eval()
        
        # Create a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Add time features if needed - ensure consistency with training
        if self.config.get('use_time_features', False):
            data_copy = self.preprocessor._add_time_features(data_copy)
            
        X, y = self.preprocessor.prepare_data(data_copy, is_training=False)
        
        with torch.no_grad():
            # Make predictions in smaller chunks if needed
            predictions = self.model(X).cpu().numpy()
            y = y.cpu().numpy()
            
            # Preserve temporal order during inverse transform
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_original = self.preprocessor.scalers['target'].inverse_transform(predictions_reshaped)
            
            # Apply smoothing if configured
            if self.config.get('use_smoothing', False):
                alpha = self.config.get('smoothing_alpha', 0.2)  # EMA alpha parameter
                print(f"Applying exponential smoothing with alpha={alpha}")
                predictions_original = self._apply_exponential_smoothing(predictions_original, alpha)
            
            # Reshape back maintaining temporal order
            predictions_original = predictions_original.reshape(predictions.shape)
            
            return predictions_original, predictions, y
    
    def _apply_exponential_smoothing(self, predictions, alpha=0.2):
        """
        Apply exponential moving average smoothing to predictions.
        Uses an adaptive approach that maintains more detail in mid-range values.
        
        Args:
            predictions: Raw predictions array
            alpha: Smoothing factor (0-1). Lower values = more smoothing.
            
        Returns:
            Smoothed predictions
        """
        # Reshape to handle multidimensional arrays
        original_shape = predictions.shape
        predictions_1d = predictions.flatten()
        
        # Initialize smoothed array with first prediction
        smoothed = np.zeros_like(predictions_1d)
        smoothed[0] = predictions_1d[0]
        
        # Determine mid-range values (using percentiles to be robust)
        valid_values = predictions_1d[~np.isnan(predictions_1d)]
        low_threshold = np.percentile(valid_values, 33)
        high_threshold = np.percentile(valid_values, 67)
        
        # Apply EMA formula with adaptive alpha
        for i in range(1, len(predictions_1d)):
            if np.isnan(predictions_1d[i]):
                smoothed[i] = smoothed[i-1]  # Propagate last valid prediction if current is NaN
            else:
                # Use higher alpha (less smoothing) for mid-range values
                current_alpha = alpha
                if low_threshold <= predictions_1d[i] <= high_threshold:
                    # Apply up to 50% more alpha in mid-range for better detail
                    current_alpha = min(0.9, alpha * 1.5)
                    
                smoothed[i] = current_alpha * predictions_1d[i] + (1 - current_alpha) * smoothed[i-1]
        
        # Reshape back to original dimensions
        return smoothed.reshape(original_shape)