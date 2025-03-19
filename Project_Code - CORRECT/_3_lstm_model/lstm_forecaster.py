import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import time
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

'''
Remove clear_gpu_memory and MemoryEfficientDataset

Try and see which parts of the code are actually necessary.
- Do we need attention?
- Do we need residual connections?
- Do we need custom loss function? Or can we use just base MSE?
- Do we need input normalization?
- Can we simplify the output projection?
- Adam VS lr_scheduler

Refactoring:
- Move prepare_data to a lstm_preprocessing.py file.
- Move smooth_mse_loss to a objective function file, where we can specify different objective
  functions to use when running the LSTM.

USE PROPER TQDM BAR.
'''

class SimpleLSTMModel(nn.Module):
    """
    Simple LSTM model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(SimpleLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.model_name = 'SimpleLSTM'
        
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Move model to device
        self.to(self.device)
        print(f"Model using device: {self.device}")
    
    def forward(self, x):
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x)
        
        # Take output from all timesteps
        out = self.dropout(out)
        
        # Pass through fully connected layer
        predictions = self.fc(out)
        
        return predictions

class train_LSTM:
    def __init__(self, model, config):
        """Initialize the trainer with model and configuration."""
        self.model = model
        self.config = config
        
        # Get device from model
        self.device = model.device
        
        # Use simple MSE loss
        self.criterion = nn.MSELoss()
        
        # Initialize optimizer (only Adam)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # Initialize scalers
        self.scalers = {}
        self.target_scaler = None
        self.is_fitted = False
        
        # Store feature names for prediction
        self.feature_names = config.get('feature_cols', [])
        self.target_dim = 1  # Default target dimension
        
        print("\nInitialized LSTM trainer:")
        print(f"  Model: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"  Learning rate: {config.get('learning_rate', 0.001)}")
        print(f"  Features: {len(self.feature_names)} input features")

    # def smooth_mse_loss(self, y_pred, y_true):
    #     """
    #     Custom loss function that combines MSE with peak detection and smoothness.
    #     """
    #     # Get loss weights from config
    #     loss_weights = self.config.get('loss_weights', {
    #         'mse': 0.3,
    #         'peak': 0.4,
    #         'gradient': 0.15,
    #         'smoothness': 0.15
    #     })
        
    #     # Base MSE loss
    #     mse_loss = F.mse_loss(y_pred, y_true)
        
    #     # Peak detection and weighting
    #     peak_std = self.config.get('peak_detection_std', 1.5)
    #     peak_weight = self.config.get('peak_weight', 5.0)
    #     high_value_weight = self.config.get('high_value_weight', 2.0)
        
    #     # Calculate mean of true values for high value weighting
    #     y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
    #     high_value_mask = y_true > y_true_mean
        
    #     # Calculate rolling statistics for peak detection
    #     # Use a simpler approach that maintains tensor size
    #     diff_from_mean = y_true - y_true_mean
    #     abs_diff = torch.abs(diff_from_mean)
    #     threshold = peak_std * torch.std(y_true, dim=1, keepdim=True)
    #     peaks = abs_diff > threshold
        
    #     # Apply weights to peaks and high values
    #     weighted_peaks = torch.where(peaks, peak_weight * torch.ones_like(y_true), torch.ones_like(y_true))
    #     weighted_peaks = torch.where(high_value_mask, weighted_peaks * high_value_weight, weighted_peaks)
    #     peak_loss = torch.mean(weighted_peaks * torch.square(y_pred - y_true))
        
    #     # Gradient matching
    #     dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    #     dy_true = y_true[:, 1:] - y_true[:, :-1]
    #     gradient_loss = F.mse_loss(dy_pred, dy_true)
        
    #     # Smoothness penalty
    #     smoothness_weight = self.config.get('smoothness_weight', 0.2)
    #     smoothness_loss = torch.mean(torch.square(dy_pred))
        
    #     # Combine all components
    #     total_loss = (
    #         loss_weights['mse'] * mse_loss +
    #         loss_weights['peak'] * peak_loss +
    #         loss_weights['gradient'] * gradient_loss +
    #         loss_weights['smoothness'] * smoothness_weight * smoothness_loss
    #     )
        
    #     return total_loss

    # def get_lr_scheduler(self, optimizer, warmup_steps=100, max_lr=0.001, patience=5):
    #     """Create a learning rate scheduler with warmup."""
    #     # Define a custom learning rate lambda function
    #     def lr_lambda(step):
    #         if step < warmup_steps:
    #             # Linear warmup from 0.1 * max_lr to max_lr
    #             return 0.1 + 0.9 * float(step) / float(max(1, warmup_steps))
    #         else:
    #             # After warmup, use a constant learning rate
    #             return 1.0
        
    #     # Create a LambdaLR scheduler for warmup
    #     warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    #     # Create a ReduceLROnPlateau scheduler for after warmup
    #     plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',
    #         factor=0.5,
    #         patience=patience,
    #         verbose=True,
    #         min_lr=1e-6
    #     )
        
    #     # Return a dictionary with both schedulers
    #     return {
    #         'warmup': warmup_scheduler,
    #         'plateau': plateau_scheduler,
    #         'current_step': 0
    #     }

    def prepare_data(self, data, is_training=True):
        """Prepare data for sequence-to-sequence prediction with minimal features."""
        if not data:
            print("Warning: Empty data dictionary provided")
            return torch.FloatTensor([])
            
        try:
            station_id = list(data.keys())[0]
            station_data = data[station_id]
            
            # Get features and target
            target_feature = self.config.get('output_features', ['vst_raw'])[0]
            
            # Print brief information about the data
            print(f"\nPreparing {'training' if is_training else 'prediction'} data...")
            
            # Convert DataFrames to Series if needed
            for feature in list(station_data.keys()):
                if isinstance(station_data[feature], pd.DataFrame):
                    if feature in station_data[feature].columns:
                        station_data[feature] = station_data[feature][feature]
                    elif len(station_data[feature].columns) == 1:
                        station_data[feature] = station_data[feature].iloc[:, 0]
                    else:
                        print(f"Warning: Cannot convert {feature} DataFrame to Series. Using first column.")
                        station_data[feature] = station_data[feature].iloc[:, 0]
            
            # Create a DataFrame with all features
            all_data = pd.DataFrame()
            
            # Identify base features vs derived features
            base_features = self.config['feature_cols'].copy()
            
            # Define time features only - removed all target-derived features
            time_features = [
                'month_sin', 'month_cos',
                'day_sin', 'day_cos', 
                'weekday_sin', 'weekday_cos'
            ]
            
            # Remove all lagged features as they were derived from target
            derived_features = time_features  # Now only contains time features
            
            # Filter out derived features from base features
            actual_base_features = [f for f in base_features if f not in derived_features]
            
            # Add actual base features from input data
            for feature in actual_base_features:
                if feature in station_data and isinstance(station_data[feature], pd.Series):
                    all_data[feature] = station_data[feature]
                else:
                    print(f"Warning: Base feature {feature} not found in data. Using zeros.")
                    if target_feature in station_data and isinstance(station_data[target_feature], pd.Series):
                        all_data[feature] = pd.Series(0.0, index=station_data[target_feature].index)
            
            # Add target feature
            if target_feature in station_data and isinstance(station_data[target_feature], pd.Series):
                all_data[target_feature] = station_data[target_feature]
            else:
                print(f"ERROR: Target feature {target_feature} not found in data!")
                return torch.FloatTensor([])
            
            # Resample to 15-minute frequency
            all_data = all_data.resample('15T').mean().interpolate(method='time').ffill().bfill()
            
            # Add time features
            print("Adding time-based features (month, day, weekday only)...")
            # Month of year (cyclical)
            all_data['month_sin'] = np.sin(2 * np.pi * all_data.index.month / 12)
            all_data['month_cos'] = np.cos(2 * np.pi * all_data.index.month / 12)
            # Day of month (cyclical)
            all_data['day_sin'] = np.sin(2 * np.pi * all_data.index.day / 31)
            all_data['day_cos'] = np.cos(2 * np.pi * all_data.index.day / 31)
            # Day of week (cyclical)
            all_data['weekday_sin'] = np.sin(2 * np.pi * all_data.index.weekday / 7)
            all_data['weekday_cos'] = np.cos(2 * np.pi * all_data.index.weekday / 7)
            
            # Only update feature_cols during training
            if is_training:
                # Define optimized feature set - ensure no duplicates
                optimized_features = actual_base_features.copy()  # Start with actual base features
                
                # Add time features if not already present
                for feature in time_features:
                    if feature not in optimized_features:
                        optimized_features.append(feature)
                
                # Update feature_cols to include optimized features
                self.config['feature_cols'] = optimized_features
                
                print(f"Using optimized feature set: {self.config['feature_cols']}")
            
            # Drop NaN values that might have been introduced by lagged features
            all_data = all_data.dropna()
            
            if len(all_data) == 0:
                print("ERROR: All data was dropped after removing NaN values!")
                return torch.FloatTensor([])
            
            # Extract features and target
            features = all_data[self.config['feature_cols']]
            target_data = all_data[target_feature]
            
            print(f"Processed {len(features)} data points with {len(features.columns)} features")
            
            # Initialize or update scalers during training with simple min-max scaling
            if is_training and not self.is_fitted:
                self.scalers = {}
                for col in features.columns:
                    if isinstance(features[col], pd.Series):
                        feature_values = features[col]
                    else:
                        feature_values = features[col].iloc[:, 0]
                    
                    # Simple min-max scaling with small padding
                    feature_min = float(feature_values.min())
                    feature_max = float(feature_values.max())
                    padding = 0.1 * (feature_max - feature_min)  # 10% padding
                    
                    self.scalers[col] = {
                        'min': feature_min - padding,
                        'max': feature_max + padding
                    }
                
                # Min-max scaling for target with asymmetric padding
                target_min = float(target_data.min())
                target_max = float(target_data.max())
                target_range = target_max - target_min
                
                # More padding above than below for water levels
                self.target_scaler = {
                    'min': target_min - 0.1 * target_range,  # 10% padding below
                    'max': target_max + 0.2 * target_range   # 20% padding above
                }
                print(f"Target scaling range: [{self.target_scaler['min']:.2f}, {self.target_scaler['max']:.2f}]")
                self.is_fitted = True

            elif not self.is_fitted:
                # Default scalers if not in training mode
                print("Warning: Scalers not initialized. Creating default scalers.")
                self.scalers = {col: {'min': 0.0, 'max': 1.0} for col in features.columns}
                self.target_scaler = {'min': 0.0, 'max': 2000.0}  # Default range for water levels
                self.is_fitted = True
            
            # Scale features using min-max scaling
            scaled_features = np.zeros((len(features), len(features.columns)))
            for i, col in enumerate(features.columns):
                if isinstance(features[col], pd.Series):
                    feature_values = features[col].values
                else:
                    feature_values = features[col].iloc[:, 0].values
                
                # Apply min-max scaling
                feature_min = self.scalers[col]['min']
                feature_max = self.scalers[col]['max']
                scaled_features[:, i] = (feature_values - feature_min) / (feature_max - feature_min)
            
            # Scale target
            target_min = self.target_scaler['min']
            target_max = self.target_scaler['max']
            scaled_target_data = (target_data.values - target_min) / (target_max - target_min)
            
            # Create tensors
            X = scaled_features.reshape(1, -1, scaled_features.shape[1])
            y = scaled_target_data.reshape(1, -1, 1)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            print(f"Tensor shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")
            print(f"Device: {X_tensor.device}")
            
            # Store the index for later use in prediction
            self.data_index = target_data.index
            
            return X_tensor, y_tensor
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            import traceback
            traceback.print_exc()
            return torch.FloatTensor([])

    def _create_data_loaders(self, X, y, batch_size):
        """Create data loaders with batches."""
        # Move tensors to CPU for data loading
        X = X.cpu()
        y = y.cpu()
        
        # Make data contiguous in memory
        X = X.contiguous()
        y = y.contiguous()
        
        # Create dataset and loader
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle during training
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return loader

    def train(self, train_data, val_data, epochs, batch_size, patience):
        """Train the model with batches."""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, is_training=True)
        X_val, y_val = self.prepare_data(val_data, is_training=False)
        
        # Create train and validation data loaders
        train_loader = self._create_data_loaders(X_train, y_train, batch_size=batch_size)
        val_loader = self._create_data_loaders(X_val, y_val, batch_size=batch_size)
        
        # Clear some memory
        del X_train, y_train, X_val, y_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        print(f"\nTraining for {epochs} epochs with patience {patience}:")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Device: {self.device}")
        
        try:
            # Training loop
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # Train for one epoch
                self.model.train()
                train_loss = 0
                total_batches = 0
                
                # Process each batch
                for batch_idx, (data, targets) in enumerate(train_loader):
                    # Print batch shapes
                    print(f"Batch {batch_idx + 1} shapes - Input: {data.shape}, Target: {targets.shape}")
                    
                    # Move data to device
                    data = data.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    outputs = self.model(data)
                    
                    # Ensure outputs and targets have the same shape
                    if outputs.shape[1] != targets.shape[1]:
                        min_length = min(outputs.shape[1], targets.shape[1])
                        outputs = outputs[:, :min_length, :]
                        targets = targets[:, :min_length, :]
                    
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    total_batches += 1
                    
                    # Clear memory
                    del data, targets, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate average training loss
                avg_train_loss = train_loss / total_batches
                
                # Validate
                val_loss = self._validate_epoch(val_loader)
                
                # Store history
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(val_loss)
                history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Print progress
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    print(f"  ✓ New best validation loss: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"  - No improvement for {patience_counter}/{patience} epochs")
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        break
                
                # Clear memory at the end of each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            if best_model_state is not None:
                print("Loading last best model state...")
                self.model.load_state_dict(best_model_state)
            raise e
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with validation loss: {best_val_loss:.6f}")
            
        return history

    def _validate_epoch(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                
                # Ensure outputs and targets have the same shape
                if outputs.shape[1] != targets.shape[1]:
                    min_length = min(outputs.shape[1], targets.shape[1])
                    outputs = outputs[:, :min_length, :]
                    targets = targets[:, :min_length, :]
                
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                total_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / total_batches
        return avg_val_loss

    def predict(self, data):
        """Make predictions on new data."""
        self.model.eval()
        
        print("\n" + "="*80)
        print("PREDICTION DEBUG INFORMATION:")
        print(f"Using device: {self.device}")
        print(f"Data keys: {list(data.keys())}")
        for station_id in data.keys():
            print(f"Station {station_id} features: {list(data[station_id].keys())}")
            for feature in data[station_id].keys():
                feature_data = data[station_id][feature]
                if isinstance(feature_data, pd.Series):
                    print(f"  {feature}: {len(feature_data)} points, range: {feature_data.index.min()} to {feature_data.index.max()}")
                elif isinstance(feature_data, pd.DataFrame):
                    print(f"  {feature} (DataFrame): {len(feature_data)} points, range: {feature_data.index.min()} to {feature_data.index.max()}")
        print("="*80 + "\n")
        
        # Get station and target info
        station_id = list(data.keys())[0]
        target_feature = self.config.get('output_features', ['vst_raw'])[0]
        
        # Create a copy of the data to avoid modifying the original
        test_data_prediction = {station_id: {}}
        for feature in data[station_id]:
            test_data_prediction[station_id][feature] = data[station_id][feature].copy()
        
        # Ensure target_series is a Series
        target_series = test_data_prediction[station_id][target_feature]
        if isinstance(target_series, pd.DataFrame):
            if target_feature in target_series.columns:
                target_series = target_series[target_feature]
            elif len(target_series.columns) == 1:
                target_series = target_series.iloc[:, 0]
            else:
                print(f"Warning: Cannot convert {target_feature} DataFrame to Series. Using first column.")
                target_series = target_series.iloc[:, 0]
            test_data_prediction[station_id][target_feature] = target_series
        
        # Prepare data for prediction
        X, _ = self.prepare_data(test_data_prediction, is_training=False)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X.to(self.device))
            predictions = predictions.cpu().numpy()
        
        # Inverse transform predictions
        target_min = self.target_scaler['min']
        target_max = self.target_scaler['max']
        unscaled_predictions = predictions * (target_max - target_min) + target_min
        
        print(f"Generated predictions shape: {unscaled_predictions.shape}")
        if unscaled_predictions.size > 0:
            print(f"Prediction range: [{unscaled_predictions.min():.2f}, {unscaled_predictions.max():.2f}]")
            if isinstance(target_series, pd.Series):
                print(f"Actual data range: [{target_series.min():.2f}, {target_series.max():.2f}]")
        
        # Reshape to 1D array
        predictions_flat = unscaled_predictions.reshape(-1)
        
        # Create a Series with the correct index
        if len(predictions_flat) > 0:
            if hasattr(self, 'data_index') and len(self.data_index) == len(predictions_flat):
                predictions_series = pd.Series(predictions_flat, index=self.data_index)
                print(f"Created predictions series with {len(predictions_series)} points")
                print(f"Predictions range from {predictions_series.index.min()} to {predictions_series.index.max()}")
                return predictions_series
            else:
                print(f"Warning: Cannot align predictions with data_index. Using target data index.")
                if len(target_series) >= len(predictions_flat):
                    predictions_index = target_series.index[:len(predictions_flat)]
                    predictions_series = pd.Series(predictions_flat, index=predictions_index)
                    print(f"Created predictions series with {len(predictions_series)} points")
                    return predictions_series
                else:
                    print(f"Warning: Target data length ({len(target_series)}) is less than predictions length ({len(predictions_flat)})")
        
        return predictions_flat
        
    #def _apply_postprocessing(self, predictions_series, target_data):
        """Enhanced post-processing with stronger smoothing."""
        print("\nApplying enhanced post-processing for smoother predictions...")
        
        # 1. Ensure predictions are within reasonable bounds
        min_allowed = max(0, target_data.min() * 0.8)  # Allow 20% below min but not negative
        max_allowed = target_data.max() * 1.2  # Allow 20% above max
        predictions_series = predictions_series.clip(min_allowed, max_allowed)
        print(f"  Clipped predictions to range: [{min_allowed:.2f}, {max_allowed:.2f}]")
        
        # 2. Apply initial exponential smoothing
        # Use larger alpha for more smoothing (smaller alpha = more smoothing)
        smoothed = predictions_series.ewm(alpha=0.15, adjust=False).mean()
        
        # 3. Detect and handle outliers using rolling statistics
        window_size = 96  # 24 hours (assuming 15-min intervals)
        rolling_median = smoothed.rolling(window=window_size, center=True).median()
        rolling_std = smoothed.rolling(window=window_size, center=True).std()
        
        # Fill NaN values at the edges
        rolling_median = rolling_median.fillna(method='ffill').fillna(method='bfill')
        rolling_std = rolling_std.fillna(method='ffill').fillna(method='bfill')
        
        # More conservative outlier threshold
        outlier_threshold = 2.5  # Further reduced from 3.0
        outliers = (smoothed < (rolling_median - outlier_threshold * rolling_std)) | \
                  (smoothed > (rolling_median + outlier_threshold * rolling_std))
        
        # Replace outliers with the median value
        if outliers.sum() > 0:
            print(f"  Detected and fixed {outliers.sum()} outliers")
            smoothed[outliers] = rolling_median[outliers]
        
        # 4. Apply final adaptive smoothing
        # Calculate local volatility
        volatility = smoothed.diff().abs().rolling(window=48).mean() / smoothed.rolling(window=48).mean()
        
        # Initialize final smoothed series
        final_smoothed = pd.Series(index=smoothed.index, dtype=float)
        
        # Apply different smoothing based on volatility
        high_vol_mask = volatility > 0.03  # Reduced threshold
        med_vol_mask = (volatility <= 0.03) & (volatility > 0.01)
        low_vol_mask = volatility <= 0.01
        
        # Apply EWM with different alphas based on volatility
        final_smoothed[high_vol_mask] = smoothed[high_vol_mask].ewm(alpha=0.2).mean()
        final_smoothed[med_vol_mask] = smoothed[med_vol_mask].ewm(alpha=0.15).mean()
        final_smoothed[low_vol_mask] = smoothed[low_vol_mask].ewm(alpha=0.1).mean()
        
        # Fill any remaining NaN values
        final_smoothed = final_smoothed.fillna(method='ffill').fillna(method='bfill')
        
        print("  Applied adaptive exponential smoothing")
        return final_smoothed

def create_full_plot(test_data, test_predictions, station_id):
    """Create an interactive plot comparing actual and predicted values."""
    # Extract the target data
    test_actual = test_data[station_id]['vst_raw']
    
    # Convert DataFrame to Series if needed
    if isinstance(test_actual, pd.DataFrame):
        if 'vst_raw' in test_actual.columns:
            test_actual = test_actual['vst_raw']
        elif len(test_actual.columns) == 1:
            test_actual = test_actual.iloc[:, 0]
        else:
            print("Error: Cannot determine which column to use from DataFrame")
            return
    
    # Handle predictions
    if not isinstance(test_predictions, pd.Series):
        # Convert numpy array to Series
        if len(test_predictions) > 0:
            # Create pandas Series for predictions, handling potential length mismatch
            pred_length = min(len(test_actual), len(test_predictions))
            predictions_index = test_actual.index[:pred_length]
            test_predictions = pd.Series(test_predictions[:pred_length], index=predictions_index, name='Predictions')
        else:
            print("Warning: Test predictions array is empty")
            return
    
    # Print data summary
    print(f"\nPlotting data summary:")
    print(f"  Actual data: {len(test_actual)} points, range: [{test_actual.min():.2f}, {test_actual.max():.2f}]")
    print(f"  Predictions: {len(test_predictions)} points, range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=test_actual.index, y=test_actual, name="Actual", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_predictions.index, y=test_predictions, name="Predicted", line=dict(color='red')))
    
    # Update layout
    fig.update_layout(
        title=f'Water Level - Actual vs Predicted (Station {station_id})',
        xaxis_title='Time',
        yaxis_title='Water Level',
        width=1200,
        height=600,
        showlegend=True
    )
    
    # Ensure the x-axis range covers the full data period
    fig.update_xaxes(range=[test_actual.index.min(), test_actual.index.max()])
    
    # Save and open plot
    html_path = 'predictions_plot.html'
    fig.write_html(html_path)
    print(f"Plot saved to: {os.path.abspath(html_path)}")
    webbrowser.open('file://' + os.path.abspath(html_path))
