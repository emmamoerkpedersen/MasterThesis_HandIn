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

'''
Remove clear_gpu_memory and MemoryEfficientDataset

Try and run code without the cos and sin features
Add rate of change features and other features to ensure that the model captures peaks
Data leakage?
Remove cuda code from main.py and lstm_forecaster.py
'''

class SimpleLSTMModel(nn.Module):
    """
    Simple LSTM model for time series forecasting.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output features
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(SimpleLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.model_name = 'SimpleLSTM'
        
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_size)
        
        # Input projection to match hidden size for residual connections
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization after each LSTM
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection with multiple layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device and print status
        self.to(self.device)
        print(f"Model using device: {self.device}")
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Orthogonal initialization for LSTM weights
                    if param.dim() > 1:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.1)
                else:
                    # Kaiming initialization for linear layers
                    if param.dim() > 1:
                        nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
                    else:
                        # For 1D tensors, use normal initialization
                        nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Input normalization
        x = self.input_norm(x)
        
        # Project input to hidden size
        h = self.input_projection(x)
        
        # Process through LSTM layers with residual connections
        for i in range(self.num_layers):
            # LSTM layer
            lstm_out, _ = self.lstm_layers[i](h)
            
            # Residual connection
            h = h + lstm_out
            
            # Layer normalization and dropout
            h = self.layer_norms[i](h)
            h = self.dropouts[i](h)
        
        # Self-attention mechanism
        attn_out, _ = self.attention(h, h, h)
        
        # Residual connection after attention
        h = h + attn_out
        
        # Final output projection
        output = self.output_projection(h)
        
        return output

class train_LSTM:
    def __init__(self, model, config):
        """Initialize the trainer with model and configuration."""
        self.model = model
        self.config = config
        
        # Get device from model
        self.device = model.device
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # Add learning rate scheduler with warmup
        self.warmup_steps = 100  # Number of warmup steps
        self.scheduler = self.get_lr_scheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_lr=config.get('learning_rate', 0.001),
            patience=5
        )
        
        # Use custom loss function with smoothness penalty
        self.smoothness_weight = config.get('smoothness_weight', 0.1)
        self.criterion = self.smooth_mse_loss
        
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
        print(f"  Smoothness weight: {self.smoothness_weight}")
        print(f"  Features: {len(self.feature_names)} input features")

    def smooth_mse_loss(self, y_pred, y_true):
        """Enhanced loss function with increased focus on smoothness."""
        # Standard MSE loss
        mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        
        # Calculate rolling statistics for peak detection
        if y_true.shape[1] > 24:  # Only if we have enough points
            rolling_mean = torch.mean(y_true.squeeze(), dim=0)
            rolling_std = torch.std(y_true.squeeze(), dim=0)
            
            # Identify peaks (points > 2 standard deviations from mean)
            peaks = (y_true.squeeze() > (rolling_mean + 2 * rolling_std)).float()
            
            # Create weights (3x importance for peaks - reduced from 5x)
            weights = 1.0 + 2.0 * peaks
            
            # Weighted MSE loss for peaks
            squared_errors = (y_pred - y_true) ** 2
            peak_loss = torch.mean(weights * squared_errors.squeeze())
        else:
            peak_loss = mse_loss
        
        # Enhanced gradient matching and smoothness components
        if y_pred.shape[1] > 1:
            # First-order gradient matching (rate of change)
            pred_grads = y_pred[:, 1:] - y_pred[:, :-1]
            true_grads = y_true[:, 1:] - y_true[:, :-1]
            gradient_loss = torch.nn.functional.mse_loss(pred_grads, true_grads)
            
            # Second-order gradient (acceleration) for additional smoothness
            pred_grads2 = pred_grads[:, 1:] - pred_grads[:, :-1]
            true_grads2 = true_grads[:, 1:] - true_grads[:, :-1]
            acceleration_loss = torch.nn.functional.mse_loss(pred_grads2, true_grads2)
            
            # Enhanced smoothness penalty
            smoothness_loss = (
                torch.mean(torch.square(pred_grads)) +  # First-order smoothness
                2.0 * torch.mean(torch.square(pred_grads2))  # Second-order smoothness
            )
            
            # Combine all losses with adjusted weights
            total_loss = (
                0.25 * mse_loss +          # Base MSE (reduced)
                0.15 * peak_loss +         # Peak accuracy (reduced)
                0.20 * gradient_loss +     # Rate of change
                0.15 * acceleration_loss + # Acceleration matching (new)
                0.25 * self.smoothness_weight * smoothness_loss  # Increased smoothness penalty
            )
            
            # Log components for debugging
            if torch.isnan(total_loss).any():
                print(f"WARNING: NaN in loss calculation!")
                print(f"MSE: {mse_loss.item()}, Peak: {peak_loss.item()}, Gradient: {gradient_loss.item()}, Smoothness: {smoothness_loss.item()}")
                return mse_loss  # Fallback to standard MSE if NaN detected
                
            return total_loss
        else:
            # If sequence length is 1, just return combined MSE and peak loss
            return 0.7 * mse_loss + 0.3 * peak_loss

    def get_lr_scheduler(self, optimizer, warmup_steps=100, max_lr=0.001, patience=5):
        """Create a learning rate scheduler with warmup."""
        # Define a custom learning rate lambda function
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0.1 * max_lr to max_lr
                return 0.1 + 0.9 * float(step) / float(max(1, warmup_steps))
            else:
                # After warmup, use a constant learning rate
                return 1.0
        
        # Create a LambdaLR scheduler for warmup
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Create a ReduceLROnPlateau scheduler for after warmup
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=patience,
            verbose=True,
            min_lr=1e-6
        )
        
        # Return a dictionary with both schedulers
        return {
            'warmup': warmup_scheduler,
            'plateau': plateau_scheduler,
            'current_step': 0
        }

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
            
            # Define derived feature types - REMOVED hourly and minute features
            time_features = [
                'month_sin', 'month_cos',
                'day_sin', 'day_cos', 
                'weekday_sin', 'weekday_cos'
            ]
            
            lagged_features = [
                f'{target_feature}_lag_96', 
                f'{target_feature}_lag_672',
                f'{target_feature}_rolling_mean_96', 
                f'{target_feature}_rolling_std_96',
                f'{target_feature}_diff_96',
                f'{target_feature}_diff_4',
                f'{target_feature}_diff_pct_96',
                f'{target_feature}_ewm_alpha_30',
                f'{target_feature}_ewm_alpha_70'
            ]
            derived_features = time_features + lagged_features
            
            # Filter out derived features from base features
            actual_base_features = [f for f in base_features if f not in derived_features]
            
            # Add actual base features from input data
            for feature in actual_base_features:
                if feature in station_data and isinstance(station_data[feature], pd.Series):
                    all_data[feature] = station_data[feature]
                else:
                    print(f"Warning: Base feature {feature} not found in data. Using zeros.")
                    # Create a Series of zeros with the same index as the target
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
            
            # Add time features - REMOVED hourly and minute features
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
            
            # Add enhanced lagged features for the target
            if target_feature in all_data.columns:
                print("Adding lagged and statistical features...")
                # Basic lag features
                all_data[f'{target_feature}_lag_96'] = all_data[target_feature].shift(96)  # 1 day (24 hours * 4)
                all_data[f'{target_feature}_lag_672'] = all_data[target_feature].shift(672)  # 1 week (168 hours * 4)
                
                # Rolling statistics
                all_data[f'{target_feature}_rolling_mean_96'] = all_data[target_feature].rolling(96).mean()
                all_data[f'{target_feature}_rolling_std_96'] = all_data[target_feature].rolling(96).std()
                
                # Differencing features (rate of change)
                all_data[f'{target_feature}_diff_96'] = all_data[target_feature].diff(96)  # Change over 1 day
                all_data[f'{target_feature}_diff_4'] = all_data[target_feature].diff(4)    # Change over 1 hour
                
                # Percentage change features
                all_data[f'{target_feature}_diff_pct_96'] = all_data[target_feature].pct_change(96)
                
                # Exponential weighted means (different smoothing factors)
                all_data[f'{target_feature}_ewm_alpha_30'] = all_data[target_feature].ewm(alpha=0.3).mean()
                all_data[f'{target_feature}_ewm_alpha_70'] = all_data[target_feature].ewm(alpha=0.7).mean()
                
                # Only update feature_cols during training
                if is_training:
                    # Define optimized feature set - ensure no duplicates
                    optimized_features = actual_base_features.copy()  # Start with actual base features
                    
                    # Add time features if not already present
                    for feature in time_features:
                        if feature not in optimized_features:
                            optimized_features.append(feature)
                    
                    # Add lagged features if not already present
                    for feature in lagged_features:
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
            
            # Initialize or update scalers during training
            if is_training and not self.is_fitted:
                self.scalers = {}
                for col in features.columns:
                    # Handle both Series and DataFrame columns
                    if isinstance(features[col], pd.Series):
                        feature_values = features[col]
                    else:
                        # If it's a DataFrame, take the first column
                        print(f"Warning: Feature {col} is a DataFrame during scaler initialization. Using first column.")
                        feature_values = features[col].iloc[:, 0]
                    
                    # Use robust scaling for each feature
                    q1 = feature_values.quantile(0.05)  # 5th percentile
                    q3 = feature_values.quantile(0.95)  # 95th percentile
                    iqr = q3 - q1
                    median = feature_values.median()
                    self.scalers[col] = {
                        'median': float(median),
                        'iqr': float(iqr) if iqr != 0 else 1.0,
                        'min': float(feature_values.min()),
                        'max': float(feature_values.max())
                    }
                
                # Use min-max scaling for target with padding
                target_min = float(target_data.min())
                target_max = float(target_data.max())
                target_range = target_max - target_min
                # Add 50% padding to capture future extremes
                self.target_scaler = {
                    'min': target_min - 0.5 * target_range,
                    'max': target_max + 0.5 * target_range
                }
                print(f"Target scaling range: [{self.target_scaler['min']:.2f}, {self.target_scaler['max']:.2f}]")
                self.is_fitted = True
            elif not self.is_fitted:
                # If we're not in training mode but scalers aren't initialized, create default scalers
                print("Warning: Scalers not initialized. Creating default scalers.")
                self.scalers = {}
                for col in features.columns:
                    self.scalers[col] = {
                        'median': 0.0,
                        'iqr': 1.0,
                        'min': -1.0,
                        'max': 1.0
                    }
                
                # Create a default target scaler with a very wide range
                self.target_scaler = {
                    'min': 0.0,
                    'max': 2000.0  # Increased from 1000.0 to handle higher water levels
                }
                self.is_fitted = True
            
            # Scale features
            scaled_features = np.zeros((len(features), len(features.columns)))
            for i, col in enumerate(features.columns):
                if col in self.scalers:
                    # Use the appropriate scaling method for each feature
                    median = self.scalers[col]['median']
                    iqr = self.scalers[col]['iqr']
                    
                    # Handle both Series and DataFrame columns
                    if isinstance(features[col], pd.Series):
                        feature_values = features[col].values
                    else:
                        # If it's a DataFrame, take the first column
                        print(f"Warning: Feature {col} is a DataFrame. Using first column.")
                        feature_values = features[col].iloc[:, 0].values
                    
                    scaled_features[:, i] = (feature_values - median) / iqr
                else:
                    print(f"Warning: No scaler found for {col}. Using standard scaling.")
                    # Fallback to standard scaling
                    if isinstance(features[col], pd.Series):
                        feature_values = features[col].values
                        mean_val = features[col].mean()
                        std_val = features[col].std()
                    else:
                        # If it's a DataFrame, take the first column
                        print(f"Warning: Feature {col} is a DataFrame. Using first column for standard scaling.")
                        feature_values = features[col].iloc[:, 0].values
                        mean_val = features[col].iloc[:, 0].mean()
                        std_val = features[col].iloc[:, 0].std()
                    
                    if std_val == 0 or pd.isna(std_val):
                        std_val = 1.0
                    scaled_features[:, i] = (feature_values - mean_val) / std_val
            
            # Scale target
            target_min = self.target_scaler['min']
            target_max = self.target_scaler['max']
            scaled_target_data = (target_data.values - target_min) / (target_max - target_min)
            
            # Create single sequence and convert to tensors
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

    def _create_data_loaders(self, X, y, batch_size, max_chunk_size):
        """Create data loaders, chunking if necessary."""
        loaders = []
        
        # Get max_chunk_size from config or use default
        max_chunk_size = self.config.get('max_chunk_size', max_chunk_size)
        
        # Move tensors to CPU for data loading
        X = X.cpu()
        y = y.cpu()
        
        if X.shape[1] > max_chunk_size:
            # Need chunking
            seq_len = X.shape[1]
            num_chunks = (seq_len + max_chunk_size - 1) // max_chunk_size
            
            for i in range(num_chunks):
                start_idx = i * max_chunk_size
                end_idx = min((i + 1) * max_chunk_size, seq_len)
                
                # Extract chunk
                X_chunk = X[:, start_idx:end_idx, :].contiguous()
                y_chunk = y[:, start_idx:end_idx, :].contiguous()
                
                # Create dataset and loader
                dataset = TensorDataset(X_chunk, y_chunk)
                loader = DataLoader(
                    dataset, 
                    batch_size=1,  # Keep batch_size=1 for sequence prediction
                    shuffle=False,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                loaders.append(loader)
                
                # Clear some memory
                del X_chunk, y_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # No chunking needed
            X = X.contiguous()
            y = y.contiguous()
            dataset = TensorDataset(X, y)
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True if torch.cuda.is_available() else False
            )
            loaders.append(loader)
        
        return loaders

    def train(self, train_data, val_data, epochs=100, batch_size=1, patience=15):
        """Train the model with sequence-to-sequence approach."""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, is_training=True)
        X_val, y_val = self.prepare_data(val_data, is_training=False)
        
        # Get chunk size from config
        max_chunk_size = self.config.get('max_chunk_size', 2500)
        need_chunking = X_train.shape[1] > max_chunk_size
        
        if need_chunking:
            print(f"\nSequence length ({X_train.shape[1]}) exceeds maximum chunk size ({max_chunk_size})")
            print(f"Will process data in chunks")
            
            # Calculate number of chunks
            seq_len = X_train.shape[1]
            num_chunks = (seq_len + max_chunk_size - 1) // max_chunk_size
            print(f"Data will be processed in {num_chunks} chunks")
        
        # Create train and validation data loaders
        train_loaders = self._create_data_loaders(X_train, y_train, batch_size=1, max_chunk_size=max_chunk_size)
        val_loaders = self._create_data_loaders(X_val, y_val, batch_size=1, max_chunk_size=max_chunk_size)
        
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
        
        # Reset scheduler step counter
        self.scheduler['current_step'] = 0
        
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
                
                # Process each chunk
                for chunk_idx, train_loader in enumerate(train_loaders):
                    chunk_loss = 0
                    chunk_batches = 0
                    
                    # Process each batch in the chunk
                    for batch_idx, (data, targets) in enumerate(train_loader):
                        try:
                            # Move data to device
                            data = data.to(self.device, non_blocking=True)
                            targets = targets.to(self.device, non_blocking=True)
                            
                            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                            
                            # Forward pass with gradient computation
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
                            
                            # Update the warmup scheduler after each batch
                            if self.scheduler['current_step'] < self.warmup_steps:
                                self.scheduler['warmup'].step()
                                self.scheduler['current_step'] += 1
                                
                            chunk_loss += loss.item()
                            chunk_batches += 1
                            
                            # Clear memory
                            del data, targets, outputs, loss
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"WARNING: Out of memory in batch. Skipping batch and clearing cache.")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                    
                    if chunk_batches > 0:
                        avg_chunk_loss = chunk_loss / chunk_batches
                        train_loss += avg_chunk_loss * chunk_batches
                        total_batches += chunk_batches
                        
                        if need_chunking:
                            print(f"  Chunk {chunk_idx+1}/{len(train_loaders)} - Loss: {avg_chunk_loss:.6f}")
                
                # Calculate average training loss
                avg_train_loss = train_loss / max(1, total_batches)
                
                # Validate
                val_loss = self._validate_epoch(val_loaders)
                
                # Update learning rate scheduler
                self.scheduler['plateau'].step(val_loss)
                
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
                
                # Clear some memory at the end of each epoch
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

    def _validate_epoch(self, val_loaders):
        """Validate the model on all validation chunks."""
        self.model.eval()
        val_loss = 0
        total_val_batches = 0
        
        with torch.no_grad():
            # Process each validation chunk
            for chunk_idx, val_loader in enumerate(val_loaders):
                chunk_val_loss = 0
                chunk_val_batches = 0
                
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
                    
                    chunk_val_loss += loss.item()
                    chunk_val_batches += 1
                
                if chunk_val_batches > 0:
                    val_loss += chunk_val_loss
                    total_val_batches += chunk_val_batches
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, total_val_batches)
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
        
        # Check if we need to add lagged features to the test data
        station_id = list(data.keys())[0]
        target_feature = self.config.get('output_features', ['vst_raw'])[0]
        
        # Create a copy of the data to avoid modifying the original
        test_data = {station_id: {}}
        for feature in data[station_id]:
            test_data[station_id][feature] = data[station_id][feature].copy()
        
        # Ensure target_series is a Series
        target_series = test_data[station_id][target_feature]
        if isinstance(target_series, pd.DataFrame):
            if target_feature in target_series.columns:
                target_series = target_series[target_feature]
            elif len(target_series.columns) == 1:
                target_series = target_series.iloc[:, 0]
            else:
                print(f"Warning: Cannot convert {target_feature} DataFrame to Series. Using first column.")
                target_series = target_series.iloc[:, 0]
            test_data[station_id][target_feature] = target_series
        
        # Use the modified data for prediction
        X, _ = self.prepare_data(test_data, is_training=False)
        
        if X.shape[0] == 0:
            print("No data to predict on!")
            return np.array([])
        
        # Check if sequence is too long and needs chunking
        max_chunk_size = 5000
        need_chunking = X.shape[1] > max_chunk_size
        
        if need_chunking:
            print(f"Processing long sequence in chunks (length: {X.shape[1]})")
            
            # Calculate number of chunks
            seq_len = X.shape[1]
            num_chunks = (seq_len + max_chunk_size - 1) // max_chunk_size
            
            # Initialize array for predictions
            all_predictions = np.zeros((X.shape[0], seq_len, 1))
            
            # Process each chunk
            with torch.no_grad():
                for i in range(num_chunks):
                    start_idx = i * max_chunk_size
                    end_idx = min((i + 1) * max_chunk_size, seq_len)
                    
                    # Extract and process chunk
                    X_chunk = X[:, start_idx:end_idx, :]
                    chunk_predictions = self.model(X_chunk)
                    chunk_predictions = chunk_predictions.cpu().numpy()
                    
                    # Store predictions
                    all_predictions[:, start_idx:end_idx, :] = chunk_predictions
            
            # Use combined predictions
            predictions = all_predictions
        else:
            # Make predictions on the entire sequence at once
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
            # Use the stored index from prepare_data
            if hasattr(self, 'data_index') and len(self.data_index) == len(predictions_flat):
                predictions_series = pd.Series(predictions_flat, index=self.data_index)
                print(f"Created predictions series with {len(predictions_series)} points")
                print(f"Predictions range from {predictions_series.index.min()} to {predictions_series.index.max()}")
                
                # Apply post-processing to improve predictions
                predictions_series = self._apply_postprocessing(predictions_series, target_series)
                return predictions_series
            else:
                print(f"Warning: Cannot align predictions with data_index. Using target data index.")
                # Try to align with target data
                if len(target_series) >= len(predictions_flat):
                    predictions_index = target_series.index[:len(predictions_flat)]
                    predictions_series = pd.Series(predictions_flat, index=predictions_index)
                    print(f"Created predictions series with {len(predictions_series)} points")
                    return predictions_series
                else:
                    print(f"Warning: Target data length ({len(target_series)}) is less than predictions length ({len(predictions_flat)})")
        
        return predictions_flat
        
    def _apply_postprocessing(self, predictions_series, target_data):
        """Enhanced post-processing with stronger smoothing."""
        print("\nApplying enhanced post-processing for smoother predictions...")
        
        # 1. Ensure predictions are within reasonable bounds
        min_allowed = max(0, target_data.min() * 0.8)  # Allow 20% below min but not negative
        max_allowed = target_data.max() * 1.2  # Allow 20% above max
        predictions_series = predictions_series.clip(min_allowed, max_allowed)
        print(f"  Clipped predictions to range: [{min_allowed:.2f}, {max_allowed:.2f}]")
        
        # 2. Enhanced outlier detection and correction
        # Use larger windows for more stable statistics
        rolling_median = predictions_series.rolling(window=48, center=True).median()  # Increased from 24
        rolling_std = predictions_series.rolling(window=96, center=True).std()  # Increased from 48
        
        # Fill NaN values at the edges
        rolling_median = rolling_median.fillna(method='ffill').fillna(method='bfill')
        rolling_std = rolling_std.fillna(method='ffill').fillna(method='bfill')
        
        # More conservative outlier threshold
        outlier_threshold = 3.0  # Reduced from 4.0
        outliers = (predictions_series < (rolling_median - outlier_threshold * rolling_std)) | \
                  (predictions_series > (rolling_median + outlier_threshold * rolling_std))
        
        # Replace outliers with the median value
        if outliers.sum() > 0:
            print(f"  Detected and fixed {outliers.sum()} outliers")
            predictions_series[outliers] = rolling_median[outliers]
        
        # 3. Apply adaptive smoothing based on local volatility
        volatility = predictions_series.diff().abs().rolling(window=24).mean() / predictions_series.rolling(window=24).mean()
        
        # Apply stronger smoothing to high-volatility regions
        smoothed = pd.Series(index=predictions_series.index, dtype=float)
        
        # Different smoothing windows based on volatility
        high_vol_mask = volatility > 0.05
        med_vol_mask = (volatility <= 0.05) & (volatility > 0.02)
        low_vol_mask = volatility <= 0.02
        
        # Apply different smoothing windows based on volatility
        smoothed[high_vol_mask] = predictions_series[high_vol_mask].rolling(window=7, center=True).mean()
        smoothed[med_vol_mask] = predictions_series[med_vol_mask].rolling(window=5, center=True).mean()
        smoothed[low_vol_mask] = predictions_series[low_vol_mask].rolling(window=3, center=True).mean()
        
        # Fill any remaining NaN values
        smoothed = smoothed.fillna(method='ffill').fillna(method='bfill')
        
        print("  Applied adaptive smoothing based on local volatility")
        return smoothed

def create_full_plot(test_data, test_predictions, station_id):
    """Create an interactive plot comparing actual and predicted values."""
    # Extract the target data
    test_actual = test_data['vst_raw']
    
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
