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
import gc
from sklearn.preprocessing import MinMaxScaler
import torch.utils.checkpoint as checkpoint

# Function to clear GPU memory
def clear_gpu_memory():
    """Clear GPU memory to avoid memory fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Memory-efficient dataset class
class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand instead of all at once."""
    
    def __init__(self, X, y=None):
        """
        Initialize the dataset.
        
        Args:
            X: Input features tensor
            y: Target tensor (optional)
        """
        # Ensure tensors are on CPU for proper pin_memory operation
        if isinstance(X, torch.Tensor) and X.is_cuda:
            self.X = X.cpu()
        else:
            self.X = X
            
        if y is not None and isinstance(y, torch.Tensor) and y.is_cuda:
            self.y = y.cpu()
        else:
            self.y = y
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

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
        self.model_name = 'SimpleLSTM'  # Add model_name attribute
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Simplified output projection to save memory
        self.hidden_to_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Enable gradient checkpointing for memory efficiency
        self.use_checkpointing = True
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Use orthogonal initialization for LSTM weights
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.1)
                else:
                    # Use Xavier/Glorot initialization for linear layers
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                # Initialize biases to small positive values
                nn.init.constant_(param, 0.01)
    
    def forward(self, x):
        # Check input dimensions
        batch_size, seq_len, features = x.size()
        
        # Check if input features match expected size
        if features != self.input_size:
            # Dynamically adjust the input normalization layer if needed
            print(f"Warning: Input features ({features}) don't match expected size ({self.input_size})")
            print(f"Adjusting input normalization layer to match input size")
            self.input_norm = nn.LayerNorm(features)
            self.input_size = features
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Apply dropout to input
        x = self.dropout(x)
        
        # Memory-efficient processing for very long sequences
        if seq_len > 1000 and self.training:
            # Process in smaller sub-sequences to save memory
            sub_seq_len = 1000
            num_sub_seqs = (seq_len + sub_seq_len - 1) // sub_seq_len
            outputs = []
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
            
            # Process each sub-sequence
            for i in range(num_sub_seqs):
                start_idx = i * sub_seq_len
                end_idx = min((i + 1) * sub_seq_len, seq_len)
                
                # Extract sub-sequence
                sub_x = x[:, start_idx:end_idx, :]
                
                # Process with or without checkpointing
                if self.training and self.use_checkpointing:
                    # Define custom forward function for checkpointing
                    def create_custom_forward(module, hidden_state):
                        def custom_forward(inputs):
                            return module(inputs, hidden_state)[0]
                        return custom_forward
                    
                    # Apply LSTM with checkpointing
                    sub_output = checkpoint.checkpoint(
                        create_custom_forward(self.lstm, hidden),
                        sub_x
                    )
                    
                    # Update hidden state (without gradient tracking)
                    with torch.no_grad():
                        _, hidden = self.lstm(sub_x, hidden)
                else:
                    # Regular forward pass
                    sub_output, hidden = self.lstm(sub_x, hidden)
                
                # Collect output
                outputs.append(sub_output)
            
            # Concatenate outputs
            output = torch.cat(outputs, dim=1)
        else:
            # Use gradient checkpointing during training to save memory
            if self.training and self.use_checkpointing:
                # Define custom forward function for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Apply LSTM with checkpointing
                output, _ = checkpoint.checkpoint(
                    create_custom_forward(self.lstm),
                    x
                )
            else:
                # Regular forward pass
                output, _ = self.lstm(x)
        
        # Apply dropout to LSTM output
        output = self.dropout(output)
        
        # Apply output projection
        output = self.hidden_to_output(output)
        
        return output

class train_LSTM:
    def __init__(self, model, config):
        """Initialize the trainer with model and configuration."""
        self.model = model
        self.config = config
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
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
        
        self.criterion = nn.MSELoss()
        
        # Initialize scalers
        self.scalers = {}
        self.target_scaler = None
        self.is_fitted = False
        
        # Initialize mixed precision training if CUDA is available
        self.use_mixed_precision = torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training to reduce memory usage")
            
        # Store feature names for prediction
        self.feature_names = config.get('feature_cols', [])
        self.target_dim = 1  # Default target dimension

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
            return torch.FloatTensor([]).to(self.device), torch.FloatTensor([]).to(self.device)
            
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
            
            # Add base features (temperature, rainfall)
            base_features = self.config['feature_cols']
            
            for feature in base_features:
                if feature in station_data and isinstance(station_data[feature], pd.Series):
                    all_data[feature] = station_data[feature]
                else:
                    print(f"Warning: Base feature {feature} not found in data")
                    # Instead of just warning, check if this is a derived feature we can create later
                    if feature not in ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 
                                      'month_sin', 'month_cos', 'vst_raw_lag_96', 'vst_raw_lag_672',
                                      'vst_raw_rolling_mean_96', 'vst_raw_diff_96']:
                        print(f"  This is a required base feature and cannot be derived. Using zeros.")
                        # Create a Series of zeros with the same index as the target
                        if target_feature in station_data and isinstance(station_data[target_feature], pd.Series):
                            all_data[feature] = pd.Series(0.0, index=station_data[target_feature].index)
            
            # Add target feature
            if target_feature in station_data and isinstance(station_data[target_feature], pd.Series):
                all_data[target_feature] = station_data[target_feature]
            else:
                print(f"ERROR: Target feature {target_feature} not found in data!")
                return torch.FloatTensor([]).to(self.device), torch.FloatTensor([]).to(self.device)
            
            # Resample to hourly frequency
            all_data = all_data.resample('15T').mean().interpolate(method='time').ffill().bfill()
            
            # Add enhanced time features
            # Hour of day (cyclical)
            all_data['hour_sin'] = np.sin(2 * np.pi * all_data.index.hour / 24)
            all_data['hour_cos'] = np.cos(2 * np.pi * all_data.index.hour / 24)
            
            # Minute of hour (cyclical) - new feature for 15-minute data
            all_data['minute_sin'] = np.sin(2 * np.pi * all_data.index.minute / 60)
            all_data['minute_cos'] = np.cos(2 * np.pi * all_data.index.minute / 60)
            
            # Month of year (cyclical)
            all_data['month_sin'] = np.sin(2 * np.pi * all_data.index.month / 12)
            all_data['month_cos'] = np.cos(2 * np.pi * all_data.index.month / 12)
            
            # Add enhanced lagged features for the target
            if target_feature in all_data.columns:
                # Add multiple lag features - adjusted for 15-minute data (4 points per hour)
                all_data[f'{target_feature}_lag_96'] = all_data[target_feature].shift(96)  # 1 day (24 hours * 4)
                all_data[f'{target_feature}_lag_672'] = all_data[target_feature].shift(672)  # 1 week (168 hours * 4)
                all_data[f'{target_feature}_lag_1344'] = all_data[target_feature].shift(1344)  # 2 weeks (168 hours * 8)
                all_data[f'{target_feature}_lag_2688'] = all_data[target_feature].shift(2688)  # 4 weeks (168 hours * 16)
                # Add rolling statistics - adjusted for 15-minute data
                all_data[f'{target_feature}_rolling_mean_96'] = all_data[target_feature].rolling(96).mean()
                
                # Add differencing features - adjusted for 15-minute data
                all_data[f'{target_feature}_diff_96'] = all_data[target_feature].diff(96)
                
                # Only update feature_cols during training
                if is_training:
                    # Define optimized feature set - ensure no duplicates
                    optimized_features = base_features.copy()  # Start with a copy to avoid modifying the original
                    
                    # Add time features if not already present
                    time_features = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'month_sin', 'month_cos']
                    for feature in time_features:
                        if feature not in optimized_features:
                            optimized_features.append(feature)
                    
                    # Add lagged features if not already present
                    lagged_features = [
                        f'{target_feature}_lag_96', 
                        f'{target_feature}_lag_672',
                        f'{target_feature}_rolling_mean_96', 
                        f'{target_feature}_diff_96'
                    ]
                    for feature in lagged_features:
                        if feature not in optimized_features:
                            optimized_features.append(feature)
                    
                    # Update feature_cols to include optimized features
                    self.config['feature_cols'] = optimized_features
                    
                    print(f"Using optimized feature set: {self.config['feature_cols']}")
            
            # Make sure all required features are in the DataFrame
            missing_features = [f for f in self.config['feature_cols'] if f not in all_data.columns]
            if missing_features:
                print(f"Missing features that will be created: {missing_features}")
                
                # Create missing features
                for feature in missing_features:
                    # Check if it's a time-based feature we can create
                    if feature == 'hour_sin':
                        all_data['hour_sin'] = np.sin(2 * np.pi * all_data.index.hour / 24)
                    elif feature == 'hour_cos':
                        all_data['hour_cos'] = np.cos(2 * np.pi * all_data.index.hour / 24)
                    elif feature == 'minute_sin':
                        all_data['minute_sin'] = np.sin(2 * np.pi * all_data.index.minute / 60)
                    elif feature == 'minute_cos':
                        all_data['minute_cos'] = np.cos(2 * np.pi * all_data.index.minute / 60)
                    elif feature == 'month_sin':
                        all_data['month_sin'] = np.sin(2 * np.pi * all_data.index.month / 12)
                    elif feature == 'month_cos':
                        all_data['month_cos'] = np.cos(2 * np.pi * all_data.index.month / 12)
                    # Check if it's a lag feature we can create
                    elif feature == f'{target_feature}_lag_96':
                        all_data[feature] = all_data[target_feature].shift(96)
                    elif feature == f'{target_feature}_lag_672':
                        all_data[feature] = all_data[target_feature].shift(672)
                    elif feature == f'{target_feature}_rolling_mean_96':
                        all_data[feature] = all_data[target_feature].rolling(96).mean()
                    elif feature == f'{target_feature}_diff_96':
                        all_data[feature] = all_data[target_feature].diff(96)
                    else:
                        # For any other missing features, use zeros
                        print(f"  Creating missing feature {feature} with zeros")
                        all_data[feature] = 0.0
            
            # Drop NaN values that might have been introduced by lagged features
            all_data = all_data.dropna()
            
            if len(all_data) == 0:
                print("ERROR: All data was dropped after removing NaN values!")
                return torch.FloatTensor([]).to(self.device), torch.FloatTensor([]).to(self.device)
            
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
                # Add 50% padding to capture future extremes (increased from 20%)
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
            
            # Store the index for later use in prediction
            self.data_index = target_data.index
            
            return X_tensor, y_tensor
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            import traceback
            traceback.print_exc()
            return torch.FloatTensor([]).to(self.device), torch.FloatTensor([]).to(self.device)

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        # Check if we have any batches
        if num_batches == 0:
            print("Warning: No training batches available")
            return float('inf')  # Return infinity to signal a problem
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Skip empty batches
            if data.shape[0] == 0 or targets.shape[0] == 0:
                print(f"Skipping empty batch {batch_idx}")
                continue
                
            # Move data to device (handles both CPU and CUDA)
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            
            # Ensure outputs and targets have the same shape
            if outputs.shape[1] != targets.shape[1]:
                min_length = min(outputs.shape[1], targets.shape[1])
                outputs = outputs[:, :min_length, :]
                targets = targets[:, :min_length, :]
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update the warmup scheduler after each batch
            if self.scheduler['current_step'] < self.warmup_steps:
                self.scheduler['warmup'].step()
                self.scheduler['current_step'] += 1
                
            total_loss += loss.item()
            
            if batch_idx == 0:
                print(f"First batch ranges - Predictions: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Recalculate num_batches in case we skipped some
        processed_batches = sum(1 for _, (data, _) in enumerate(train_loader) if data.shape[0] > 0)
        
        if processed_batches == 0:
            print("Warning: No valid batches were processed")
            return float('inf')
            
        return total_loss / processed_batches

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        # Return early if no validation data
        if num_batches == 0:
            print("Warning: No validation data available")
            return float('inf')  # Return infinity to trigger early stopping
        
        processed_batches = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                # Skip empty batches
                if data.shape[0] == 0 or targets.shape[0] == 0:
                    print(f"Skipping empty validation batch {batch_idx}")
                    continue
                    
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                
                # Ensure outputs and targets have the same shape
                if outputs.shape[1] != targets.shape[1]:
                    min_length = min(outputs.shape[1], targets.shape[1])
                    outputs = outputs[:, :min_length, :]
                    targets = targets[:, :min_length, :]
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                processed_batches += 1
                
                if batch_idx == 0:
                    print(f"Validation batch shapes - Input: {data.shape}, Output: {outputs.shape}, Target: {targets.shape}")
        
        if processed_batches == 0:
            print("Warning: No valid validation batches were processed")
            return float('inf')
                
        return total_loss / processed_batches

    def train(self, train_data, val_data, epochs=100, batch_size=1, patience=15):
        """Train the model with optimized process and better progress updates."""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, is_training=True)
        X_val, y_val = self.prepare_data(val_data, is_training=False)
        
        # Check if sequence is too long and needs chunking
        max_chunk_size = 5000  # Reduced from 10000 to 5000 to save memory
        need_chunking = X_train.shape[1] > max_chunk_size
        
        if need_chunking:
            print(f"\nSequence length ({X_train.shape[1]}) exceeds maximum chunk size ({max_chunk_size})")
            print(f"Will process data in chunks to save memory")
            
            # Calculate number of chunks
            seq_len = X_train.shape[1]
            num_chunks = (seq_len + max_chunk_size - 1) // max_chunk_size
            print(f"Data will be processed in {num_chunks} chunks")
            
            # Create data loaders for each chunk
            train_loaders = []
            for i in range(num_chunks):
                start_idx = i * max_chunk_size
                end_idx = min((i + 1) * max_chunk_size, seq_len)
                chunk_len = end_idx - start_idx
                
                X_train_chunk = X_train[:, start_idx:end_idx, :]
                y_train_chunk = y_train[:, start_idx:end_idx, :]
                
                # Create memory-efficient dataset
                train_dataset = MemoryEfficientDataset(X_train_chunk, y_train_chunk)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA is available
                )
                train_loaders.append(train_loader)
            
            # Create validation data loaders for each chunk
            val_loaders = []
            val_seq_len = X_val.shape[1]
            val_num_chunks = (val_seq_len + max_chunk_size - 1) // max_chunk_size
            
            for i in range(val_num_chunks):
                start_idx = i * max_chunk_size
                end_idx = min((i + 1) * max_chunk_size, val_seq_len)
                
                # Extract chunk
                X_val_chunk = X_val[:, start_idx:end_idx, :]
                y_val_chunk = y_val[:, start_idx:end_idx, :]
                
                # Create memory-efficient dataset
                val_dataset = MemoryEfficientDataset(X_val_chunk, y_val_chunk)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA is available
                )
                val_loaders.append(val_loader)
        else:
            # Create memory-efficient datasets
            train_dataset = MemoryEfficientDataset(X_train, y_train)
            val_dataset = MemoryEfficientDataset(X_val, y_val)
            
            # Create data loaders
            train_loaders = [DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA is available
            )]
            val_loaders = [DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA is available
            )]
        
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
        
        # Get model name safely
        model_name = getattr(self.model, 'model_name', type(self.model).__name__)
        print(f"Model: {model_name} with {sum(p.numel() for p in self.model.parameters())} parameters")
        
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
                    # Skip empty batches
                    if data.shape[0] == 0 or targets.shape[0] == 0:
                        continue
                        
                    # Move data to device (handles both CPU and CUDA)
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Use mixed precision for forward pass if available
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(data)
                            
                            # Ensure outputs and targets have the same shape
                            if outputs.shape[1] != targets.shape[1]:
                                min_length = min(outputs.shape[1], targets.shape[1])
                                outputs = outputs[:, :min_length, :]
                                targets = targets[:, :min_length, :]
                            
                            loss = self.criterion(outputs, targets)
                        
                        # Scale loss and do backward pass
                        self.scaler.scale(loss).backward()
                        
                        # Unscale before gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Update weights with scaled gradients
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Regular forward and backward pass
                        outputs = self.model(data)
                        
                        # Ensure outputs and targets have the same shape
                        if outputs.shape[1] != targets.shape[1]:
                            min_length = min(outputs.shape[1], targets.shape[1])
                            outputs = outputs[:, :min_length, :]
                            targets = targets[:, :min_length, :]
                        
                        loss = self.criterion(outputs, targets)
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
                
                # Clear GPU cache after each chunk
                clear_gpu_memory()
                
                if chunk_batches > 0:
                    avg_chunk_loss = chunk_loss / chunk_batches
                    train_loss += avg_chunk_loss * chunk_batches
                    total_batches += chunk_batches
                    
                    if need_chunking:
                        print(f"  Chunk {chunk_idx+1}/{len(train_loaders)} - Loss: {avg_chunk_loss:.6f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / max(1, total_batches)
            
            # Validate
            self.model.eval()
            val_loss = 0
            total_val_batches = 0
            
            with torch.no_grad():
                # Process each validation chunk
                for chunk_idx, val_loader in enumerate(val_loaders):
                    chunk_val_loss = 0
                    chunk_val_batches = 0
                    
                    for batch_idx, (data, targets) in enumerate(val_loader):
                        # Skip empty batches
                        if data.shape[0] == 0 or targets.shape[0] == 0:
                            continue
                            
                        data = data.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Use mixed precision for validation if available
                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(data)
                                
                                # Ensure outputs and targets have the same shape
                                if outputs.shape[1] != targets.shape[1]:
                                    min_length = min(outputs.shape[1], targets.shape[1])
                                    outputs = outputs[:, :min_length, :]
                                    targets = targets[:, :min_length, :]
                                
                                loss = self.criterion(outputs, targets)
                        else:
                            outputs = self.model(data)
                            
                            # Ensure outputs and targets have the same shape
                            if outputs.shape[1] != targets.shape[1]:
                                min_length = min(outputs.shape[1], targets.shape[1])
                                outputs = outputs[:, :min_length, :]
                                targets = targets[:, :min_length, :]
                            
                            loss = self.criterion(outputs, targets)
                        
                        chunk_val_loss += loss.item()
                        chunk_val_batches += 1
                    
                    # Clear GPU cache after each validation chunk
                    clear_gpu_memory()
                    
                    if chunk_val_batches > 0:
                        val_loss += chunk_val_loss
                        total_val_batches += chunk_val_batches
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(1, total_val_batches)
            
            # Update learning rate scheduler
            self.scheduler['plateau'].step(avg_val_loss)
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"  ✓ New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"  - No improvement for {patience_counter}/{patience} epochs")
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with validation loss: {best_val_loss:.6f}")
            
        return history

    def predict(self, data):
        """Make predictions on new data."""
        self.model.eval()
        
        print("\n" + "="*80)
        print("PREDICTION DEBUG INFORMATION:")
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
        
        # Check if we need to add lagged features that were used during training
        lagged_features = [col for col in self.config['feature_cols'] 
                          if col.startswith(f'{target_feature}_lag_') 
                          or col.startswith(f'{target_feature}_rolling_')
                          or col.startswith(f'{target_feature}_diff_')]
        
        time_features = [col for col in self.config['feature_cols'] 
                        if col.endswith('_sin') or col.endswith('_cos')]
        
        if target_feature in data[station_id]:
            print(f"Adding features to test data:")
            if lagged_features:
                print(f"  Lagged features: {lagged_features}")
            if time_features:
                print(f"  Time features: {time_features}")
                
            # Create a copy of the data to avoid modifying the original
            test_data = {station_id: {}}
            for feature in data[station_id]:
                test_data[station_id][feature] = data[station_id][feature].copy()
        
            # Create a DataFrame with the target feature
            target_series = test_data[station_id][target_feature]
            
            # Ensure target_series is a Series
            if isinstance(target_series, pd.DataFrame):
                if target_feature in target_series.columns:
                    target_series = target_series[target_feature]
                elif len(target_series.columns) == 1:
                    target_series = target_series.iloc[:, 0]
                else:
                    print(f"Warning: Cannot convert {target_feature} DataFrame to Series. Using first column.")
                    target_series = target_series.iloc[:, 0]
                test_data[station_id][target_feature] = target_series
            
            # Add all required features
            for feature in self.config['feature_cols']:
                if feature not in test_data[station_id]:
                    # Check if it's a time-based feature we can create
                    if feature == 'hour_sin':
                        test_data[station_id][feature] = np.sin(2 * np.pi * target_series.index.hour / 24)
                    elif feature == 'hour_cos':
                        test_data[station_id][feature] = np.cos(2 * np.pi * target_series.index.hour / 24)
                    elif feature == 'minute_sin':
                        test_data[station_id][feature] = np.sin(2 * np.pi * target_series.index.minute / 60)
                    elif feature == 'minute_cos':
                        test_data[station_id][feature] = np.cos(2 * np.pi * target_series.index.minute / 60)
                    elif feature == 'month_sin':
                        test_data[station_id][feature] = np.sin(2 * np.pi * target_series.index.month / 12)
                    elif feature == 'month_cos':
                        test_data[station_id][feature] = np.cos(2 * np.pi * target_series.index.month / 12)
                    # Check if it's a lag feature we can create
                    elif feature == f'{target_feature}_lag_96':
                        test_data[station_id][feature] = target_series.shift(96)
                    elif feature == f'{target_feature}_lag_672':
                        test_data[station_id][feature] = target_series.shift(672)
                    elif feature == f'{target_feature}_rolling_mean_96':
                        test_data[station_id][feature] = target_series.rolling(96).mean()
                    elif feature == f'{target_feature}_diff_96':
                        test_data[station_id][feature] = target_series.diff(96)
                    else:
                        # For any other missing features, use zeros
                        print(f"  Creating missing feature {feature} with zeros")
                        test_data[station_id][feature] = pd.Series(0.0, index=target_series.index)
            
            # Use the modified data for prediction
            data = test_data
        
        # Prepare data for prediction
        X, y = self.prepare_data(data, is_training=False)
        
        if X.shape[0] == 0:
            print("No data to predict on!")
            return np.array([])
        
        # Get target feature for index alignment
        station_id = list(data.keys())[0]
        target_feature = self.config.get('output_features', ['vst_raw'])[0]
        target_data = data[station_id][target_feature]
        
        # Convert DataFrame to Series if needed
        if isinstance(target_data, pd.DataFrame):
            if target_feature in target_data.columns:
                target_data = target_data[target_feature]
            elif len(target_data.columns) == 1:
                target_data = target_data.iloc[:, 0]
        
        # Check if sequence is too long and needs chunking
        max_chunk_size = 5000  # Reduced from 10000 to 5000 to save memory
        need_chunking = X.shape[1] > max_chunk_size
        
        if need_chunking:
            print(f"Prediction sequence length ({X.shape[1]}) exceeds maximum chunk size ({max_chunk_size})")
            print(f"Will process predictions in chunks to save memory")
            
            # Calculate number of chunks
            seq_len = X.shape[1]
            num_chunks = (seq_len + max_chunk_size - 1) // max_chunk_size
            print(f"Predictions will be processed in {num_chunks} chunks")
            
            # Initialize array for predictions
            all_predictions = np.zeros((X.shape[0], seq_len, 1))
            
            # Process each chunk
            with torch.no_grad():
                for i in range(num_chunks):
                    start_idx = i * max_chunk_size
                    end_idx = min((i + 1) * max_chunk_size, seq_len)
                    chunk_len = end_idx - start_idx
                    
                    print(f"Processing prediction chunk {i+1}/{num_chunks} (indices {start_idx}:{end_idx})")
                    
                    # Extract chunk
                    X_chunk = X[:, start_idx:end_idx, :]
                    X_chunk = X_chunk.to(self.device)
                    
                    # Make predictions on chunk with mixed precision if available
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            chunk_predictions = self.model(X_chunk)
                    else:
                        chunk_predictions = self.model(X_chunk)
                    
                    chunk_predictions = chunk_predictions.cpu().numpy()
                    
                    # Store predictions
                    all_predictions[:, start_idx:end_idx, :] = chunk_predictions
                    
                    # Clear GPU cache after each chunk
                    clear_gpu_memory()
            
            # Use combined predictions
            predictions = all_predictions
        else:
            # Make predictions on the entire sequence at once
            X = X.to(self.device)
            with torch.no_grad():
                # Use mixed precision for inference if available
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(X)
                else:
                    predictions = self.model(X)
                predictions = predictions.cpu().numpy()
        
        # Inverse transform predictions
        target_min = self.target_scaler['min']
        target_max = self.target_scaler['max']
        unscaled_predictions = predictions * (target_max - target_min) + target_min
        
        print(f"Generated predictions shape: {unscaled_predictions.shape}")
        if unscaled_predictions.size > 0:
            print(f"Prediction range: [{unscaled_predictions.min():.2f}, {unscaled_predictions.max():.2f}]")
            if isinstance(target_data, pd.Series):
                print(f"Actual data range: [{target_data.min():.2f}, {target_data.max():.2f}]")
        
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
                # 1. Ensure predictions are within reasonable bounds but allow for wider range
                min_allowed = max(0, target_data.min() * 0.5)  # Allow 50% below min but not negative (was 20%)
                max_allowed = target_data.max() * 1.5  # Allow 50% above max (was 20%)
                predictions_series = predictions_series.clip(min_allowed, max_allowed)
                
                # 2. Apply minimal smoothing to preserve dynamics
                predictions_series = predictions_series.rolling(window=2, center=True).mean().fillna(method='ffill').fillna(method='bfill')
                
                # 3. Apply lighter exponential smoothing to preserve more dynamics
                alpha = 0.5  # Increased from 0.3 to preserve more of the original signal
                smoothed_predictions = pd.Series(index=predictions_series.index)
                smoothed_predictions.iloc[0] = predictions_series.iloc[0]
                
                for i in range(1, len(predictions_series)):
                    smoothed_predictions.iloc[i] = alpha * predictions_series.iloc[i] + (1 - alpha) * smoothed_predictions.iloc[i-1]
                
                # 4. Ensure we preserve important peaks and valleys with a lower threshold
                if isinstance(target_data, pd.Series) and len(target_data) > 96:  # Changed from 24 to 96 for 15-minute data
                    # Calculate standard deviation using a 1-day window (96 points for 15-minute data)
                    target_std = target_data.rolling(96).std().mean() * 0.5  # Reduced threshold to preserve more dynamics
                    
                    # Only apply smoothing where the change is less than the standard deviation
                    for i in range(1, len(smoothed_predictions)):
                        if abs(predictions_series.iloc[i] - predictions_series.iloc[i-1]) > target_std:
                            smoothed_predictions.iloc[i] = predictions_series.iloc[i]
                
                return smoothed_predictions
            else:
                print(f"Warning: Cannot align predictions with data_index. Using target data index.")
                # Try to align with target data
                if len(target_data) >= len(predictions_flat):
                    predictions_index = target_data.index[:len(predictions_flat)]
                    predictions_series = pd.Series(predictions_flat, index=predictions_index)
                    print(f"Created predictions series with {len(predictions_series)} points")
                    return predictions_series
                else:
                    print(f"Warning: Target data length ({len(target_data)}) is less than predictions length ({len(predictions_flat)})")
        
        return predictions_flat

def create_full_plot(test_data, test_predictions, station_id):
    """Create an interactive plot comparing actual and predicted values."""
    # Extract the target data
    test_actual = test_data['vst_raw']
    
    # Print detailed debug information
    print("\n" + "="*80)
    print("PLOTTING DEBUG INFORMATION:")
    print(f"Test data keys: {list(test_data.keys())}")
    print(f"Test actual data type: {type(test_actual)}")
    print(f"Test actual length: {len(test_actual) if hasattr(test_actual, '__len__') else 'N/A'}")
    print(f"Test predictions type: {type(test_predictions)}")
    
    # Convert DataFrame to Series if needed
    if isinstance(test_actual, pd.DataFrame):
        print("Converting DataFrame to Series...")
        if 'vst_raw' in test_actual.columns:
            test_actual = test_actual['vst_raw']
        else:
            # If there's only one column, use that
            if len(test_actual.columns) == 1:
                test_actual = test_actual.iloc[:, 0]
            else:
                # Try to find a numeric column
                numeric_cols = test_actual.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    test_actual = test_actual[numeric_cols[0]]
                else:
                    print("Error: Cannot determine which column to use from DataFrame")
                    return
    
    if isinstance(test_actual, pd.Series) and not test_actual.empty:
        print(f"Test actual date range: {test_actual.index.min()} to {test_actual.index.max()}")
        print(f"Test actual sample values: {test_actual.iloc[:5].values}")
        print(f"Test actual range: [{test_actual.min():.2f}, {test_actual.max():.2f}]")
    else:
        print("Warning: Test actual data is empty or not a pandas Series")
        return
    
    # Handle predictions
    if isinstance(test_predictions, pd.Series):
        print(f"Predictions are already a Series with {len(test_predictions)} points")
        print(f"Predictions date range: {test_predictions.index.min()} to {test_predictions.index.max()}")
        print(f"Predictions sample values: {test_predictions.iloc[:5].values}")
        print(f"Predictions range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
        predictions_series = test_predictions
    else:
        # Convert numpy array to Series
        if hasattr(test_predictions, 'shape'):
            print(f"Test predictions shape: {test_predictions.shape}")
        
        if len(test_predictions) > 0:
            print(f"Test predictions sample values: {test_predictions[:5]}")
            print(f"Test predictions range: [{np.min(test_predictions):.2f}, {np.max(test_predictions):.2f}]")
            
            # Create pandas Series for predictions, handling potential length mismatch
            pred_length = min(len(test_actual), len(test_predictions))
            predictions_index = test_actual.index[:pred_length]
            predictions_series = pd.Series(test_predictions[:pred_length], index=predictions_index, name='Predictions')
        else:
            print("Warning: Test predictions array is empty")
            return
    
    print("="*80)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=test_actual.index, y=test_actual, name="Actual", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions_series.index, y=predictions_series, name="Predicted", line=dict(color='red')))
    
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
