"""
LSTM-based Forecasting Model for Anomaly Detection in Time Series Data.

This module implements LSTM-based forecasting models for time series anomaly detection.
Two primary model architectures are provided:

1. LSTMForecaster: A unidirectional LSTM that predicts future values based only on past data.
   - Takes a sequence of input_length timesteps
   - Predicts output_length future timesteps
   - Purely causal (no future information leakage)

2. BidirectionalForecaster: A more advanced architecture combining:
   - A bidirectional LSTM encoder to capture patterns from the input sequence
   - A unidirectional decoder that forecasts future values
   - Leverages pattern recognition while maintaining proper forecasting

Key Components:
- ForecasterWrapper: Manages data preprocessing, training, evaluation, and prediction
- train_forecaster: High-level function to create and train models
- evaluate_forecaster: Evaluates models on test data for anomaly detection

Anomaly Detection Approach:
1. Train model on normal (clean) data to learn typical patterns
2. Apply model to test data (potentially containing anomalies)
3. Compare model forecasts with actual values
4. Flag significant deviations as potential anomalies
5. Calculate performance metrics using known ground truth

Data Handling:
- Uses sliding windows to create input-output pairs
- Normalizes data using Min-Max scaling
- Handles multiple features if available
- Supports multi-station training

The forecasting approach is particularly effective for water level data, as it:
1. Preserves temporal causality (predictions only use past data)
2. Can detect both sudden spikes and gradual drift
3. Adapts to normal seasonal or daily patterns

TODO:
   # Add rate-of-change detection after calculating errors
   # Flag sudden changes that exceed physical limits
   rate_of_change = np.abs(np.diff(aligned_actual, prepend=aligned_actual[0]))
   physical_anomalies = (rate_of_change > 50).astype(int)  # 50mm per interval is suspicious
   anomaly_flags = np.logical_or(anomaly_flags, physical_anomalies).astype(int)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Union, Optional
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import json
from tqdm.auto import tqdm  # Import tqdm for progress bars

class LSTMForecaster(nn.Module):
    """
    Unidirectional LSTM model for forecasting future time series values.
    
    This model takes a sequence of past observations and predicts a sequence of future values.
    It maintains temporal causality by only using past information to predict the future.
    
    Architecture:
    1. Multi-layer LSTM to process the input sequence
    2. For each prediction step:
       a. Pass current input through LSTM
       b. Project LSTM output to predict next value
       c. Use prediction as input for next timestep
    
    This implementation enables multi-step forecasting by feeding predictions
    back as inputs (autoregressive approach).
    
    Attributes:
        input_dim (int): Number of input features (e.g., 1 for just water level)
        hidden_dim (int): Size of LSTM hidden state
        output_length (int): Number of future timesteps to predict
        num_layers (int): Number of stacked LSTM layers
        lstm (nn.LSTM): LSTM layers
        fc (nn.Linear): Linear layer for final prediction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_length: int = 24, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the LSTM forecaster.
        
        Args:
            input_dim: Number of input features (e.g., 1 for just water level)
            hidden_dim: Hidden dimension size (larger = more capacity but more params)
            output_length: Number of future steps to predict
            num_layers: Number of stacked LSTM layers (deeper = more complex patterns)
            dropout: Dropout rate for regularization (helps prevent overfitting)
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length
        self.num_layers = num_layers
        
        # LSTM layers - processes the input sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,         # Number of input features
            hidden_size=hidden_dim,       # Size of hidden state
            num_layers=num_layers,        # Number of stacked LSTMs
            batch_first=True,             # Input shape: [batch, seq, features]
            dropout=dropout if num_layers > 1 else 0  # Only apply dropout between layers
        )
        
        # Fully connected output layer - maps hidden state to predicted value
        self.fc = nn.Linear(hidden_dim, 1)  # Predict a single value (water level)
        
    def forward(self, x):
        """
        Forward pass through the forecaster.
        
        For each prediction step:
        1. Pass current input through LSTM
        2. Use hidden state to predict next value
        3. Use prediction as input for next timestep
        
        Args:
            x: Input sequence of shape [batch, input_length, features]
                This contains the historical data to base predictions on
            
        Returns:
            torch.Tensor: Output sequence of shape [batch, output_length, 1]
                These are the predicted future values, one for each future timestep
        """
        batch_size = x.size(0)
        
        # Process input sequence through LSTM to get initial hidden state
        # lstm_out: [batch, seq_len, hidden_dim]
        # h_state: [num_layers, batch, hidden_dim]
        # c_state: [num_layers, batch, hidden_dim]
        lstm_out, (h_state, c_state) = self.lstm(x)
        
        # Get the last hidden state (not used directly, but kept for clarity)
        h_n = h_state[-1].view(batch_size, self.hidden_dim)
        
        # Initialize outputs list to store predictions
        outputs = []
        
        # Generate 'output_length' predictions autoregessively
        # Start with the last input timestep
        current_input = x[:, -1, :].unsqueeze(1)  # Shape: [batch, 1, features]
        
        for _ in range(self.output_length):
            # Forward through LSTM using previous states
            out, (h_state, c_state) = self.lstm(current_input, (h_state, c_state))
            
            # Predict next value from LSTM output
            # out.squeeze(1): [batch, hidden_dim]
            # next_pred: [batch, 1, 1]
            next_pred = self.fc(out.squeeze(1)).unsqueeze(1)
            outputs.append(next_pred)
            
            # Update input for next prediction (using predicted value)
            if self.input_dim > 1:
                # For multivariate case: keep predicted value for first feature, zero others
                next_input = torch.zeros(batch_size, 1, self.input_dim, device=x.device)
                next_input[:, :, 0] = next_pred.squeeze(2)  # Assuming Value is first feature
            else:
                # For univariate case: just use the prediction
                next_input = next_pred
            
            current_input = next_input
        
        # Concatenate all predictions along sequence dimension
        # output shape: [batch, output_length, 1]
        return torch.cat(outputs, dim=1)

class BidirectionalForecaster(nn.Module):
    """
    Bidirectional LSTM model for enhanced pattern recognition with forecasting capability.
    
    This model uses a hybrid approach:
    1. A bidirectional LSTM encoder captures patterns from both directions in the input
    2. A unidirectional LSTM decoder generates future forecasts
    
    This architecture combines the strengths of bidirectional LSTMs for pattern recognition
    with proper forecasting constraints (no future information leakage during prediction).
    
    The encoder-decoder architecture allows the model to:
    - Understand complex relationships in the historical data (bidirectional)
    - Generate future predictions that respect temporal causality (unidirectional)
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Size of LSTM hidden state (per direction)
        output_length (int): Number of future timesteps to predict
        num_layers (int): Number of stacked LSTM layers
        encoder (nn.LSTM): Bidirectional LSTM for encoding input sequence
        decoder (nn.LSTM): Unidirectional LSTM for generating forecasts
        encoder_fc (nn.Linear): Linear layer for encoder output
        output_fc (nn.Linear): Linear layer for final prediction
    """
    
    def __init__(self, input_dim, hidden_dim, output_length=24, num_layers=2, dropout=0.2):
        """
        Initialize the bidirectional forecaster.
        
        Args:
            input_dim: Number of input features (e.g., 1 for water level only)
            hidden_dim: Hidden dimension size per direction (actual hidden size will be 2*hidden_dim)
            output_length: Number of future timesteps to predict
            num_layers: Number of stacked LSTM layers (deeper = more complex patterns)
            dropout: Dropout rate for regularization (helps prevent overfitting)
        """
        super(BidirectionalForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length
        self.num_layers = num_layers
        
        # Bidirectional encoder to capture patterns from the input sequence
        # This processes the input in both forward and backward directions
        self.encoder = nn.LSTM(
            input_size=input_dim,         # Number of input features
            hidden_size=hidden_dim,       # Size of hidden state (per direction)
            num_layers=num_layers,        # Number of stacked LSTMs
            batch_first=True,             # Input shape: [batch, seq, features]
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout between layers
            bidirectional=True            # Process sequence in both directions
        )
        
        # Forward-only decoder for forecasting future values
        # This ensures that during prediction, we don't use future information
        self.decoder = nn.LSTM(
            input_size=1,                 # Input is the previous predicted value
            hidden_size=hidden_dim * 2,   # Match bidirectional encoder output size
            num_layers=num_layers,        # Same structure as encoder
            batch_first=True,             # Input shape: [batch, seq, features]
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False           # Forward-only for forecasting (causal)
        )
        
        # Linear layers for processing
        self.encoder_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # Process encoder output
        self.output_fc = nn.Linear(hidden_dim * 2, 1)  # Map decoder output to predictions
        
        
    def forward(self, x):
        """
        Forward pass through the bidirectional forecaster.
        
        Process:
        1. Apply bidirectional encoder to input sequence
        2. Extract and combine forward/backward hidden states
        3. Use hidden states to initialize decoder
        4. Generate forecasts autoregressively with the decoder
        
        Args:
            x: Input sequence of shape [batch, input_length, features]
                This contains the historical data to base predictions on
            
        Returns:
            torch.Tensor: Output sequence of shape [batch, output_length, 1]
                These are the predicted future values, one for each future timestep
        """
        batch_size = x.size(0)
        
        # Apply a small amount of noise during training to prevent overfitting
        # This adds random variation that forces the model to be more robust
        if self.training:
            x = x + torch.randn_like(x) * 0.01
        
        # Run input sequence through bidirectional encoder
        # encoder_out: [batch, seq_len, hidden_dim*2]
        # h_n, c_n: [num_layers*2, batch, hidden_dim]
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # Extract and combine hidden states from both directions
        # This step is crucial for proper bidirectional-to-unidirectional transition
        # Reshape hidden states from [num_layers*2, batch, hidden] to [num_layers, batch, hidden*2]
        
        # For hidden state: separate forward and backward directions
        h_n_forward = h_n[0::2]  # Forward direction (even indices)
        h_n_backward = h_n[1::2]  # Backward direction (odd indices)
        # Combine directions along feature dimension
        h_n_combined = torch.cat([h_n_forward, h_n_backward], dim=2)
        
        # Same for cell state
        c_n_forward = c_n[0::2]
        c_n_backward = c_n[1::2]
        c_n_combined = torch.cat([c_n_forward, c_n_backward], dim=2)
        
        # Initialize decoder input with last value from input sequence
        # This starts the autoregressive generation process
        decoder_input = x[:, -1:, 0:1]  # Shape: [batch, 1, 1]
        
        # Initialize decoder states and prepare for forecast generation
        outputs = []
        h_state, c_state = h_n_combined, c_n_combined
        
        # Generate 'output_length' predictions autoregressively
        for _ in range(self.output_length):
            # Forward through decoder using previous state
            out, (h_state, c_state) = self.decoder(decoder_input, (h_state, c_state))
            
            # Predict next value
            next_pred = self.output_fc(out).view(batch_size, 1, 1)
            outputs.append(next_pred)
            
            # Use prediction as input for next timestep
            decoder_input = next_pred
        
        # Concatenate all predictions along sequence dimension
        # Output shape: [batch, output_length, 1]
        outputs = torch.cat(outputs, dim=1)
        return outputs
class ForecasterWrapper:
    """
    Wrapper class for LSTM forecasting models with comprehensive functionality.
    
    This class provides a high-level interface for:
    1. Data preprocessing and normalization
    2. Model creation (unidirectional or bidirectional)
    3. Training with early stopping
    4. Validation and inference
    5. Handling of time series data with proper sequence creation
    
    The wrapper manages:
    - Feature scaling/normalization using MinMaxScaler
    - Creation of sliding window sequences for training
    - Progress tracking and reporting
    - Model configuration and hyperparameters
    - Device management (CPU/GPU)
    
    Attributes:
        config (dict): Full configuration dictionary
        input_length (int): Length of input sequences
        feature_cols (list): Names of feature columns
        all_features (list): All features including weather data
        n_features (int): Total number of features
        hidden_dim (int): Size of LSTM hidden layer
        num_layers (int): Number of LSTM layers
        use_bidirectional (bool): Whether to use bidirectional model
        model (nn.Module): The PyTorch LSTM model (either uni or bidirectional)
        device (torch.device): Device to use for computation
        optimizer (torch.optim): Optimizer for training
        criterion (nn.Module): Loss function
        scaler (MinMaxScaler): Scaler for feature normalization
        is_fitted (bool): Whether the scaler has been fitted
    """
    
    def __init__(self, config):
        """
        Initialize the forecaster wrapper with configuration.
        
        Args:
            config: Dictionary containing all configuration parameters
                input_length: Length of input sequences
                feature_cols: List of feature column names
                weather_cols: List of weather feature column names
                hidden_dim: Size of LSTM hidden state
                num_layers: Number of LSTM layers
                dropout_rate: Dropout rate for regularization
                learning_rate: Learning rate for optimizer
                batch_size: Batch size for training
                use_bidirectional: Whether to use bidirectional model
        """
        # Store configuration
        self.config = config.copy()
        
        # Extract key parameters
        self.input_length = config.get('input_length', 72)  # Default 72 timesteps
        self.feature_cols = config.get('feature_cols', ['Value'])  # Default to just Value
        self.weather_cols = config.get('weather_cols', [])  # Optional weather features
        self.all_features = self.feature_cols + self.weather_cols  # Combined feature list
        self.n_features = len(self.all_features)  # Total number of features
        
        # Model hyperparameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 128)
        self.use_bidirectional = config.get('use_bidirectional', False)  # Default to bidirectional
        
        # Create the PyTorch model based on configuration
        if self.use_bidirectional:
            # Bidirectional model with encoder-decoder architecture
            self.model = BidirectionalForecaster(
                input_dim=self.n_features,  # Number of input features
                hidden_dim=self.hidden_dim,  # Size of hidden state per direction
                output_length=config.get('output_length', 24),  # Future prediction steps
                num_layers=self.num_layers,
                dropout=self.dropout_rate
            )
        else:
            # Original unidirectional model
            self.model = LSTMForecaster(
                input_dim=self.n_features,
                hidden_dim=self.hidden_dim,
                output_length=config.get('output_length', 24),
                num_layers=self.num_layers,
                dropout=self.dropout_rate
            )
        
        # Set computation device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)  # Move model to device
        
        # Initialize optimizer with model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function - Mean Squared Error for regression
        self.criterion = nn.MSELoss()
        
        # Data preprocessing - MinMaxScaler normalizes features to 0-1 range
        self.scaler = MinMaxScaler()
        self.is_fitted = False  # Tracks whether scaler has been fitted to data
    
    def prepare_data(self, data_dict, feature_cols):
        """
        Prepare data for training or prediction by creating sequences and normalizing.
        
        This method:
        1. Extracts data from each station in the dictionary
        2. Fits the scaler if not already fitted
        3. Normalizes the data using the scaler
        4. Creates sliding window sequences with the appropriate lookback
        5. Combines data from all stations for batch processing
        
        Args:
            data_dict: Dictionary mapping station keys to their data
                Each entry contains a 'vst_raw' DataFrame with time series data
            feature_cols: List of feature column names to use
            
        Returns:
            Tuple of (X, y, timestamps):
                X: Tensor of input sequences [batch, input_length, features]
                y: Tensor of target sequences [batch, output_length, 1]
                timestamps: Array of timestamps for each sequence
        """
        X_all = []  # Input sequences from all stations
        y_all = []  # Target sequences from all stations
        timestamps_all = []  # Timestamps for all sequences
        
        # Process each station's data
        for station_key, station_data in data_dict.items():
            # Get DataFrame with time series data
            df = station_data.get('vst_raw', None)
            if df is None or df.empty:
                continue  # Skip stations with no data
            
            # Check if sufficient data for a sequence
            if len(df) < self.input_length:
                print(f"Skipping {station_key}: insufficient data (need {self.input_length}, got {len(df)})")
                continue
            
            # Extract features (ensure all required columns exist)
            features = []
            for col in feature_cols:
                if col in df.columns:
                    features.append(df[col].values)
                else:
                    # Use zeros if feature is missing
                    features.append(np.zeros(len(df)))
            
            # Stack features to create feature array [time, features]
            feature_array = np.column_stack(features)
            
            # Fit scaler once on entire dataset (first time only)
            # This ensures consistent scaling across all data
            if not self.is_fitted:
                self.scaler.fit(feature_array)
                self.is_fitted = True
            
            # Scale features to 0-1 range
            scaled_features = self.scaler.transform(feature_array)
            
            # Create sequences for this station
            X, y, timestamps = self._create_sequences(
                scaled_features,  # Normalized features
                df.index.values,  # Timestamps
                input_length=self.input_length  # Lookback window
            )
            
            # Add to combined lists
            X_all.append(X)
            y_all.append(y)
            timestamps_all.append(timestamps)
        
        # Combine data from all stations
        if X_all:
            # Stack along batch dimension
            X_combined = np.vstack(X_all)  # [total_sequences, input_length, features]
            y_combined = np.vstack(y_all)  # [total_sequences, target_length, 1]
            timestamps_combined = np.vstack(timestamps_all)
            
            # Convert to torch tensors and move to device
            X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y_combined, dtype=torch.float32).to(self.device)
            
            return X_tensor, y_tensor, timestamps_combined
        else:
            # Return empty tensors if no valid data
            return torch.tensor([]), torch.tensor([]), np.array([])
    
    def fit(self, train_data, validation_data=None, epochs=100, patience=10, verbose=True):
        """
        Train the forecaster model with comprehensive progress tracking.
        
        Features:
        - Early stopping based on validation loss
        - Progress bars for epochs and batches
        - Detailed logging of training metrics
        - Best model checkpoint saving
        
        Args:
            train_data: Dictionary of training data by station
            validation_data: Optional dictionary of validation data by station
            epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before early stopping
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing training history:
                train_loss: List of training losses per epoch
                val_loss: List of validation losses per epoch
                best_epoch: Best epoch based on validation loss
                best_val_loss: Best validation loss achieved
        """
        # Set model to training mode
        self.model.train()
        
        # Print configuration details if verbose
        if verbose:
            print("\n=== LSTM Model Configuration ===")
            print(f"Model type: {'Bidirectional' if self.use_bidirectional else 'Unidirectional'} LSTM")
            print(f"Input length: {self.input_length}")
            print(f"Features: {self.all_features}")
            print(f"Hidden dimensions: {self.hidden_dim}")
            print(f"Number of layers: {self.num_layers}")
            print(f"Batch size: {self.batch_size}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Device: {self.device}")
            print("============================\n")
        
        # Prepare training data
        if verbose:
            print("Preparing training data...")
        X_train, y_train, _ = self.prepare_data(train_data, self.all_features)
        
        # Prepare validation data if provided
        if validation_data:
            if verbose:
                print("Preparing validation data...")
            X_val, y_val, _ = self.prepare_data(validation_data, self.all_features)
        else:
            X_val, y_val = None, None
        
        # Print data sizes
        if verbose:
            print(f"\nTraining data: {X_train.shape[0]} sequences")
            if X_val is not None:
                print(f"Validation data: {X_val.shape[0]} sequences")
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Main training loop with tqdm progress bar
        epoch_pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        for epoch in epoch_pbar:
            # Train model for one epoch
            epoch_loss = self._train_epoch(X_train, y_train, verbose=verbose)
            history['train_loss'].append(epoch_loss)
            
            # Update progress bar with loss information
            epoch_info = f"Loss: {epoch_loss:.6f}"
            
            # Validate model if validation data provided
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, verbose=verbose)
                history['val_loss'].append(val_loss)
                epoch_info += f", Val Loss: {val_loss:.6f}"
                
                # Check for early stopping - if validation improves, reset patience
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    history['best_val_loss'] = best_val_loss
                    history['best_epoch'] = epoch + 1
                    epoch_info += " ✓"  # Mark best epoch
                else:
                    # If no improvement, increment patience counter
                    patience_counter += 1
                    if patience_counter >= patience:
                        epoch_pbar.set_postfix_str(f"{epoch_info} - Early stopping!")
                        epoch_pbar.update(epochs - epoch - 1)  # Update to 100%
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            # Update progress bar
            epoch_pbar.set_postfix_str(epoch_info)
        
        # Load best model if using validation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"\nRestored best model from epoch {history['best_epoch']} with validation loss: {best_val_loss:.6f}")
        
        return history

    def _train_epoch(self, X, y, verbose=True):
        """
        Train the model for one epoch with detailed progress tracking.
        
        This method:
        1. Creates a DataLoader for batch processing
        2. Shows progress with tqdm
        3. Computes loss and performs backpropagation
        4. Handles shape mismatches between outputs and targets
        
        Args:
            X: Input tensor of shape [batch, input_length, features]
            y: Target tensor of shape [batch, output_length, 1]
            verbose: Whether to show progress bar and output messages
            
        Returns:
            Average loss for the epoch
        """
        total_loss = 0
        batch_count = 0
        
        # Create DataLoader for batch processing
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), 
            batch_size=self.batch_size, 
            shuffle=True  # Shuffle data for better training
        )
        
        # Training loop with progress bar for batches
        batch_pbar = tqdm(
            train_loader, 
            desc="Batches", 
            leave=False,  # Don't leave the progress bar when done
            disable=not verbose  # Only show if verbose
        )
        
        # Set model to training mode (enables dropout, etc.)
        self.model.train()
        
        # Process each batch
        for batch_X, batch_y in batch_pbar:
            # Ensure data is on the correct device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass - get model predictions
            outputs = self.model(batch_X)
            
            # Calculate loss based on the shapes
            # Handle case where outputs and targets have different lengths
            if outputs.shape[1] != batch_y.shape[1]:
                if verbose and batch_count == 0:  # Print only on first batch
                    print(f"Warning: Output shape {outputs.shape} doesn't match target shape {batch_y.shape}")
                    print(f"Using first {min(outputs.shape[1], batch_y.shape[1])} steps for loss calculation")
                
                # Use the minimum length for loss calculation
                min_len = min(outputs.shape[1], batch_y.shape[1])
                loss = self.criterion(outputs[:, :min_len, :], batch_y[:, :min_len, :])
            else:
                # Standard case: shapes match
                loss = self.criterion(outputs, batch_y)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update model weights
            
            # Track metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar with current loss
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count
        return avg_loss

    def _validate(self, X, y, verbose=True):
        """
        Validate the model on validation data.
        
        This method:
        1. Sets model to evaluation mode (disables dropout)
        2. Makes predictions on validation data
        3. Calculates validation loss
        4. Tracks progress with tqdm
        
        Args:
            X: Input validation tensor of shape [batch, input_length, features]
            y: Target validation tensor of shape [batch, output_length, 1]
            verbose: Whether to show progress bar
            
        Returns:
            Average validation loss
        """
        total_loss = 0
        batch_count = 0
        
        # Create DataLoader for batch processing
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y), 
            batch_size=self.batch_size, 
            shuffle=False  # No need to shuffle validation data
        )
        
        # Validation loop with progress bar
        val_pbar = tqdm(
            val_loader, 
            desc="Validation", 
            leave=False,  # Don't leave the progress bar
            disable=not verbose  # Only show if verbose
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation for validation
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                # Ensure data is on the correct device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass - get predictions
                outputs = self.model(batch_X)
                
                # Calculate loss - handle shape differences
                if outputs.shape[1] != batch_y.shape[1]:
                    # Use minimum length for comparison
                    min_len = min(outputs.shape[1], batch_y.shape[1])
                    loss = self.criterion(outputs[:, :min_len, :], batch_y[:, :min_len, :])
                else:
                    # Standard case: shapes match
                    loss = self.criterion(outputs, batch_y)
                
                # Track metrics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                val_pbar.set_postfix(val_loss=f"{loss.item():.6f}")
        
        # Calculate average validation loss
        avg_loss = total_loss / batch_count
        return avg_loss

    def predict(self, X):
        """
        Make predictions with the trained model.
        
        This method:
        1. Sets model to evaluation mode
        2. Handles conversion between numpy arrays and torch tensors
        3. Performs inference without gradient computation
        4. Formats outputs for further processing
        
        Args:
            X: Input data as numpy array or torch tensor
                Shape should be [batch, input_length, features]
            
        Returns:
            numpy.ndarray: Predictions
                For bidirectional model: [batch, output_length]
                For unidirectional model: [batch, output_length]
        """
        # Move to device and convert to tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(self.device)
        
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            if self.use_bidirectional:
                # Bidirectional model - direct forecast
                outputs = self.model(X)
                # Reshape if needed
                if outputs.dim() == 3 and outputs.shape[2] == 1:
                    outputs = outputs.squeeze(2)
            else:
                # Original unidirectional model - autoregressive generation
                outputs = []
                # Generate 'output_length' predictions
                current_input = X
                h_state, c_state = None, None
                
                for _ in range(self.input_length):
                    # Forward through LSTM
                    out, (h_state, c_state) = self.model.lstm(
                        current_input, 
                        (h_state, c_state) if h_state is not None else None
                    )
                    
                    # Predict next value
                    next_pred = self.model.fc(out[:, -1, :]).unsqueeze(1)  # [batch, 1, 1]
                    outputs.append(next_pred)
                    
                    # Update input for next prediction
                    if self.n_features > 1:
                        # For multivariate: zero other features
                        next_input = torch.zeros(X.shape[0], 1, self.n_features, device=self.device)
                        next_input[:, :, 0] = next_pred.squeeze(2)  # Value in first dimension
                    else:
                        # For univariate: just use prediction
                        next_input = next_pred
                    
                    # Shift window or extend sequence
                    if current_input.shape[1] >= self.input_length:
                        # Remove oldest, add newest (sliding window)
                        current_input = torch.cat((current_input[:, 1:, :], next_input), dim=1)
                    else:
                        # Add to sequence
                        current_input = torch.cat((current_input, next_input), dim=1)
                
                # Concatenate all predictions
                outputs = torch.cat(outputs, dim=1)
        
        # Convert to numpy for further processing
        return outputs.cpu().numpy()

    def _create_sequences(self, data, timestamps, input_length):
        """
        Create sequences for training or prediction using sliding windows.
        
        This method creates input-output pairs for sequence modeling:
        - Each input is a sequence of length input_length
        - Each output is a sequence of future values to predict
        
        Args:
            data: Feature array of shape [time, features]
            timestamps: Array of timestamps aligned with data
            input_length: Length of input sequences
            
        Returns:
            Tuple of (X, y, timestamps):
                X: Array of input sequences [num_sequences, input_length, features]
                y: Array of target sequences [num_sequences, output_length, 1]
                timestamps: Array of timestamp sequences [num_sequences, input_length]
        """
        X, y, ts = [], [], []
        output_length = self.config.get('output_length', 24)  # How many steps to predict
        
        # Create sliding windows with proper forecast horizon
        for i in range(len(data) - input_length - output_length + 1):
            # Input sequence: window of historical data
            input_seq = data[i:i+input_length]
            
            # Target sequence: future values to predict
            # Use only the first feature (Value) as target
            target_seq = data[i+input_length:i+input_length+output_length, 0:1]
            
            # Timestamps corresponding to this input sequence
            seq_timestamps = timestamps[i:i+input_length]
            
            # Add to collections
            X.append(input_seq)
            y.append(target_seq)
            ts.append(seq_timestamps)
        
        # Convert to numpy arrays
        return np.array(X), np.array(y), np.array(ts)

def train_forecaster(train_data, validation_data=None, config=None, base_model=None):
    """
    Train a forecaster model with comprehensive configuration and validation.
    
    This high-level function handles:
    1. Model creation and initialization
    2. Transfer learning from base model (if provided)
    3. Training with early stopping based on validation data
    4. Configuration management and hyperparameter handling
    
    The function supports training on multiple stations simultaneously by aggregating
    their data and creating a common model that can generalize across locations.
    
    Args:
        train_data (dict): Dictionary of training data by station
            Each entry should contain a 'vst_raw' DataFrame with time series data
            Format: {station_key: {'vst_raw': pd.DataFrame}}
        
        validation_data (dict, optional): Dictionary of validation data by station
            Used for early stopping and model selection
            Should have the same format as train_data
        
        config (dict, optional): Model configuration dictionary with parameters:
            input_length: Length of input sequences (default: 72)
            hidden_dim: Hidden dimension size (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout_rate: Dropout rate for regularization (default: 0.2)
            learning_rate: Learning rate for optimizer (default: 0.001)
            batch_size: Batch size for training (default: 128)
            epochs: Maximum training epochs (default: 100)
            patience: Early stopping patience (default: 10)
            use_bidirectional: Whether to use bidirectional model (default: True)
            
        base_model (ForecasterWrapper, optional): Pre-trained model for weight initialization
            Enables transfer learning or fine-tuning
        
    Returns:
        tuple: (model, history)
            model (ForecasterWrapper): Trained forecaster model
            history (dict): Training history with metrics:
                train_loss: List of training losses per epoch
                val_loss: List of validation losses per epoch
                best_epoch: Best epoch based on validation loss
                best_val_loss: Best validation loss achieved
    
    Example:
        ```python
        # Train a model on multiple years of station data
        model, history = train_forecaster(
            train_data={
                'station1_2019': {'vst_raw': df_2019},
                'station1_2020': {'vst_raw': df_2020}
            },
            validation_data={
                'station1_2021': {'vst_raw': df_2021}
            },
            config=LSTM_CONFIG
        )
        ```
    """
    if config is None:
        config = {}
    
    # Create model
    model = ForecasterWrapper(config)
    
    # If base_model is provided, use it for initialization
    if base_model is not None and isinstance(base_model, ForecasterWrapper):
        # Copy weights from base model
        model.model.load_state_dict(base_model.model.state_dict())
        model.scaler = base_model.scaler
        model.is_fitted = base_model.is_fitted
        print("Initialized from base model")
    
    # Train the model
    history = model.fit(
        train_data=train_data,
        validation_data=validation_data,
        epochs=config.get('epochs', 100),
        patience=config.get('patience', 10),
        verbose=config.get('verbose', True)
    )
    
    return model, history

def evaluate_forecaster(model, test_data, ground_truth=None, config=None, split_datasets=None):
    """
    Evaluate a forecaster model using true forecasting for anomaly detection.
    
    This comprehensive evaluation function:
    1. Applies a trained model to test data containing potential anomalies
    2. Uses sliding windows to forecast future values from input windows
    3. Compares forecasts against actual values to detect anomalies
    4. Uses adaptive thresholding with local context for improved detection
    5. Calculates performance metrics against ground truth (if available)
    6. Creates visualizations and detailed results for analysis
    
    The forecasting approach maintains temporal causality by only using past data
    to predict future values, making it suitable for real-world applications.
    
    Args:
        model (ForecasterWrapper): Trained forecaster model
            Should be already trained on clean data
        
        test_data (dict): Dictionary of test data by station-year
            Format: {station_key: {'vst_raw_modified': pd.DataFrame}}
            The 'vst_raw_modified' DataFrame contains data with potential anomalies
        
        ground_truth (dict, optional): Dictionary of ground truth anomaly labels
            Format: {station_key: {'periods': [{'start': timestamp, 'end': timestamp}]}}
            Each entry contains periods marking known anomalies for evaluation
        
        config (dict, optional): Configuration parameters:
            input_length: Length of input windows (default: 72)
            output_length: Number of future points to predict (default: 24)
            stride: How many steps to move window (default: 4)
            anomaly_threshold_percentile: Percentile for anomaly threshold (default: 95)
        
        split_datasets (dict, optional): Additional dataset information
            Used for extended evaluation or cross-validation
    
    Returns:
        dict: Results dictionary with detailed evaluation metrics:
            Format: {station_key: {
                'results_df': pd.DataFrame with all metrics and predictions,
                'timestamps': Array of timestamps,
                'original_values': Array of actual values,
                'predictions': Array of forecasted values,
                'prediction_errors': Array of absolute errors,
                'anomaly_flags': Binary array marking detected anomalies,
                'threshold': Anomaly detection threshold,
                'z_scores': Z-scores for each point,
                'imputed_data': Data with anomalies replaced by predictions,
                'metrics': Performance metrics (precision, recall, F1, accuracy),
                'original_data': Original clean data (if available),
                'modified_data': Modified data with potential anomalies
            }}
    
    Notes:
        - The function uses an ensemble approach by averaging predictions for the same point
          from multiple overlapping windows, improving robustness
        - Local context is considered when flagging anomalies, making detection more adaptive
          to normal variations in the data
        - Smoothing is applied to predictions to reduce noise and zigzag patterns
    """
    if config is None:
        config = {}
    
    if ground_truth is None:
        ground_truth = {}
    
    results = {}
    
    # Configure forecasting parameters
    input_length = config.get('input_length', 72)
    forecast_horizon = config.get('output_length', 24)
    stride = config.get('stride', 4)  # How many steps to move window
    
    # Process each station/year
    for station_key, station_data in test_data.items():
        try:
            print(f"Evaluating {station_key} with forecasting approach...")
            
            # Get data with potential anomalies
            df = station_data.get('vst_raw_modified', None)
            if df is None or df.empty:
                continue
            
            # Check if we have enough data
            if len(df) < input_length + forecast_horizon:
                print(f"Warning: Data length ({len(df)}) is insufficient for forecasting")
                results[station_key] = create_empty_results(df)
                continue
            
            # Prepare containers for results
            all_timestamps = []
            all_actual_values = []
            all_predicted_values = []
            successful_forecasts = 0
            
            # Calculate total forecasting windows
            total_windows = (len(df) - input_length - forecast_horizon + 1) // stride
            
            # Process windows with tqdm progress bar
            window_pbar = tqdm(
                range(0, len(df) - input_length - forecast_horizon + 1, stride),
                desc="Forecasting",
                total=total_windows,
                unit="windows"
            )
            
            for i in window_pbar:
                try:
                    # Extract input window
                    input_window = df.iloc[i:i+input_length]
                    
                    # Extract target window (what we want to predict)
                    target_window = df.iloc[i+input_length:i+input_length+forecast_horizon]
                    
                    # Update progress bar description
                    window_pbar.set_description(
                        f"Window {i//stride + 1}/{total_windows} [Predict: {target_window.index[0].date()} to {target_window.index[-1].date()}]"
                    )
                    
                    # Prepare input data
                    if 'Value' in input_window.columns:
                        # Get input values
                        values = input_window['Value'].values.reshape(-1, 1)
                        
                        # Scale input using model's scaler
                        if model.is_fitted:
                            dummy = np.zeros((len(values), model.n_features))
                            dummy[:, 0] = values.flatten()
                            scaled_values = model.scaler.transform(dummy)[:, 0:1]
                        else:
                            min_val, max_val = np.min(values), np.max(values)
                            scaled_values = (values - min_val) / (max_val - min_val + 1e-10)
                        
                        # Create tensor input
                        X = torch.tensor(scaled_values.reshape(1, -1, 1), dtype=torch.float32)
                        
                        # Add zeros for additional features if needed
                        if model.n_features > 1:
                            zeros = torch.zeros(1, X.shape[1], model.n_features-1, dtype=torch.float32)
                            X = torch.cat([X, zeros], dim=2)
                        
                        # Get forecasts from model
                        forecasts_normalized = model.predict(X)
                        
                        # Inverse transform the forecasts to the original scale
                        if model.is_fitted:
                            # Debug the shapes
                            #window_pbar.write(f"Debug - shapes - forecasts_normalized: {forecasts_normalized.shape}, forecast_horizon: {forecast_horizon}")
                            
                            # Make sure we only use the forecast_horizon number of predictions
                            if len(forecasts_normalized.shape) == 1:
                                # 1D array case
                                if len(forecasts_normalized) > forecast_horizon:
                                    forecasts_normalized = forecasts_normalized[:forecast_horizon]
                            elif len(forecasts_normalized.shape) == 2:
                                # 2D array case (batch, sequence)
                                if forecasts_normalized.shape[1] > forecast_horizon:
                                    forecasts_normalized = forecasts_normalized[:, :forecast_horizon]
                                forecasts_normalized = forecasts_normalized.reshape(-1)  # Flatten to 1D
                            
                            # Create dummy array with the right shape for inverse transformation
                            dummy = np.zeros((len(forecasts_normalized), model.n_features))
                            dummy[:, 0] = forecasts_normalized  # Put predictions in first column
                            
                            # Inverse transform
                            forecasts_unscaled = model.scaler.inverse_transform(dummy)[:, 0]
                            forecasts = forecasts_unscaled  # Use unscaled forecasts
                        else:
                            # If model not fitted with scaler, just use as-is
                            forecasts = forecasts_normalized
                        
                        # Ensure forecasts have the right shape
                        if forecasts.shape[0] == 1:  # First batch
                            forecasts = forecasts[0]  # Remove batch dimension
                        
                        # Truncate or pad to match target window length
                        if len(forecasts) != len(target_window):
                            if len(forecasts) > len(target_window):
                                forecasts = forecasts[:len(target_window)]
                            else:
                                padding = np.full(len(target_window) - len(forecasts), forecasts[-1])
                                forecasts = np.concatenate([forecasts, padding])
                        
                        # Store results
                        all_timestamps.extend(target_window.index)
                        all_actual_values.extend(target_window['Value'].values)
                        all_predicted_values.extend(forecasts)
                        successful_forecasts += 1
                    else:
                        window_pbar.write(f"Window {i//stride + 1}: No 'Value' column found, skipping")
                        
                except Exception as e:
                    window_pbar.write(f"Error forecasting window {i//stride + 1}: {str(e)}")
            
            print(f"Successfully completed {successful_forecasts} out of {total_windows} forecasts ({successful_forecasts/total_windows*100:.1f}%)")
            
            # Check if we have any forecasts
            if len(all_timestamps) == 0 or not successful_forecasts:
                print("No valid forecasts were generated")
                results[station_key] = create_empty_results(df)
                continue
            
            # Handle overlapping windows by averaging forecasts for the same timestamp
            unique_timestamps = sorted(set(all_timestamps))
            final_forecasts = {}
            
            for ts in unique_timestamps:
                # Find all forecasts for this timestamp
                indices = [i for i, t in enumerate(all_timestamps) if t == ts]
                pred_values = [all_predicted_values[i] for i in indices]
                actual_values = [all_actual_values[i] for i in indices]
                
                # Average the values
                final_forecasts[ts] = {
                    'actual': np.mean(actual_values),
                    'predicted': np.mean(pred_values),
                    'forecast_count': len(pred_values)
                }
            
            # Create aligned arrays
            aligned_timestamps = np.array(unique_timestamps)
            aligned_actual = np.array([final_forecasts[ts]['actual'] for ts in aligned_timestamps])
            aligned_predicted = np.array([final_forecasts[ts]['predicted'] for ts in aligned_timestamps])
            
            # Add smoothing to predictions
            window_size = 3  # Small window to preserve detail while removing zigzags
            aligned_predicted_smooth = pd.Series(aligned_predicted).rolling(
                window=window_size, min_periods=1, center=True
            ).mean().values

            # Use the smoothed predictions for error calculation
            aligned_predicted = aligned_predicted_smooth
            
            # Calculate forecast errors
            forecast_errors = np.abs(aligned_predicted - aligned_actual)
            
            # Calculate local statistics with a rolling window
            error_series = pd.Series(forecast_errors)
            window_size_local = 128  # 12 hours with 15-min data
            rolling_mean = error_series.rolling(window=window_size_local, min_periods=1, center=True).mean()
            rolling_std = error_series.rolling(window=window_size_local, min_periods=1, center=True).std()

            # Calculate z-scores relative to local context
            z_scores_local = (error_series - rolling_mean) / (rolling_std + 1e-10)  # Avoid division by zero

            # Flag anomalies using z-score threshold instead of global percentile
            z_threshold = 2.0  # Lower from 2.5 to catch more anomalies
            anomaly_flags = (z_scores_local > z_threshold).astype(int)

            # Store z-scores for visualization
            z_scores = z_scores_local.values

            # For compatibility, still calculate a threshold value (for plotting)
            percentile_threshold = config.get('anomaly_threshold_percentile', 95)
            threshold = np.percentile(forecast_errors, percentile_threshold)
            
            # Create imputed data (replace anomalies with predictions)
            imputed_data = aligned_actual.copy()
            imputed_data[anomaly_flags == 1] = aligned_predicted[anomaly_flags == 1]
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'timestamp': aligned_timestamps,
                'original': aligned_actual,
                'predicted': aligned_predicted,
                'error': forecast_errors,
                'z_score': z_scores, 
                'is_anomaly': anomaly_flags,
                'imputed': imputed_data
            }).set_index('timestamp')
            
            # Store results
            station_results = {
                'results_df': results_df,
                'timestamps': aligned_timestamps,
                'original_values': aligned_actual,
                'predictions': aligned_predicted,  # Change key name for clarity
                'reconstructed_values': aligned_predicted,  # Keep for backward compatibility
                'prediction_errors': forecast_errors,
                'reconstruction_errors': forecast_errors,  # Keep for backward compatibility
                'anomaly_flags': anomaly_flags,
                'threshold': threshold,
                'z_scores': z_scores,
                'imputed_data': imputed_data,
                'original_data': station_data.get('vst_raw', None),
                'modified_data': df
            }
            
            # Calculate performance metrics if ground truth is available
            if station_key in ground_truth and ground_truth[station_key] is not None:
                try:
                    # Create binary ground truth array aligned with timestamps
                    gt_array = np.zeros(len(aligned_timestamps))
                    
                    for period in ground_truth[station_key].get('periods', []):
                        # Mark points within anomaly periods as 1
                        mask = (pd.DatetimeIndex(aligned_timestamps) >= period['start']) & (pd.DatetimeIndex(aligned_timestamps) <= period['end'])
                        gt_array[mask] = 1

                    # Calculate metrics
                    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                    
                    precision = precision_score(gt_array, anomaly_flags, zero_division=0)
                    recall = recall_score(gt_array, anomaly_flags, zero_division=0)
                    f1 = f1_score(gt_array, anomaly_flags, zero_division=0)
                    accuracy = accuracy_score(gt_array, anomaly_flags)
                    
                    # Add to results
                    station_results['metrics'] = {
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
                    import traceback
                    traceback.print_exc()
            
            # Print summary statistics
            print(f"Processed {len(results_df)} points")
            print(f"Error range: [{np.min(forecast_errors):.6f}, {np.max(forecast_errors):.6f}]")
            print(f"Z-score range: [{np.min(z_scores):.2f}, {np.max(z_scores):.2f}]")
            print(f"Anomaly threshold: {threshold:.6f}")
            print(f"Found {np.sum(anomaly_flags)} potential anomalies ({np.sum(anomaly_flags)/len(anomaly_flags)*100:.2f}%)")
            
            # Store results for this station
            results[station_key] = station_results
            
        except Exception as e:
            print(f"Error evaluating {station_key}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def create_empty_results(data):
    """
    Create empty results structure when processing fails.
    
    This utility function generates a consistent empty results dictionary
    when anomaly detection processing cannot be completed. It ensures that
    downstream functions expecting a complete results structure will not fail,
    maintaining graceful degradation.
    
    The function attempts to extract timestamps and values from the input data
    if available, or creates empty arrays if not.
    
    Args:
        data (pd.DataFrame or None): Input data frame, potentially containing
            timestamps and 'Value' column. Can be None or empty.
    
    Returns:
        dict: Empty results structure with these components:
            results_df: Empty DataFrame
            timestamps: Array of timestamps (if available) or empty array
            original_values: Array of original values or zeros
            reconstructed_values: Copy of original_values
            reconstruction_errors: Array of zeros matching original_values
            anomaly_flags: Array of zeros matching original_values
            threshold: 0.0 (default threshold)
            z_scores: Array of zeros matching original_values
            imputed_data: Copy of original_values
    
    Notes:
        This function is used by evaluate_forecaster when it cannot process
        data due to:
        - Insufficient data length
        - Missing columns
        - Processing errors
        - No valid sequences or windows
    """
    print("Returning empty results structure")
    
    # Create minimal empty results
    if isinstance(data, pd.DataFrame) and not data.empty:
        sample_timestamps = data.index.values
        sample_values = data['Value'].values if 'Value' in data.columns else np.zeros(len(data))
    else:
        # Create truly empty results if data is not available
        sample_timestamps = np.array([])
        sample_values = np.array([])
    
    return {
        'results_df': pd.DataFrame(),
        'timestamps': sample_timestamps,
        'original_values': sample_values,
        'reconstructed_values': sample_values.copy(),
        'reconstruction_errors': np.zeros_like(sample_values),
        'anomaly_flags': np.zeros_like(sample_values, dtype=int),
        'threshold': 0.0,
        'z_scores': np.zeros_like(sample_values),
        'imputed_data': sample_values.copy()
    }

'''
1. Architecture: Encoder-Decoder with Bidirectional Pattern Recognition
The BidirectionalForecaster model uses a sophisticated encoder-decoder architecture:
Encoder Component
Bidirectional LSTM: Processes the input sequence in both forward and backward directions
Pattern Recognition: Captures complex patterns from both past and future context within the input window
Hidden State Fusion: Combines forward and backward hidden states to create a rich representation
Decoder Component
Unidirectional LSTM: Takes the encoder's hidden states and generates future predictions
Autoregressive Generation: Each predicted value becomes input for the next prediction
Temporal Causality: Only uses information available up to the prediction point

# During forward pass:
# 1. Process input with bidirectional encoder
encoder_out, (h_n, c_n) = self.encoder(x)

# 2. Combine forward/backward states 
h_n_forward = h_n[0::2]  # Forward direction
h_n_backward = h_n[1::2]  # Backward direction
h_n_combined = torch.cat([h_n_forward, h_n_backward], dim=2)

# 3. Generate predictions autoregressively with decoder
for _ in range(self.output_length):
    out, (h_state, c_state) = self.decoder(decoder_input, (h_state, c_state))
    next_pred = self.output_fc(out).view(batch_size, 1, 1)
    outputs.append(next_pred)
    decoder_input = next_pred  # Use prediction as next input

2. Sliding Window Forecasting with Ensemble Predictions
The system uses an advanced multi-window approach to generate robust predictions:
Sliding Window Creation
Input Windows: Each window contains input_length (72) historical points
Forecast Horizons: For each window, the model predicts output_length (8) future points
Stride Parameter: Windows slide forward by stride (12) points each time
Multiple Perspectives: This creates overlapping windows that view the same future point from different past contexts
# Calculate total forecasting windows
total_windows = (len(df) - input_length - forecast_horizon + 1) // stride

# Process each window
for i in range(0, len(df) - input_length - forecast_horizon + 1, stride):
    # Extract input window (past data)
    input_window = df.iloc[i:i+input_length]
    
    # Extract target window (future data to predict)
    target_window = df.iloc[i+input_length:i+input_length+forecast_horizon]
    
    # Generate forecasts and store results
    # ...

3. Ensemble Prediction Mechanism
This is where the power of the approach becomes evident:
Multiple Forecasts Per Point: Each future point gets predicted multiple times from different windows
Ensemble Averaging: All predictions are averaged to produce the final output
This approach:
Combines diverse perspectives from multiple windows
Reduces overfitting by averaging out idiosyncrasies of individual forecasts
Provides a more stable and accurate prediction

4. Aggregation of Predictions: For each timestamp, predictions from all windows are averaged
This final step:
Ensures a single, consistent prediction for each point in time
Provides a robust and accurate forecast across the entire dataset
The ensemble effect is critical - it:
Reduces noise in individual predictions
Captures different aspects of the pattern from different contexts
Makes prediction more robust to local irregularities
Improves overall forecast accuracy

3. Context-Aware Anomaly Detection
After generating predictions, the system uses a sophisticated context-aware approach to detect anomalies:
Local Error Analysis
Error Calculation: forecast_errors = np.abs(aligned_predicted - aligned_actual)
Local Statistics: Calculate rolling mean and standard deviation of errors
  error_series = pd.Series(forecast_errors)
  window_size_local = 48  # 12 hours with 15-min data
  rolling_mean = error_series.rolling(window=window_size_local, center=True).mean()
  rolling_std = error_series.rolling(window=window_size_local, center=True).std()

Z-Score Based Flagging
Local Z-Scores: Calculate how unusual each error is in its local context
z_scores = (forecast_errors - rolling_mean) / rolling_std

Anomaly Detection Threshold
  anomaly_flags = (z_scores_local > z_threshold).astype(int)

This approach is particularly powerful because:
It adapts to different error levels in different parts of the time series
It accounts for seasonal or time-dependent variability
It can detect subtle anomalies in low-variance regions
It's less likely to flag normal variations in high-variance regions

4. The Complete Process Flow
Training Phase:
Model is trained on clean data from multiple years
Learns normal water level patterns for each station
Uses early stopping to prevent overfitting

Evaluation Phase:
Apply trained model to new data with potential anomalies
For each station:
Create sliding windows with overlap
Generate predictions from each window
Combine predictions into ensemble forecast
Smooth predictions to reduce noise
Calculate local error statistics
Flag anomalies using adaptive z-score threshold
Calculate performance metrics against ground truth
Provide detailed visualization and analysis

Memory Efficiency:
Processes data in manageable windows
Doesn't require loading entire time series into memory
Scalable to very long time series

'''