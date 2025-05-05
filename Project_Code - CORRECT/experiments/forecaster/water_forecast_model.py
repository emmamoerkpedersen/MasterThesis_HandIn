import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from tqdm.auto import tqdm  # Import tqdm for progress bars

# Add the necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import preprocessing module
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

class ForecastingLSTM(nn.Module):
    """
    Advanced LSTM model for water level forecasting with:
    - Multi-head attention mechanisms
    - Dual-branch architecture for separate time scale processing
    - Residual connections for stability during anomalies
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, 
                 use_attention=False, num_attention_heads=3):
        super(ForecastingLSTM, self).__init__()
        self.model_name = 'ForecastingLSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        self.long_term_emphasis = 0.5  # Default value, will be overridden if specified
        
        # Long-term branch LSTM for capturing long-term dependencies
        self.long_term_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Short-term branch LSTM for handling recent patterns
        self.short_term_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size//2,  # Smaller size for short-term
            num_layers=2,  # Fewer layers for short-term
            batch_first=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # Multi-head attention mechanisms (optional)
        if use_attention:
            # Create multiple attention heads for long-term branch
            self.long_term_attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                ) for _ in range(num_attention_heads)
            ])
            
            # Projection layer to combine attention heads
            self.long_term_attention_combine = nn.Linear(hidden_size * num_attention_heads, hidden_size)
            
            # Simpler attention for short-term branch
            self.short_term_attention = nn.Sequential(
                nn.Linear(hidden_size//2, hidden_size//2),
                nn.Tanh(),
                nn.Linear(hidden_size//2, 1)
            )
        
        # Fusion layer to combine long and short term representations
        self.fusion_layer = nn.Linear(hidden_size + hidden_size//2, hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Store temporal attention weights
        self.temporal_attention_weights = None
        
        # Normalization layers for stability
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
    
    def apply_multi_head_attention(self, lstm_out, attention_heads, attention_combine):
        """Apply multi-head attention and combine results."""
        if not self.use_attention:
            return lstm_out[:, -1, :]
            
        # Apply each attention head
        head_outputs = []
        attention_weights = []
        
        for head in attention_heads:
            # Calculate attention weights for this head
            weights = head(lstm_out)
            weights = torch.softmax(weights, dim=1)
            attention_weights.append(weights)
            
            # Apply attention weights
            attended = torch.bmm(weights.transpose(1, 2), lstm_out)
            head_outputs.append(attended)
        
        # Concatenate all head outputs
        multi_head_output = torch.cat(head_outputs, dim=2)
        
        # Store temporal attention weights (average across heads)
        if self.training:
            self.temporal_attention_weights = torch.mean(torch.stack(attention_weights), dim=0).detach()
        
        # Combine through projection layer
        return attention_combine(multi_head_output).squeeze(1)
    
    def apply_simple_attention(self, lstm_out, attention_layer):
        """Apply simple attention for short-term branch."""
        # Calculate attention weights
        weights = attention_layer(lstm_out)
        weights = torch.softmax(weights, dim=1)
        
        # Apply attention weights
        attended = torch.bmm(weights.transpose(1, 2), lstm_out)
        
        return attended.squeeze(1)
    
    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        
        # Process through long-term branch
        long_term_out, _ = self.long_term_lstm(x)
        
        # Process through short-term branch
        # For short-term, we might want to focus on more recent data
        recent_steps = min(seq_len, 48)  # Look at most recent ~12 hours
        short_term_out, _ = self.short_term_lstm(x[:, -recent_steps:, :])
        
        if self.use_attention:
            # Apply attention to both branches
            long_term_repr = self.apply_multi_head_attention(
                long_term_out, 
                self.long_term_attention_heads,
                self.long_term_attention_combine
            )
            
            short_term_repr = self.apply_simple_attention(
                short_term_out,
                self.short_term_attention
            )
        else:
            # Use last hidden states if no attention
            long_term_repr = long_term_out[:, -1, :]
            short_term_repr = short_term_out[:, -1, :]
        
        # Get the long-term emphasis factor (default to 0.5 if not specified)
        long_term_emphasis = getattr(self, 'long_term_emphasis', 0.5)
        
        # Combine representations with emphasis on long-term
        combined_repr = torch.cat([
            long_term_repr * long_term_emphasis,  # Emphasized long-term
            short_term_repr * (1 - long_term_emphasis)  # De-emphasized short-term
        ], dim=1)
        
        fused_repr = self.fusion_layer(combined_repr)
        
        if self.training:
            fused_repr = self.dropout(fused_repr)
        
        # Apply residual connections and layer normalization
        hidden1 = self.fc1(fused_repr)
        hidden1 = self.layer_norm1(hidden1 + fused_repr)  # Residual connection
        
        hidden2 = self.fc2(hidden1)
        hidden2 = self.layer_norm2(hidden2 + hidden1)  # Another residual connection
        
        # Generate forecast
        forecasts = self.output_layer(hidden2)
        
        return forecasts
    
    def get_temporal_attention(self):
        """Return the current temporal attention weights. Used for temporal attention analysis, else returns None"""
        if self.temporal_attention_weights is None:
            return None
        return self.temporal_attention_weights.cpu().numpy()

class WaterLevelForecaster:
    """
    Main class to handle water level forecasting with anomaly detection.
    """
    def __init__(self, config):
        """
        Initialize the water level forecasting model with anomaly detection.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.anomaly_threshold = config.get('z_score_threshold', 5)
        self.prediction_window = config.get('prediction_window', 24)
        
        # Removed unused attributes:
        # - temporal_attention_history
        # - feature_tiers
        # - confidence_thresholds
        # - recent_predictions
        # - recent_window

    def build_model(self, input_size):
        """
        Build the LSTM model for forecasting.
        
        Args:
            input_size: Number of input features
        """
        # Extract model parameters from config
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        dropout = self.config.get('dropout', 0.2)
        output_size = self.prediction_window  # Predict multiple steps ahead
        use_attention = self.config.get('use_attention', True)
        
        # Advanced model architecture settings
        use_dual_branch = self.config.get('use_dual_branch', False)
        use_residual = self.config.get('use_residual', False)
        long_term_emphasis = self.config.get('long_term_emphasis', 0.5)
        
        # Print summary of model architecture
        print(f"\nBuilding model with the following architecture:")
        print(f"  - Input size: {input_size}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Output size: {output_size}")
        print(f"  - LSTM layers: {num_layers}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Attention: {'Enabled' if use_attention else 'Disabled'}")
        print(f"  - Dual-branch: {'Enabled' if use_dual_branch else 'Disabled'}")
        print(f"  - Residual connections: {'Enabled' if use_residual else 'Disabled'}")
        print(f"  - Long-term emphasis: {long_term_emphasis:.2f}")
        
        # Initialize and return the model
        model = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            use_attention=use_attention
        ).to(self.device)
        
        # Set long-term emphasis
        model.long_term_emphasis = long_term_emphasis
        
        self.model = model
        return model
    
    def train(self, train_data, val_data, project_root, station_id):
        """
        Train the forecasting model with a simplified training approach.
        
        Args:
            train_data: Training data
            val_data: Validation data
            project_root: Root directory of the project
            station_id: ID of the station to use
            
        Returns:
            Trained model
        """
        # Initialize preprocessor if not already
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor(self.config)
        
        if train_data is None or val_data is None:
            # Load and split data if not provided
            train_data, val_data, _ = self.preprocessor.load_and_split_data(project_root, station_id)
        
        # Extract parameters from config
        num_epochs = self.config.get('epochs', 50)  # Reduced default epochs
        patience = self.config.get('patience', 10)
        learning_rate = self.config.get('learning_rate', 0.001)
        batch_size = self.config.get('batch_size', 16)
        
        # Prepare data for training
        X_train_df = train_data[self.preprocessor.feature_cols]
        y_train_df = pd.DataFrame(train_data[self.preprocessor.output_features])
        
        X_val_df = val_data[self.preprocessor.feature_cols]
        y_val_df = pd.DataFrame(val_data[self.preprocessor.output_features])
        
        # Scale data
        print("Scaling training and validation data...")
        X_train_scaled, y_train_scaled = self.preprocessor.feature_scaler.fit_transform(X_train_df, y_train_df)
        X_val_scaled, y_val_scaled = self.preprocessor.feature_scaler.transform(X_val_df, y_val_df)
        
        try:
            # Create sequences with overlap for forecasting
            print("Creating training sequences...")
            X_train, y_train = self.preprocessor._create_overlap_sequences(X_train_scaled, y_train_scaled)
            print("Creating validation sequences...")
            X_val, y_val = self.preprocessor._create_overlap_sequences(X_val_scaled, y_val_scaled)
        except Exception as e:
            print(f"Error creating overlap sequences: {e}")
            print("Falling back to standard sequence creation...")
            X_train, y_train = self.preprocessor._create_sequences(X_train_scaled, y_train_scaled)
            X_val, y_val = self.preprocessor._create_sequences(X_val_scaled, y_val_scaled)
            
            # Adjust model prediction window if needed
            if y_train.shape[1] < self.prediction_window:
                print(f"Warning: Sequence length {y_train.shape[1]} is shorter than prediction window {self.prediction_window}")
                self.prediction_window = y_train.shape[1]
                print(f"Adjusted prediction window to {self.prediction_window}")
        
        # Convert to PyTorch tensors
        print("Converting to PyTorch tensors...")
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Build model if not already built
        input_size = X_train.shape[2]
        if self.model is None:
            print(f"Building model with input size {input_size}...")
            self.build_model(input_size=input_size)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Use a combination of losses for robustness
        mse_criterion = nn.MSELoss()
        robust_criterion = nn.SmoothL1Loss()
        
        # Training loop with simplified approach
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None
        
        # Keep track of losses for plotting
        training_loss_history = []
        validation_loss_history = []
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training
            self.model.train()
            train_loss = 0
            
            # Process in batches
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            
            # Create progress bar for batches
            batch_pbar = tqdm(range(num_batches), leave=False, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            
            for i in batch_pbar:
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Add a small amount of noise for robustness (10% of the time)
                if np.random.random() < 0.1:
                    noise = torch.randn_like(X_batch) * 0.05
                    X_batch = X_batch + noise
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                
                # Reshape for loss calculation if needed
                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view(y_batch.shape)
                
                # Combine MSE and Huber loss for robustness
                mse_loss = mse_criterion(y_pred, y_batch)
                robust_loss = robust_criterion(y_pred, y_batch)
                
                # Use 70% MSE, 30% robust loss
                loss = 0.7 * mse_loss + 0.3 * robust_loss
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
            avg_train_loss = train_loss / num_batches
            training_loss_history.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                # Standard validation
                val_preds = self.model(X_val)
                
                # Reshape if needed
                if val_preds.shape != y_val.shape:
                    val_preds = val_preds.view(y_val.shape)
                
                # Calculate validation loss
                val_loss = 0.7 * mse_criterion(val_preds, y_val).item() + 0.3 * robust_criterion(val_preds, y_val).item()
            
            validation_loss_history.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.6f}", 
                "val_loss": f"{val_loss:.6f}",
                "no_improve": epochs_no_improve
            })
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_weights = self.model.state_dict().copy()
                # Add a checkmark to indicate improvement
                epoch_pbar.set_postfix({
                    "train_loss": f"{avg_train_loss:.6f}", 
                    "val_loss": f"{val_loss:.6f} ✓",
                    "no_improve": 0
                })
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        
        return self.model
    
    def predict(self, test_data, steps_ahead=None, stride_mode='default'):
        """
        Make forecasts for the test data.
        
        Args:
            test_data: Test data
            steps_ahead: Number of steps to forecast ahead (defaults to config value)
            stride_mode: Stride mode for prediction windows:
                         'default': use stride=1 (traditional sliding window)
                         'prediction_window': use stride equal to prediction window (more realistic operational scenario)
                         An integer can also be specified for a custom stride
                
        Returns:
            Dictionary with forecasts, anomalies, original data, and clean data
        """
        print("Preparing for prediction...")
        self.model.eval()
        
        if steps_ahead is None:
            steps_ahead = self.prediction_window
        else:
            # Ensure steps_ahead does not exceed the model's output capacity
            steps_ahead = min(steps_ahead, self.prediction_window)
            print(f"Using {steps_ahead} steps ahead for forecasting")
        
        # Determine stride based on mode
        if stride_mode == 'default':
            stride = 1
        elif stride_mode == 'prediction_window':
            stride = steps_ahead
        elif isinstance(stride_mode, int):
            stride = stride_mode
        else:
            print(f"Unrecognized stride mode '{stride_mode}', using default (stride=1)")
            stride = 1
            
        print(f"Using stride of {stride} for predictions")
        
        # Store a copy of the original data
        original_data = test_data.copy()
        
        # Create working copy of data
        working_data = test_data.copy()
        
        # Prepare features and targets
        X_test_df = working_data[self.preprocessor.feature_cols]
        y_test_df = pd.DataFrame(working_data[self.preprocessor.output_features])
        
        # Scale data
        print("Scaling test data...")
        X_test_scaled, y_test_scaled = self.preprocessor.feature_scaler.transform(X_test_df, y_test_df)
        
        # Get sequence length from config
        sequence_length = self.config.get('sequence_length', 200)
        
        # Optimize prediction process by batching
        batch_size = self.config.get('prediction_batch_size', 64)  # Default to 64 if not specified
        
        # Determine how many sliding windows we'll need based on the stride
        num_windows = (len(X_test_scaled) - sequence_length) // stride + 1
        
        print(f"Making predictions with {num_windows} sliding windows using stride={stride}...")
        
        forecasts = []
        timestamps = []
        
        # Process in batches for faster prediction
        with torch.no_grad():
            # Create a progress bar for prediction windows
            window_pbar = tqdm(range(0, num_windows, batch_size), desc="Predicting", unit="batch")
            
            for batch_start in window_pbar:
                batch_end = min(batch_start + batch_size, num_windows)
                batch_forecasts = []
                batch_timestamps = []
                
                # Create batch of sequences
                batch_sequences = []
                for batch_idx in range(batch_start, batch_end):
                    # Calculate the actual index in the dataset based on stride
                    i = batch_idx * stride
                    
                    # Extract sequence
                    X_seq = X_test_scaled[i:i+sequence_length]
                    batch_sequences.append(X_seq)
                    batch_timestamps.append(working_data.index[i+sequence_length-1])
                
                # Convert batch to tensor
                if batch_sequences:
                    X_batch = torch.FloatTensor(np.array(batch_sequences)).to(self.device)
                    
                    # Make predictions for the entire batch at once
                    batch_preds = self.model(X_batch)
                    
                    # Add predictions to results
                    for pred in batch_preds.cpu().numpy():
                        # Only keep predictions up to steps_ahead
                        batch_forecasts.append(pred[:steps_ahead])
                
                forecasts.extend(batch_forecasts)
                timestamps.extend(batch_timestamps)
                
                # Update progress bar
                window_pbar.set_postfix({"windows": f"{batch_end}/{num_windows}"})
        
        # Convert to numpy arrays
        forecasts = np.array(forecasts)
        
        print("Processing prediction results...")
        
        # For each timestep, we now have predictions for the next 'steps_ahead' steps
        # Convert to DataFrame for easier manipulation
        forecasts_df = pd.DataFrame(
            forecasts, 
            index=timestamps
        )
        
        # Inverse transform the forecasts to original scale
        original_scale_forecasts = []
        for i in range(forecasts.shape[1]):
            col_forecasts = forecasts_df.iloc[:, i].values.reshape(-1, 1)
            original_scale_col = self.preprocessor.feature_scaler.inverse_transform_target(col_forecasts)
            original_scale_forecasts.append(original_scale_col)
        
        # Combine into a single array
        original_scale_forecasts = np.hstack(original_scale_forecasts)
        
        # Create a new DataFrame with the forecasts
        forecast_df = pd.DataFrame(
            original_scale_forecasts, 
            index=timestamps,
            columns=[f'step_{i+1}' for i in range(forecasts.shape[1])]
        )
        
        # If using stride > 1, we need to "forward fill" the forecasts
        # to get predictions for every timestep
        if stride > 1:
            # Get all possible timestamps from the data
            all_timestamps = original_data.index
            
            # Only use timestamps that are within our prediction range
            valid_timestamps = all_timestamps[all_timestamps >= forecast_df.index[0]]
            
            # Create a new DataFrame that will hold forecasts for all timestamps
            filled_forecast_df = pd.DataFrame(index=valid_timestamps, 
                                             columns=forecast_df.columns)
            
            # For each prediction and step
            for t_idx, timestamp in enumerate(forecast_df.index):
                # Place each step at its actual future timestamp
                for step in range(1, steps_ahead + 1):
                    step_col = f'step_{step}'
                    
                    # Calculate the target timestamp (when this prediction is for)
                    try:
                        # Try to add the appropriate number of time units
                        target_idx = all_timestamps.get_loc(timestamp) + step
                        if target_idx < len(all_timestamps):
                            target_timestamp = all_timestamps[target_idx]
                            # Place the prediction at its actual future position
                            filled_forecast_df.loc[target_timestamp, step_col] = forecast_df.loc[timestamp, step_col]
                    except (KeyError, TypeError):
                        # Skip if timestamp not found or index not compatible
                        continue
            
            # For visualization clarity, we'll also add the step_1 predictions at their origin points
            for timestamp in forecast_df.index:
                filled_forecast_df.loc[timestamp, 'step_1'] = forecast_df.loc[timestamp, 'step_1']
            
            # Now use this filled version for further processing
            forecast_df = filled_forecast_df.copy()
        
        # Extract data for output
        actual_data = working_data[self.preprocessor.output_features]
        
        # Align with forecast index
        actual_data = actual_data.reindex(forecast_df.index)
        
        # Get the clean data for the same timestamps
        clean_data = original_data[self.preprocessor.output_features].reindex(forecast_df.index)
        
        print("Detecting anomalies...")
        # Detect anomalies by comparing actual data with predictions
        detected_anomalies = self.detect_anomalies(actual_data, forecast_df['step_1'])
        
        print("Prediction completed successfully.")
        
        # Return results
        results = {
            'timestamps': timestamps,
            'clean_data': clean_data,
            'error_injected_data': actual_data,
            'forecasts': forecast_df,
            'detected_anomalies': detected_anomalies,
            'stride_used': stride
        }
        
        return results
    
    def detect_anomalies(self, actual_values, predicted_values):
        """
        Detect anomalies by comparing actual values with predicted values.
        
        Args:
            actual_values: Actual water level values
            predicted_values: Predicted water level values
            
        Returns:
            DataFrame with anomaly flags and scores, including confidence levels
        """
        # Print debug information about input data types
        print(f"\nAnomaly Detection - Input Types:")
        print(f"actual_values type: {type(actual_values)}, shape: {getattr(actual_values, 'shape', 'N/A')}")
        print(f"predicted_values type: {type(predicted_values)}, shape: {getattr(predicted_values, 'shape', 'N/A')}")
        
        # Ensure we're working with flat numpy arrays
        try:
            actual = actual_values.values.flatten() if hasattr(actual_values, 'values') else np.array(actual_values).flatten()
            predicted = predicted_values.values.flatten() if hasattr(predicted_values, 'values') else np.array(predicted_values).flatten()
            
            # Make sure we have compatible data types for np.isnan()
            actual = actual.astype(float)
            predicted = predicted.astype(float)
            
            print(f"Converted arrays - actual: {actual.shape}, predicted: {predicted.shape}")
            
            # Handle NaN values by replacing with their corresponding pair or with mean
            try:
                mask = np.isnan(actual) | np.isnan(predicted)
                if mask.any():
                    # Fill NaNs in either series with the corresponding value from the other series if available
                    # otherwise use the mean of the series
                    actual_mean = np.nanmean(actual)
                    predicted_mean = np.nanmean(predicted)
                    
                    for i in np.where(mask)[0]:
                        if np.isnan(actual[i]) and not np.isnan(predicted[i]):
                            actual[i] = predicted[i]
                        elif np.isnan(predicted[i]) and not np.isnan(actual[i]):
                            predicted[i] = actual[i]
                        else:
                            # Both are NaN, fill with means
                            actual[i] = actual_mean
                            predicted[i] = predicted_mean
            except (TypeError, ValueError) as e:
                print(f"Warning: Error handling NaN values: {e}")
                # More robust fallback - replace any NaN values
                actual = np.nan_to_num(actual, nan=0.0)
                predicted = np.nan_to_num(predicted, nan=0.0)
        
        except Exception as e:
            print(f"Error preparing data for anomaly detection: {e}")
            print(f"actual_values type: {type(actual_values)}")
            if hasattr(actual_values, 'values'):
                print(f"actual_values.values type: {type(actual_values.values)}")
            print(f"predicted_values type: {type(predicted_values)}")
            if hasattr(predicted_values, 'values'):
                print(f"predicted_values.values type: {type(predicted_values.values)}")
            
            # Re-raise the exception for proper debugging
            raise
        
        # Calculate absolute residuals (actual - predicted)
        absolute_residuals = np.abs(actual - predicted)
        
        # Calculate percentage error for scaling detection
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_error = np.abs((actual - predicted) / (predicted + 1e-10)) * 100
        
        # Replace inf/nan with large values
        percent_error = np.nan_to_num(percent_error, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # Calculate metrics for anomaly detection using robust statistics
        # For absolute errors
        median_residual = np.median(absolute_residuals)
        mad_residual = np.median(np.abs(absolute_residuals - median_residual)) * 1.4826  # Factor for normal distribution
        
        # For percentage errors
        median_percent = np.median(percent_error)
        mad_percent = np.median(np.abs(percent_error - median_percent)) * 1.4826
        
        # Handle cases when MAD is zero or close to zero
        mad_residual = max(mad_residual, 1e-6)
        mad_percent = max(mad_percent, 1e-6)
        
        # Calculate z-scores for both metrics
        z_scores_abs = (absolute_residuals - median_residual) / mad_residual
        z_scores_pct = (percent_error - median_percent) / mad_percent
        
        # Add simplified streak detection
        streak_window = 5  # Look at consecutive points
        streak_factor = np.zeros_like(z_scores_abs)
        
        for i in range(streak_window, len(z_scores_abs)):
            # Look at maximum z-score in previous window
            prev_max_z = max(
                np.max(z_scores_abs[i-streak_window:i]),
                np.max(z_scores_pct[i-streak_window:i])
            )
            
            # If previous window had high z-scores, increase streak factor
            if prev_max_z > self.anomaly_threshold * 0.7:
                streak_factor[i] = min(1.0, prev_max_z / self.anomaly_threshold)
        
        # Calculate a combined score with simpler streak influence
        combined_z = np.maximum(z_scores_abs, z_scores_pct)
        combined_z = combined_z * (1 + streak_factor * 0.5)  # Lower impact from streak factor
        
        # Calculate absolute error (to avoid flagging very small changes)
        abs_error = np.abs(actual - predicted)
        
        # Calculate percent error
        percent_error = np.abs(actual - predicted) / (np.abs(predicted) + 1e-10) * 100
        
        # Identify anomalies based on multiple criteria
        # 1. Combined z-score must exceed threshold
        # 2. Either absolute error is significant OR percent error is significant
        threshold_exceeded = combined_z > self.anomaly_threshold
        significant_abs_error = abs_error > 1.0  # Minimum 1.0 unit absolute error
        significant_pct_error = percent_error > 15.0  # Minimum 15% error
        
        # Apply all criteria
        anomaly_flags = threshold_exceeded & (significant_abs_error | significant_pct_error)
        
        # Define confidence levels based on z-score
        # Instead of anomaly types, we'll use confidence levels
        confidence_levels = np.full(len(actual), 'normal', dtype=object)
        
        # Define thresholds for confidence levels
        low_confidence_threshold = self.anomaly_threshold
        medium_confidence_threshold = self.anomaly_threshold * 1.3
        high_confidence_threshold = self.anomaly_threshold * 1.8
        
        # Assign confidence levels based on z-score
        confidence_levels[anomaly_flags & (combined_z < medium_confidence_threshold)] = 'low'
        confidence_levels[anomaly_flags & (combined_z >= medium_confidence_threshold) & (combined_z < high_confidence_threshold)] = 'medium'
        confidence_levels[anomaly_flags & (combined_z >= high_confidence_threshold)] = 'high'
        
        # Create DataFrame with anomaly information
        anomalies = pd.DataFrame({
            'actual': actual,
            'predicted': predicted,
            'abs_error': absolute_residuals,
            'percent_error': percent_error,
            'z_score_abs': z_scores_abs,
            'z_score_pct': z_scores_pct,
            'streak_factor': streak_factor,
            'z_score': combined_z,
            'is_anomaly': anomaly_flags,
            'confidence': confidence_levels
        }, index=actual_values.index if hasattr(actual_values, 'index') else None)
        
        return anomalies
        
    def save_model(self, path):
        """
        Save the trained model to a file.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_architecture': {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'output_size': self.model.output_size
                }
            }, path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def load_model(self, path):
        """
        Load a trained model from a file.
        
        Args:
            path: Path to load the model from
        """
        if not Path(path).exists():
            print(f"Model file {path} not found")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update config if needed
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Check if attention should be enabled based on model state dict
        if any('attention' in key for key in checkpoint['model_state_dict'].keys()):
            self.config['use_attention'] = True
            print("Enabling attention based on model weights")
            
        # Get number of attention heads if available
        num_attention_heads = 8  # Default
        attention_head_keys = [k for k in checkpoint['model_state_dict'].keys() if 'long_term_attention_heads' in k]
        if attention_head_keys:
            # Extract head count from pattern 'long_term_attention_heads.N...'
            head_indices = set()
            for key in attention_head_keys:
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    head_indices.add(int(parts[1]))
            if head_indices:
                num_attention_heads = max(head_indices) + 1
                print(f"Detected {num_attention_heads} attention heads in model")
        
        # Get model architecture
        if 'model_architecture' in checkpoint:
            arch = checkpoint['model_architecture']
            self.model = ForecastingLSTM(
                input_size=arch['input_size'],
                hidden_size=arch['hidden_size'],
                num_layers=arch['num_layers'],
                output_size=arch['output_size'],
                dropout=self.config.get('dropout', 0.2),
                use_attention=self.config.get('use_attention', False),
                num_attention_heads=num_attention_heads
            ).to(self.device)
        else:
            # Backward compatibility
            print("Model architecture not found in checkpoint, using config values")
            input_size = self.config.get('input_size')
            if input_size is None:
                print("Cannot load model: input_size not provided in config")
                return False
            self.build_model(input_size)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {path}")
        return True
    
   