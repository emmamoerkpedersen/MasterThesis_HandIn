import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add the necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import preprocessing module
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

class ForecastingLSTM(nn.Module):
    """
    LSTM model for water level forecasting with configurable components:
    - Optional filter LSTM stream
    - Multi-head attention mechanisms
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, 
                 use_feature_importance=False, use_filter_lstm=False, use_attention=False,
                 num_attention_heads=4):  # Added num_attention_heads parameter
        super(ForecastingLSTM, self).__init__()
        self.model_name = 'ForecastingLSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_feature_importance = use_feature_importance
        self.use_filter_lstm = use_filter_lstm
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        
        # Main LSTM for processing raw input
        self.main_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Filter LSTM for maintaining stable state (optional)
        if use_filter_lstm:
            self.filter_lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Feature importance layers - only if enabled
        if use_feature_importance:
            self.feature_importance = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
                nn.Softmax(dim=1)  # Ensure weights sum to 1 across features
            )
        
        # Multi-head attention mechanisms (optional)
        if use_attention:
            # Create multiple attention heads for main LSTM
            self.main_attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                ) for _ in range(num_attention_heads)
            ])
            
            # Projection layer to combine attention heads
            self.main_attention_combine = nn.Linear(hidden_size * num_attention_heads, hidden_size)
            
            if use_filter_lstm:
                # Create multiple attention heads for filter LSTM
                self.filter_attention_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1)
                    ) for _ in range(num_attention_heads)
                ])
                
                # Projection layer to combine filter attention heads
                self.filter_attention_combine = nn.Linear(hidden_size * num_attention_heads, hidden_size)
        
        # Gating mechanism to combine streams (only if using filter_lstm)
        if use_filter_lstm:
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to map to output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Store feature importance scores
        self.feature_importance_scores = None
        self.temporal_attention_weights = None
    
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
    
    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        
        # Calculate feature importance only if enabled
        if self.use_feature_importance:
            # Reshape to (batch_size * seq_len, num_features)
            x_reshaped = x.reshape(-1, num_features)
            feature_weights = self.feature_importance(x_reshaped)
            
            # Reshape back to (batch_size, seq_len, num_features)
            feature_weights = feature_weights.reshape(batch_size, seq_len, num_features)
            
            # Store average feature importance scores
            if self.training:
                self.feature_importance_scores = feature_weights.mean(dim=(0, 1)).detach()
            
            # Apply feature weights to input
            x_weighted = x * feature_weights
        else:
            x_weighted = x
            self.feature_importance_scores = None
        
        # Process through main LSTM
        main_out, _ = self.main_lstm(x_weighted)
        
        if self.use_attention:
            main_attended = self.apply_multi_head_attention(
                main_out, 
                self.main_attention_heads,
                self.main_attention_combine
            )
        else:
            main_attended = main_out[:, -1, :]  # Use last hidden state if no attention
        
        if self.use_filter_lstm:
            # Process through filter LSTM
            filter_out, _ = self.filter_lstm(x_weighted)
            
            if self.use_attention:
                filter_attended = self.apply_multi_head_attention(
                    filter_out,
                    self.filter_attention_heads,
                    self.filter_attention_combine
                )
            else:
                filter_attended = filter_out[:, -1, :]  # Use last hidden state if no attention
            
            # Calculate gating weights and combine streams
            combined = torch.cat([main_attended, filter_attended], dim=1)
            gate = self.gate_network(combined)
            final_output = gate * main_attended + (1 - gate) * filter_attended
        else:
            final_output = main_attended
        
        if self.training:
            final_output = self.dropout(final_output)
        
        # Generate forecast
        forecasts = self.fc(final_output)
        
        return forecasts
    
    def get_feature_importance(self):
        """Return the current feature importance scores. Used for feature importance analysis, else returns None"""
        if self.feature_importance_scores is None:
            return None
        return self.feature_importance_scores.cpu().numpy()
    
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
        self.feature_importance_history = []
        self.temporal_attention_history = []
        
        # Define feature tiers based on importance
        self.feature_tiers = {
            'tier1': [
                'water_level_ma_3h',
                'water_level_ma_12h',
                'water_level_lag_6h'
            ],
            'tier2': [
                'water_level_ma_roc_6h',
                'water_level_roc_12h',
                'water_level_ma_roc_24h'
            ],
            'tier3': [
                'water_level_lag_72h',
                'water_level_ma_48h',
                'water_level_ma_96h'
            ]
        }
        
        # Confidence thresholds for each tier
        self.confidence_thresholds = {
            'tier1': 0.8,  # High confidence needed to stop at tier 1
            'tier2': 0.6,  # Medium confidence for tier 2
            'tier3': 0.4   # Lower threshold for tier 3
        }

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
        use_feature_importance = self.config.get('use_feature_importance', False)
        use_filter_lstm = self.config.get('use_filter_lstm', True)
        use_attention = self.config.get('use_attention', True)
        
        # Initialize and return the model
        model = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            use_feature_importance=use_feature_importance,
            use_filter_lstm=use_filter_lstm,
            use_attention=use_attention
        ).to(self.device)
        
        self.model = model
        return model
    
    def train(self, train_data, val_data, project_root, station_id):
        """
        Train the forecasting model.
        
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
        num_epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        learning_rate = self.config.get('learning_rate', 0.001)
        batch_size = self.config.get('batch_size', 16)
        
        # Prepare data for training
        X_train_df = train_data[self.preprocessor.feature_cols]
        y_train_df = pd.DataFrame(train_data[self.preprocessor.output_features])
        
        X_val_df = val_data[self.preprocessor.feature_cols]
        y_val_df = pd.DataFrame(val_data[self.preprocessor.output_features])
        
        # Scale data
        X_train_scaled, y_train_scaled = self.preprocessor.feature_scaler.fit_transform(X_train_df, y_train_df)
        X_val_scaled, y_val_scaled = self.preprocessor.feature_scaler.transform(X_val_df, y_val_df)
        
        try:
            # Create sequences with overlap for forecasting
            X_train, y_train = self.preprocessor._create_overlap_sequences(X_train_scaled, y_train_scaled)
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
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Build model if not already built
        input_size = X_train.shape[2]
        if self.model is None:
            self.build_model(input_size=input_size)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.SmoothL1Loss()
        
        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None
        
        feature_importance_per_epoch = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            epoch_importance = np.zeros(len(self.preprocessor.feature_cols))
            
            # Process in batches
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            
            for i in range(num_batches):

                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                
                # Get feature importance for this batch
                importance = self.model.get_feature_importance()
                if importance is not None:
                    epoch_importance += importance
                
                # Reshape for loss calculation if needed
                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view(y_batch.shape)
                
                # Create mask for valid targets (non-NaN)
                valid_mask = ~torch.isnan(y_batch)
                
                # Only calculate loss for valid targets
                if valid_mask.any():
                    valid_y_pred = y_pred[valid_mask]
                    valid_y_batch = y_batch[valid_mask]
                    loss = criterion(valid_y_pred, valid_y_batch)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
            
            avg_train_loss = train_loss / num_batches
            
            # Average feature importance for this epoch
            epoch_importance /= num_batches
            feature_importance_per_epoch.append(epoch_importance)
            
            # Validation
            self.model.eval()
            val_loss = 0
            valid_val_samples = 0
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(X_val))
                    
                    X_batch = X_val[start_idx:end_idx]
                    y_batch = y_val[start_idx:end_idx]
                    
                    y_pred = self.model(X_batch)
                    
                    # Reshape if needed
                    if y_pred.shape != y_batch.shape:
                        y_pred = y_pred.view(y_batch.shape)
                    
                    # Create mask for valid targets (non-NaN)
                    valid_mask = ~torch.isnan(y_batch)
                    
                    # Only calculate loss for valid targets
                    if valid_mask.any():
                        valid_y_pred = y_pred[valid_mask]
                        valid_y_batch = y_batch[valid_mask]
                        loss = criterion(valid_y_pred, valid_y_batch)
                        val_loss += loss.item()
                        valid_val_samples += 1
            
            # Calculate average validation loss only over valid samples
            avg_val_loss = val_loss / valid_val_samples if valid_val_samples > 0 else float('inf')
            
            # Print progress with feature importance
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                # Print top 5 most important features
                feature_importance = list(zip(self.preprocessor.feature_cols, epoch_importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                print("\nTop 5 important features:")
                for feature, importance in feature_importance[:5]:
                    print(f"{feature}: {importance:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_weights = self.model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Store feature importance history
        self.feature_importance_history = feature_importance_per_epoch
        
        # Load best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        
        # Print final feature importance analysis
        final_importance = np.mean(feature_importance_per_epoch[-10:], axis=0)  # Average of last 10 epochs
        feature_importance = list(zip(self.preprocessor.feature_cols, final_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nFinal Feature Importance Analysis:")
        print("\nTop 10 Most Important Features:")
        for feature, importance in feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")
        
        print("\nLeast Important Features:")
        for feature, importance in feature_importance[-5:]:
            print(f"{feature}: {importance:.4f}")
        
        return self.model
    
    def predict(self, test_data, steps_ahead=None, inject_errors=False, error_periods=None):
        """
        Make forecasts for the test data.
        
        Args:
            test_data: Test data
            steps_ahead: Number of steps to forecast ahead (defaults to config value)
            inject_errors: Whether to inject synthetic errors into the test data
            error_periods: List of dictionaries with error periods and types
                Each dict should have: 'start', 'end', 'type', 'magnitude'
                Types: 'noise', 'offset', 'scaling', 'missing'
            
        Returns:
            Dictionary with forecasts, anomalies, original data, and error-injected data
        """
        self.model.eval()
        prediction_importance = []
        
        if steps_ahead is None:
            steps_ahead = self.prediction_window
        
        # Store a copy of the original clean data
        original_clean_data = test_data.copy()
        
        # Create working copy of data
        working_data = test_data.copy()
        
        # Inject errors if requested
        if inject_errors:
            working_data = self._inject_errors(working_data, error_periods)
        
        # Store the data with errors
        data_with_errors = working_data.copy()
        
        # Prepare test data
        X_test_df = working_data[self.preprocessor.feature_cols]
        y_test_df = pd.DataFrame(working_data[self.preprocessor.output_features])
        
        # Scale data
        X_test_scaled, y_test_scaled = self.preprocessor.feature_scaler.transform(X_test_df, y_test_df)
        
        # Get sequence length from config
        sequence_length = self.config.get('sequence_length', 200)
        
        # We'll make predictions for each timestep in the test data
        # by using a sliding window approach
        forecasts = []
        timestamps = []
        
        with torch.no_grad():
            for i in range(len(X_test_scaled) - sequence_length + 1):
                # Extract sequence
                X_seq = X_test_scaled[i:i+sequence_length]
                
                # Convert to tensor and reshape
                X_seq_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                
                # Make prediction
                pred = self.model(X_seq_tensor)
                
                # Get feature importance for this prediction
                importance = self.model.get_feature_importance()
                if importance is not None:
                    prediction_importance.append(importance)
                
                # Store the result
                forecasts.append(pred.cpu().numpy()[0])
                timestamps.append(working_data.index[i+sequence_length-1])
        
        # Convert to numpy arrays
        forecasts = np.array(forecasts)
        
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
        
        # Extract original data (with errors if injected)
        error_injected_data = data_with_errors[self.preprocessor.output_features].loc[forecast_df.index]
        
        # Get the clean data for the same timestamps
        clean_data = original_clean_data[self.preprocessor.output_features].loc[forecast_df.index]
        
        # Detect anomalies by comparing error-injected data with predictions
        if inject_errors:
            # Compare predictions to clean data to evaluate prediction quality
            prediction_quality = self.detect_anomalies(clean_data, forecast_df['step_1'])
            # Compare error-injected data to clean data to identify true anomalies
            true_anomalies = self.detect_anomalies(error_injected_data, clean_data)
            # Compare error-injected data to predictions to see what the model detects
            detected_anomalies = self.detect_anomalies(error_injected_data, forecast_df['step_1'])
        else:
            # Regular anomaly detection on the data
            detected_anomalies = self.detect_anomalies(error_injected_data, forecast_df['step_1'])
            true_anomalies = None
            prediction_quality = None
        
        # Average feature importance during prediction
        if prediction_importance:
            avg_prediction_importance = np.mean(prediction_importance, axis=0)
            feature_importance = list(zip(self.preprocessor.feature_cols, avg_prediction_importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\nFeature Importance During Prediction:")
            print("\nTop 10 Most Important Features:")
            for feature, importance in feature_importance[:10]:
                print(f"{feature}: {importance:.4f}")
        
        # Return results
        results = {
            'timestamps': timestamps,
            'clean_data': clean_data,
            'error_injected_data': error_injected_data,
            'forecasts': forecast_df,
            'detected_anomalies': detected_anomalies,
            'true_anomalies': true_anomalies,
            'prediction_quality': prediction_quality,
            'feature_importance': feature_importance if prediction_importance else None
        }
        
        return results
    
    def _inject_errors(self, data, error_periods=None):
        """
        Inject synthetic errors into the data.
        
        Args:
            data: DataFrame with water level data
            error_periods: List of dictionaries with error periods and types
                Each dict should have: 'start', 'end', 'type', 'magnitude'
                Types: 'noise', 'offset', 'scaling', 'missing'
            
        Returns:
            DataFrame with injected errors
        """
        # Use default error periods if none provided
        if error_periods is None:
            error_periods = [
                {'start': '2022-02-10', 'end': '2022-03-01', 'type': 'offset', 'magnitude': 100},
                {'start': '2022-05-15', 'end': '2022-06-01', 'type': 'scaling', 'magnitude': 1.5},
                {'start': '2022-09-01', 'end': '2022-09-15', 'type': 'noise', 'magnitude': 50},
                {'start': '2022-11-01', 'end': '2022-11-15', 'type': 'missing', 'magnitude': 0}
            ]
        
        # Get the output feature column name
        output_feature = self.preprocessor.output_features
        if isinstance(output_feature, list):
            output_feature = output_feature[0]
        
        # Create a working copy of the data to modify
        modified_data = data.copy()
        
        # Get all columns that contain the water level data (including lagged features)
        water_level_cols = [col for col in modified_data.columns 
                           if output_feature in col or 
                           (col.startswith('water_level_lag') and 'h' in col)]
        
        # Inject errors for each specified period
        for period in error_periods:
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])
            error_type = period['type']
            magnitude = period['magnitude']
            
            # Get mask for the period
            mask = (modified_data.index >= start) & (modified_data.index <= end)
            
            # Skip if no data points in this period
            if not mask.any():
                continue
            
            # Apply different types of errors to all water level columns
            for col in water_level_cols:
                if error_type == 'noise':
                    # Generate random noise with the correct shape
                    noise = np.random.normal(0, magnitude, size=sum(mask))
                    # Apply noise directly
                    modified_data.loc[mask, col] = modified_data.loc[mask, col].values + noise
                
                elif error_type == 'offset':
                    # Apply offset
                    modified_data.loc[mask, col] = modified_data.loc[mask, col].values + magnitude
                
                elif error_type == 'scaling':
                    # Apply scaling
                    modified_data.loc[mask, col] = modified_data.loc[mask, col].values * magnitude
                
                elif error_type == 'missing':
                    # Set values to NaN (simulating missing data)
                    modified_data.loc[mask, col] = np.nan
        
        return modified_data
    
    def detect_anomalies(self, actual_values, predicted_values):
        """
        Detect anomalies by comparing actual values with predicted values.
        
        Args:
            actual_values: Actual water level values
            predicted_values: Predicted water level values
            
        Returns:
            DataFrame with anomaly flags and scores, including anomaly type classification
        """
        # Ensure we're working with flat arrays
        actual = actual_values.values.flatten() if hasattr(actual_values, 'values') else np.array(actual_values).flatten()
        predicted = predicted_values.values.flatten() if hasattr(predicted_values, 'values') else np.array(predicted_values).flatten()
        
        # Handle NaN values by replacing with their corresponding pair or with mean
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
        
        # Calculate absolute residuals (actual - predicted)
        absolute_residuals = np.abs(actual - predicted)
        
        # Calculate percentage error for scaling detection
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_error = np.abs((actual - predicted) / (predicted + 1e-10)) * 100
        
        # Replace inf/nan with large values
        percent_error = np.nan_to_num(percent_error, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        # Calculate metrics for anomaly detection using robust statistics
        # For absolute errors (good for offset detection)
        median_residual = np.median(absolute_residuals)
        mad_residual = np.median(np.abs(absolute_residuals - median_residual)) * 1.4826  # Factor for normal distribution
        
        # For percentage errors (good for scaling detection)
        median_percent = np.median(percent_error)
        mad_percent = np.median(np.abs(percent_error - median_percent)) * 1.4826
        
        # Handle cases when MAD is zero or close to zero
        mad_residual = max(mad_residual, 1e-6)
        mad_percent = max(mad_percent, 1e-6)
        
        # Calculate z-scores for both metrics
        z_scores_abs = (absolute_residuals - median_residual) / mad_residual
        z_scores_pct = (percent_error - median_percent) / mad_percent
        
        # Determine anomaly type based on which z-score is higher
        offset_anomalies = z_scores_abs > z_scores_pct
        scaling_anomalies = z_scores_pct > z_scores_abs
        
        # Calculate a combined score favoring the higher of the two
        z_scores = np.maximum(z_scores_abs, z_scores_pct)
        
        # Identify anomalies based on threshold
        anomaly_flags = z_scores > self.anomaly_threshold
        
        # Determine anomaly types
        anomaly_types = np.full(len(actual), 'normal', dtype=object)
        anomaly_types[anomaly_flags & offset_anomalies] = 'offset'
        anomaly_types[anomaly_flags & scaling_anomalies] = 'scaling'
        
        # Create DataFrame with anomaly information
        anomalies = pd.DataFrame({
            'actual': actual,
            'predicted': predicted,
            'abs_error': absolute_residuals,
            'percent_error': percent_error,
            'z_score_abs': z_scores_abs,
            'z_score_pct': z_scores_pct,
            'z_score': z_scores,
            'is_anomaly': anomaly_flags,
            'anomaly_type': anomaly_types
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
        
        # Get model architecture
        if 'model_architecture' in checkpoint:
            arch = checkpoint['model_architecture']
            self.model = ForecastingLSTM(
                input_size=arch['input_size'],
                hidden_size=arch['hidden_size'],
                num_layers=arch['num_layers'],
                output_size=arch['output_size'],
                dropout=self.config.get('dropout', 0.2)
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
    
    def _detect_anomalies(self, predictions, actual_values, threshold_std=3.0):
        """
        Detect anomalies in predictions based on prediction errors.
        
        Args:
            predictions (pd.Series): Model predictions
            actual_values (pd.Series): Actual water level values
            threshold_std (float): Number of standard deviations for anomaly threshold
            
        Returns:
            pd.DataFrame: DataFrame containing anomaly information
        """
        # Calculate prediction errors
        errors = actual_values - predictions
        error_mean = errors.mean()
        error_std = errors.std()
        
        # Calculate thresholds
        upper_threshold = error_mean + threshold_std * error_std
        lower_threshold = error_mean - threshold_std * error_std
        
        # Identify anomalies
        anomalies = pd.DataFrame(index=predictions.index)
        anomalies['error'] = errors
        anomalies['is_anomaly'] = (errors > upper_threshold) | (errors < lower_threshold)
        anomalies['anomaly_type'] = 'normal'
        anomalies.loc[errors > upper_threshold, 'anomaly_type'] = 'positive_spike'
        anomalies.loc[errors < lower_threshold, 'anomaly_type'] = 'negative_spike'
        anomalies['error_magnitude'] = np.abs(errors)
        
        return anomalies[anomalies['is_anomaly']]

    def predict_iteratively(self, X, y=None, max_iterations=5, convergence_threshold=0.01):
        """
        Make predictions iteratively, using previous predictions as features.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Actual values for tracking errors
            max_iterations (int): Maximum number of iterations
            convergence_threshold (float): Threshold for prediction change to stop iterations
            
        Returns:
            dict: Dictionary containing prediction results and metadata
        """
        results = {
            'timestamps': X.index,
            'clean_data': pd.DataFrame(X[self.preprocessor.output_features]) if y is None else pd.DataFrame(y),
            'predictions_by_iteration': pd.DataFrame(index=X.index),
            'feature_importance': [],
            'convergence_info': {}
        }
        
        # Initial prediction
        current_features = X.copy()
        initial_pred_results = self.predict(current_features)
        # Extract the first step predictions from the forecasts DataFrame
        initial_pred = initial_pred_results['forecasts']['step_1']
        results['predictions_by_iteration']['iteration_1'] = initial_pred
        
        # Store initial prediction in forecasts
        results['forecasts'] = pd.DataFrame(index=X.index)
        results['forecasts']['step_1'] = initial_pred
        
        # Track feature importance if available
        if 'feature_importance' in initial_pred_results:
            results['feature_importance'].append({
                'iteration': 1,
                'importances': initial_pred_results['feature_importance']
            })
        
        # Iterative prediction
        for i in range(2, max_iterations + 1):
            # Add previous prediction as feature
            prev_pred = results['predictions_by_iteration'][f'iteration_{i-1}']
            current_features['prev_prediction'] = prev_pred
            current_features['pred_diff'] = prev_pred - y if y is not None else 0
            current_features['pred_pct_change'] = prev_pred.pct_change(fill_method=None)
            
            # Make new prediction
            new_pred_results = self.predict(current_features)
            # Extract the first step predictions
            new_pred = new_pred_results['forecasts']['step_1']
            results['predictions_by_iteration'][f'iteration_{i}'] = new_pred
            
            # Update forecasts with latest prediction
            results['forecasts']['step_1'] = new_pred
            
            # Track feature importance if available
            if 'feature_importance' in new_pred_results:
                results['feature_importance'].append({
                    'iteration': i,
                    'importances': new_pred_results['feature_importance']
                })
            
            # Check convergence
            pred_change = np.abs(new_pred - prev_pred).mean()
            results['convergence_info'][f'iteration_{i}'] = pred_change
            
            if pred_change < convergence_threshold:
                print(f"Converged after {i} iterations (change: {pred_change:.6f})")
                break
        
        # Detect anomalies using the final predictions
        final_predictions = results['predictions_by_iteration'].iloc[:, -1]  # Get last iteration
        if y is not None:
            results['detected_anomalies'] = self.detect_anomalies(
                actual_values=results['clean_data'],
                predicted_values=final_predictions
            )
        else:
            # If no ground truth, compare with initial predictions
            results['detected_anomalies'] = self.detect_anomalies(
                actual_values=results['clean_data'],
                predicted_values=final_predictions
            )
        
        return results 