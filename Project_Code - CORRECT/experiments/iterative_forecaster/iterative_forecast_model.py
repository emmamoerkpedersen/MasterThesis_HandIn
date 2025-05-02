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
    LSTM model for water level forecasting with configurable components:
    - Multi-head mechanisms
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(ForecastingLSTM, self).__init__()
        self.model_name = 'ForecastingLSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  # This is the prediction window size
        self.num_layers = num_layers
    
        
        # Main LSTM for processing raw input
        self.main_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to map to output
        # Output size is prediction_window since we want to predict multiple steps
        self.fc = nn.Linear(hidden_size, output_size)
        
        
        # Add a simple anomaly detection layer
        # Input size is 1 since we're detecting anomalies for each prediction individually
        self.anomaly_detector = nn.Linear(1, 1)  # Binary classification: anomaly or not
        
        # Add storage for previous predictions
        self.previous_preds = None
        self.previous_flags = None
  
    def forward(self, x, seq_idx=0, prev_predictions=None, prev_anomaly_flags=None):
        """
        Forward pass with sequence-based iterative prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            seq_idx: Index of the current sequence
            prev_predictions: Optional tensor of previous predictions
            prev_anomaly_flags: Optional tensor of previous anomaly flags
            
        Returns:
            tuple: (predictions, anomaly_flags), where predictions has shape (batch_size, prediction_window)
                  and anomaly_flags has shape (batch_size, prediction_window)
        """
        batch_size, seq_len, num_features = x.size()
        
        # If we have previous predictions, incorporate them into input
        if prev_predictions is not None:
            # Create updated input using previous predictions
            x_updated = x.clone()
            
            # Calculate how much of the current sequence should use previous predictions
            sequence_start_idx = seq_idx * self.output_size  # Using output_size as prediction window
            overlap_start = max(0, sequence_start_idx - seq_len)
            overlap_length = min(seq_len - overlap_start, prev_predictions.size(1))
            
            if overlap_length > 0:
                # Replace only the overlapping portion with previous predictions
                x_updated[:, overlap_start:overlap_start+overlap_length, 0] = prev_predictions[:, -overlap_length:]
                
                # Add anomaly flags as features, but only for the overlapping portion
                if prev_anomaly_flags is not None:
                    # Create a tensor of zeros for the full sequence length
                    flag_feature = torch.zeros(batch_size, seq_len, 1, device=x.device)
                    # Fill in the overlapping portion with actual flags
                    flag_feature[:, overlap_start:overlap_start+overlap_length] = prev_anomaly_flags[:, -overlap_length:].unsqueeze(-1)
                    # Concatenate with input
                    x_updated = torch.cat([x_updated, flag_feature], dim=2)
            
            # Use the updated input
            x = x_updated
        
        # Process through LSTM
        main_out, _ = self.main_lstm(x)
        
        
        final_output = main_out[:, -1, :]
        
        if self.training:
            final_output = self.dropout(final_output)
        
        # Generate forecast for multiple steps
        forecast = self.fc(final_output)  # Shape: (batch_size, prediction_window)
        
        # Generate anomaly flags for each prediction
        anomaly_flags = torch.zeros(batch_size, self.output_size, device=x.device)
        for i in range(self.output_size):
            # Get prediction for current timestep and ensure correct shape
            current_pred = forecast[:, i:i+1]  # Shape: (batch_size, 1)
            # Pass through anomaly detector and maintain shape
            anomaly_score = self.anomaly_detector(current_pred)  # Shape: (batch_size, 1)
            anomaly_flags[:, i] = torch.sigmoid(anomaly_score).squeeze(-1)  # Shape: (batch_size)
        
        return forecast, anomaly_flags
    
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
        
        # Add a memory of recent predictions for stability
        self.recent_predictions = []
        self.recent_window = config.get('recent_predictions_window', 20)

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
 
        
        # Initialize and return the model
        model = ForecastingLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        
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
        criterion = nn.SmoothL1Loss(reduction='none')  # Use 'none' to handle NaN masking
        
        # Initialize best validation loss and early stopping variables
        best_val_loss = float('inf')
        best_model_weights = None
        epochs_no_improve = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            valid_samples = 0  # Track number of valid samples
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                batch_x = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Process sequence with previous predictions
                pred, anomaly_flags = self.model(batch_x)
                
                # Ensure shapes match for loss calculation
                if pred.shape != batch_y.shape:
                    pred = pred.view(batch_y.shape)
                
                # Calculate loss for all predictions
                loss = criterion(pred, batch_y)
                
                # Create mask for non-NaN values
                mask = ~torch.isnan(batch_y)
                
                # Apply mask to loss
                masked_loss = loss * mask.float()
                
                # Calculate mean loss only over valid samples
                valid_samples_in_batch = mask.sum().item()
                if valid_samples_in_batch > 0:
                    batch_loss = masked_loss.sum() / valid_samples_in_batch
                    batch_loss.backward()
                    optimizer.step()
                    
                    train_loss += batch_loss.item() * valid_samples_in_batch
                    valid_samples += valid_samples_in_batch
            
            # Calculate average loss over valid samples
            avg_train_loss = train_loss / valid_samples if valid_samples > 0 else float('inf')
            
            # Validation
            self.model.eval()
            val_loss = 0
            valid_val_samples = 0
            
            with torch.no_grad():
                val_preds, val_flags = self.model(X_val)
                
                # Ensure shapes match for validation loss calculation
                if val_preds.shape != y_val.shape:
                    val_preds = val_preds.view(y_val.shape)
                
                # Calculate validation loss with NaN handling
                val_loss = criterion(val_preds, y_val)
                val_mask = ~torch.isnan(y_val)
                masked_val_loss = val_loss * val_mask.float()
                valid_val_samples = val_mask.sum().item()
                
                if valid_val_samples > 0:
                    val_loss = masked_val_loss.sum() / valid_val_samples
                else:
                    val_loss = float('inf')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_weights = self.model.state_dict().copy()
                # Add a checkmark to indicate improvement
                print(f"Epoch {epoch+1}/{num_epochs}, train_loss: {avg_train_loss:.6f}, val_loss: {val_loss:.6f} ✓")
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
    
    def predict(self, test_data):
        """
        Make predictions using the sequence-based approach.
        Each sequence predicts the next prediction_window timesteps without overlap.
        
        Example for sequence_length=10, prediction_window=5:
        Sequence 1: Input[0:10] -> Predict[10:15]  (t1-t10 -> t11-t15)
        Sequence 2: Input[5:15] -> Predict[15:20]  (t6-t15 -> t16-t20)
        And so on...
        
        Args:
            test_data: DataFrame with test data
            
        Returns:
            dict: Dictionary containing predictions and anomaly information
        """
        self.model.eval()
        
        # Prepare data
        X_test_df = test_data[self.preprocessor.feature_cols]
        y_test_df = pd.DataFrame(test_data[self.preprocessor.output_features])
        
        # Scale data
        X_test_scaled, y_test_scaled = self.preprocessor.feature_scaler.transform(X_test_df, y_test_df)
        
        # Get sequence parameters
        sequence_length = self.config.get('sequence_length', 100)
        prediction_window = self.config.get('prediction_window', 5)
        
        # Initialize arrays to store predictions and flags
        # We'll have predictions for all timesteps after sequence_length
        predictions = np.zeros((len(test_data) - sequence_length, 1))
        anomaly_flags = np.zeros((len(test_data) - sequence_length, 1))
        
        # Process data in sequences
        with torch.no_grad():
            prev_pred = None
            prev_flags = None
            
            # Iterate through sequences (using prediction_window as stride to avoid overlapping predictions)
            for i in range(0, len(X_test_scaled) - sequence_length, prediction_window):
                # Extract sequence
                sequence = X_test_scaled[i:i+sequence_length]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Make prediction
                pred, flags = self.model(sequence_tensor, i // prediction_window, prev_pred, prev_flags)
                
                # Convert predictions to numpy
                pred_np = pred.cpu().numpy()  # Shape: (batch_size, prediction_window)
                flags_np = flags.cpu().numpy()  # Shape: (batch_size, prediction_window)
                
                # Store predictions and flags at the correct position
                pred_start_idx = i
                pred_end_idx = min(pred_start_idx + prediction_window, len(predictions))
                
                # Store predictions and flags
                predictions[pred_start_idx:pred_end_idx] = pred_np[0, :pred_end_idx-pred_start_idx].reshape(-1, 1)
                anomaly_flags[pred_start_idx:pred_end_idx] = flags_np[0, :pred_end_idx-pred_start_idx].reshape(-1, 1)
                
                # Update previous predictions and flags
                prev_pred = pred
                prev_flags = flags
        
        # Convert predictions back to original scale
        final_predictions = self.preprocessor.feature_scaler.inverse_transform_target(predictions)
        
        # Create DataFrame for predictions and actual values with proper column names
        predictions_df = pd.DataFrame(final_predictions, 
                                    index=test_data.index[sequence_length:],
                                    columns=[f'step_{i+1}' for i in range(1)])  # Using step_1 for now since we're storing one step at a time
        actual_values = test_data[self.preprocessor.output_features].iloc[sequence_length:]
        
        # Detect anomalies by comparing predictions with actual values
        detected_anomalies = self.detect_anomalies(
            actual_values=actual_values,
            predicted_values=predictions_df['step_1']  # Use step_1 column for anomaly detection
        )
        
        # Create results dictionary with adjusted index
        # The index starts from sequence_length onwards since we can't predict the first sequence_length points
        results = {
            'forecasts': predictions_df,
            'anomaly_flags': pd.DataFrame(anomaly_flags, index=test_data.index[sequence_length:], columns=['step_1']),
            'clean_data': actual_values,
            'detected_anomalies': detected_anomalies
        }
        
        return results
    
    def _inject_errors(self, data, error_periods=None):
        """
        DEPRECATED: Inject synthetic errors into the raw water level data.
        
        This method is kept for backward compatibility but does NOT properly update
        derived features (like lag features, moving averages, etc.) after error injection.
        
        For proper error injection that updates all derived features, use the 
        reconstruct_features_with_errors() function in run_forecast.py instead.
        
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
            
            # Apply error to the target output feature only
            if error_type == 'noise':
                # Generate random noise with the correct shape
                noise = np.random.normal(0, magnitude, size=sum(mask))
                # Apply noise directly
                modified_data.loc[mask, output_feature] += noise
            
            elif error_type == 'offset':
                # Apply offset
                modified_data.loc[mask, output_feature] += magnitude
            
            elif error_type == 'scaling':
                # Apply scaling
                modified_data.loc[mask, output_feature] *= magnitude
            
            elif error_type == 'missing':
                # Set values to NaN (simulating missing data)
                modified_data.loc[mask, output_feature] = np.nan
        
        return modified_data
    
    def detect_anomalies(self, actual_values, predicted_values):
        """
        Detect anomalies by comparing actual values with predicted values.
        
        Args:
            actual_values: Actual water level values
            predicted_values: Predicted water level values
            
        Returns:
            DataFrame with anomaly flags and scores, including confidence levels
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
        
        # Identify anomalies based on threshold
        anomaly_flags = combined_z > self.anomaly_threshold
        
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
