import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from tqdm.auto import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt

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
        
        # Anomaly detection layer - outputs anomaly score (higher = more likely to be anomaly)
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, where 1 means more likely to be anomaly
        )
        
        # Add storage for previous predictions
        self.previous_preds = None
        self.previous_flags = None
  
    def forward(self, x, seq_idx=0, prev_predictions=None, prev_anomaly_flags=None, is_training=False):
        """
        Forward pass with sequence-based iterative prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            seq_idx: Index of the current sequence
            prev_predictions: Optional tensor of previous predictions
            prev_anomaly_flags: Optional tensor of previous anomaly flags
            
        Returns:
            tuple: (predictions, anomaly_flags, anomaly_scores), where predictions has shape (batch_size, prediction_window)
                  and anomaly_flags has shape (batch_size, prediction_window)
        """
        batch_size, seq_len, num_features = x.size()
        
        # If we have previous predictions, incorporate them into input
        if prev_predictions is not None:
            # Create updated input using previous predictions
            x_updated = x.clone()
            # Calculate how much of the current sequence should use previous predictions
            sequence_start_idx = seq_idx * self.output_size
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

        if is_training:
            final_output = self.dropout(final_output)
        
        # Generate forecast for multiple steps
        forecast = self.fc(final_output)  # Shape: (batch_size, prediction_window)
        
        # Calculate anomaly scores for each prediction
        anomaly_scores = self.anomaly_detector(final_output)  # Shape: (batch_size, 1)
        
        # Generate anomaly flags for each prediction
        anomaly_flags = torch.zeros(batch_size, self.output_size, device=x.device)
        for i in range(self.output_size):
            # Get prediction for current timestep and ensure correct shape
            current_pred = forecast[:, i:i+1]  # Shape: (batch_size, 1)
            # Pass through anomaly detector and maintain shape
            anomaly_score = self.anomaly_detector(current_pred)  # Shape: (batch_size, 1)
            anomaly_flags[:, i] = torch.sigmoid(anomaly_score).squeeze(-1)  # Shape: (batch_size)
        
        return forecast, anomaly_flags, anomaly_scores
    
    def train_iteratively(self, x, y, optimizer, criterion, max_iterations=5, convergence_threshold=0.01):
        """
        Train the model iteratively using its own predictions.
        
        Args:
            x: Input tensor
            y: Target tensor
            optimizer: Optimizer instance
            criterion: Loss function
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence
            
        Returns:
            tuple: (final predictions, final anomaly flags, training loss)
        """
        self.train()
        current_predictions = None
        current_flags = None
        best_loss = float('inf')
        best_predictions = None
        best_flags = None
        
        print(f"\nStarting iterative training with max_iterations={max_iterations}")
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Make predictions using current state
            predictions, flags, anomaly_scores = self(x, prev_predictions=current_predictions, 
                                                    prev_anomaly_flags=current_flags, is_training=True)
            
            # Calculate loss with anomaly-based weighting
            loss = criterion(predictions, y)
            weighted_loss = (loss * (1 - anomaly_scores)).mean()
            
            # Backpropagate
            weighted_loss.backward()
            optimizer.step()
            
            # Log iteration details
            if iteration > 0:
                pred_change = torch.abs(current_predictions - predictions).mean().item()
                print(f"Iteration {iteration + 1}:")
                print(f"  - Loss: {weighted_loss.item():.6f}")
                print(f"  - Prediction change: {pred_change:.6f}")
                print(f"  - Average anomaly score: {anomaly_scores.mean().item():.4f}")
            
            # Update current predictions and flags
            current_predictions = predictions.detach()
            current_flags = flags.detach()
            
            # Check for convergence
            if iteration > 0:
                if pred_change < convergence_threshold:
                    print(f"Convergence reached at iteration {iteration + 1}")
                    break
            
            # Store best results
            if weighted_loss.item() < best_loss:
                best_loss = weighted_loss.item()
                best_predictions = current_predictions
                best_flags = current_flags
        
        print(f"Training completed after {iteration + 1} iterations")
        return best_predictions, best_flags, best_loss

    def visualize_iterative_progress(self, x, y, max_iterations=5):
        """
        Visualize how predictions change across iterations.
        
        Args:
            x: Input tensor
            y: Target tensor
            max_iterations: Maximum number of iterations to visualize
        """
        self.eval()
        current_predictions = None
        current_flags = None
        
        # Store predictions for each iteration
        all_predictions = []
        all_anomaly_scores = []
        
        with torch.no_grad():
            for iteration in range(max_iterations):
                predictions, flags, anomaly_scores = self(x, prev_predictions=current_predictions, 
                                                        prev_anomaly_flags=current_flags)
                
                all_predictions.append(predictions.cpu().numpy())
                all_anomaly_scores.append(anomaly_scores.cpu().numpy())
                
                current_predictions = predictions
                current_flags = flags
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_anomaly_scores = np.array(all_anomaly_scores)
        y_np = y.cpu().numpy()
        
        # Plot predictions across iterations
        plt.figure(figsize=(15, 10))
        
        # Plot actual values
        plt.plot(y_np, label='Actual', color='black', linewidth=2)
        
        # Plot predictions for each iteration
        for i in range(max_iterations):
            plt.plot(all_predictions[i], label=f'Iteration {i+1}', alpha=0.7)
        
        plt.title('Predictions Across Iterations')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot anomaly scores across iterations
        plt.figure(figsize=(15, 5))
        for i in range(max_iterations):
            plt.plot(all_anomaly_scores[i], label=f'Iteration {i+1}', alpha=0.7)
        
        plt.title('Anomaly Scores Across Iterations')
        plt.xlabel('Time Steps')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_iterative_metrics(self, x, y, max_iterations=5):
        """
        Analyze and track key metrics across iterations.
        
        Args:
            x: Input tensor
            y: Target tensor
            max_iterations: Maximum number of iterations to analyze
            
        Returns:
            dict: Dictionary containing metrics for each iteration
        """
        self.eval()
        current_predictions = None
        current_flags = None
        
        metrics = {
            'iterations': [],
            'mse': [],
            'mae': [],
            'anomaly_score_mean': [],
            'anomaly_score_std': [],
            'prediction_change': []
        }
        
        with torch.no_grad():
            previous_predictions = None
            for iteration in range(max_iterations):
                predictions, flags, anomaly_scores = self(x, prev_predictions=current_predictions, 
                                                        prev_anomaly_flags=current_flags)
                
                # Calculate metrics
                mse = torch.mean((predictions - y) ** 2).item()
                mae = torch.mean(torch.abs(predictions - y)).item()
                
                if previous_predictions is not None:
                    pred_change = torch.mean(torch.abs(predictions - previous_predictions)).item()
                else:
                    pred_change = 0.0
                
                # Store metrics
                metrics['iterations'].append(iteration + 1)
                metrics['mse'].append(mse)
                metrics['mae'].append(mae)
                metrics['anomaly_score_mean'].append(torch.mean(anomaly_scores).item())
                metrics['anomaly_score_std'].append(torch.std(anomaly_scores).item())
                metrics['prediction_change'].append(pred_change)
                
                current_predictions = predictions
                current_flags = flags
                previous_predictions = predictions
        
        # Print metrics summary
        print("\nIterative Training Metrics Summary:")
        print(f"{'Iteration':<10} {'MSE':<15} {'MAE':<15} {'Anomaly Score':<15} {'Prediction Change':<15}")
        print("-" * 70)
        
        for i in range(len(metrics['iterations'])):
            print(f"{metrics['iterations'][i]:<10} {metrics['mse'][i]:<15.6f} {metrics['mae'][i]:<15.6f} "
                  f"{metrics['anomaly_score_mean'][i]:<15.4f} {metrics['prediction_change'][i]:<15.6f}")
        
        return metrics

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
        self.scaler = None
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
        """Train the model using iterative training approach"""
        print("\nStarting iterative training...")
        
        # Initialize preprocessor and scaler
        self.preprocessor = DataPreprocessor(self.config)
        self.scaler = MinMaxScaler()
        
        # Prepare data
        print("Preparing training data...")
        train_features = self.preprocessor.prepare_features(train_data, project_root, station_id)
        val_features = self.preprocessor.prepare_features(val_data, project_root, station_id)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        val_features_scaled = self.scaler.transform(val_features)
        
        # Create sequences
        sequence_length = self.config['sequence_length']
        prediction_window = self.config['prediction_window']
        sequence_stride = self.config['sequence_stride']
        
        # Create sequences with overlap
        train_sequences = create_overlap_sequences(
            train_features_scaled, 
            sequence_length, 
            prediction_window,
            sequence_stride
        )
        val_sequences = create_overlap_sequences(
            val_features_scaled, 
            sequence_length, 
            prediction_window,
            sequence_stride
        )
        
        # Initialize model
        input_size = train_sequences[0][0].shape[1]
        self.model = ForecastingLSTM(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=prediction_window,
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.SmoothL1Loss(reduction='none')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config['patience']
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            num_batches = 0
            
            # Process training data in batches
            for batch_x, batch_y in train_sequences:
                batch_x = torch.FloatTensor(batch_x).to(self.device)
                batch_y = torch.FloatTensor(batch_y).to(self.device)
                
                # Train iteratively on this batch
                predictions, flags, loss = self.model.train_iteratively(
                    batch_x, batch_y, optimizer, criterion,
                    max_iterations=self.config['max_iterations'],
                    convergence_threshold=self.config['convergence_threshold']
                )
                
                train_loss += loss
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            
            # Validation
            self.model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_sequences:
                    batch_x = torch.FloatTensor(batch_x).to(self.device)
                    batch_y = torch.FloatTensor(batch_y).to(self.device)
                    
                    # Make predictions with current model state
                    val_pred, val_flags, val_anomaly_scores = self.model(batch_x)
                    
                    # Calculate validation loss with NaN handling and anomaly weighting
                    loss = criterion(val_pred, batch_y)
                    mask = ~torch.isnan(batch_y)
                    # Weight loss by (1 - anomaly_scores) to give less weight to anomalous predictions
                    masked_loss = (loss * mask.float() * (1 - val_anomaly_scores)).mean()
                    
                    val_loss += masked_loss.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Training Loss: {avg_train_loss:.6f}")
            print(f"  Validation Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return self.model
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.eval()
        
        # Prepare features
        features = self.preprocessor.prepare_features(data, self.config['project_root'], self.config['station_id'])
        features_scaled = self.scaler.transform(features)
        
        # Create sequences
        sequence_length = self.config['sequence_length']
        prediction_window = self.config['prediction_window']
        sequences = create_overlap_sequences(
            features_scaled, 
            sequence_length, 
            prediction_window,
            self.config['sequence_stride']
        )
        
        # Initialize storage for predictions and flags
        all_predictions = []
        all_flags = []
        all_anomaly_scores = []
        
        # Initialize previous predictions and flags
        prev_pred = None
        prev_flags = None
        
        with torch.no_grad():
            for i, (sequence, _) in enumerate(sequences):
                sequence_tensor = torch.FloatTensor(sequence).to(self.device)
                
                # Make prediction
                pred, flags, anomaly_scores = self.model(
                    sequence_tensor, 
                    i // prediction_window, 
                    prev_pred, 
                    prev_flags
                )
                
                # Store results
                all_predictions.append(pred.cpu().numpy())
                all_flags.append(flags.cpu().numpy())
                all_anomaly_scores.append(anomaly_scores.cpu().numpy())
                
                # Update previous predictions and flags
                prev_pred = pred
                prev_flags = flags
        
        # Combine predictions
        predictions = np.concatenate(all_predictions)
        flags = np.concatenate(all_flags)
        anomaly_scores = np.concatenate(all_anomaly_scores)
        
        # Create results dictionary
        results = {
            'forecasts': predictions,
            'anomaly_flags': flags,
            'anomaly_scores': anomaly_scores,
            'clean_data': data[self.preprocessor.output_features].values
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
