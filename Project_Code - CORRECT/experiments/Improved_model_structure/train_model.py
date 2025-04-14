import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
#from experiments.Improved_model_structure.model import LSTMModel
from _3_lstm_model.model import LSTMModel
from tqdm import tqdm
from _3_lstm_model.objective_functions import get_objective_function
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

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
            min_lr=1e-6
        )
        
        # Choose loss function based on configuration
        self.criterion = get_objective_function(config.get('objective_function'))
            
        # Print whether gradient clipping is enabled
        print(f"Using gradient clipping with max norm: {self.grad_clip_value}")
        print(f"Using learning rate scheduler with patience {self.scheduler.patience}, factor {self.scheduler.factor}")
    

    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation 
        """
        self.model.train() if training else self.model.eval()
        warmup_length = self.config.get('warmup_length', 100)
        total_loss = 0
        
        # Lists to store validation predictions and targets
        all_predictions = []
        all_targets = []

        with torch.set_grad_enabled(training):
            # Use tqdm only for training and disable it for validation to avoid nested progress bars
            iterable = tqdm(data_loader, desc="Batches", leave=False, disable=not training) if training else data_loader
            for batch_idx, (batch_X, batch_y) in enumerate(iterable):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                # Create warm-up mask
                warmup_mask = torch.ones_like(batch_y, dtype=torch.bool)
                warmup_mask[:, :warmup_length, :] = False

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
                    
                    # Update progress bar with batch loss
                    if batch_idx % 10 == 0:  # Update less frequently to reduce overhead
                        iterable.set_postfix({'batch_loss': f'{loss.item():.6f}'})
                
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

    def evaluate_predictions(self, predictions, targets, data_index=None):
        """
        Calculate performance metrics for model predictions.
        
        Args:
            predictions: Model predictions (numpy array)
            targets: Target values (numpy array)
            data_index: Optional pandas DatetimeIndex for calculating peak metrics
            
        Returns:
            dict: Dictionary of performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Remove NaN values for metric calculation
        valid_mask = (~np.isnan(targets)) & (~np.isnan(predictions))
        valid_targets = targets[valid_mask]
        valid_predictions = predictions[valid_mask]
        
        if len(valid_targets) == 0:
            return {
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan'),
                'mean_error': float('nan'),
                'std_error': float('nan'),
                'peak_mae': float('nan'),
                'peak_rmse': float('nan')
            }
        
        # Calculate standard metrics
        rmse = np.sqrt(mean_squared_error(valid_targets, valid_predictions))
        mae = mean_absolute_error(valid_targets, valid_predictions)
        
        # Calculate and validate R² score (can be extremely negative for poor models)
        r2 = r2_score(valid_targets, valid_predictions)
        r2_original = r2  # Store original value for reporting
        
        # Provide more informative message about the R² value quality
        if r2 < -1.0:
            # Different warning messages based on how negative the R² is
            if r2 < -10.0:
                print(f"Warning: R² value is extremely negative ({r2:.4f}). Model predictions are very poor and may need significant improvement.")
            elif r2 < -5.0:
                print(f"Warning: R² value is highly negative ({r2:.4f}). The model may be struggling with this dataset.")
            else:
                print(f"Warning: R² value is negative ({r2:.4f}). This indicates the model performs worse than a simple mean baseline.")
                
            # Calculate mean of target to understand baseline
            target_mean = np.mean(valid_targets)
            print(f"Target mean: {target_mean:.4f}, Target std: {np.std(valid_targets):.4f}")
            
            # Optional: explain what R² means
            print("Note: R² < 0 means the model performs worse than predicting the mean value for all points.")
            print("      R² = 0 means the model performs as well as predicting the mean.")
            print("      R² = 1 means perfect predictions.")
            
            # Constrain the value to -1.0 for reporting
            r2 = -1.0
        
        # Calculate error statistics
        errors = valid_predictions - valid_targets
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Calculate peak-specific metrics (if we have indices)
        peak_mae = float('nan')
        peak_rmse = float('nan')
        
        if data_index is not None and len(valid_targets) > 0:
            try:
                # Identify peaks (top 10% of values)
                peak_threshold = np.percentile(valid_targets, 90)
                peak_mask = valid_targets >= peak_threshold
                
                if np.sum(peak_mask) > 0:
                    peak_mae = mean_absolute_error(
                        valid_targets[peak_mask], 
                        valid_predictions[peak_mask]
                    )
                    peak_rmse = np.sqrt(mean_squared_error(
                        valid_targets[peak_mask], 
                        valid_predictions[peak_mask]
                    ))
            except Exception as e:
                print(f"Error calculating peak metrics: {e}")
        
        # Add the original R² value for debugging
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'r2_original': r2_original,  # Store the original unconstrained value
            'mean_error': mean_error,
            'std_error': std_error,
            'peak_mae': peak_mae,
            'peak_rmse': peak_rmse
        }
        
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
            num_workers=0 if self.device.type == 'cuda' else 0  # Set to 0 since data is already on GPU
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0 if self.device.type == 'cuda' else 0  # Set to 0 since data is already on GPU
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

        # Training loop with progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            # Run training epoch
            train_loss = self._run_epoch(train_loader, training=True)
            
            # Run validation epoch
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
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Step the learning rate scheduler based on smoothed validation loss
            self.scheduler.step(smoothed_val_loss)

            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.6f}',
                'Val Loss': f'{val_loss:.6f}',
                'LR': f'{current_lr:.6f}',
                'Smooth Val': f'{smoothed_val_loss:.6f}',
                'Patience': f'{patience_counter}/{patience}'
            })
            
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
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'LR': f'{current_lr:.6f}',
                    'Smooth Val': f'{smoothed_val_loss:.6f}',
                    'Patience': f'{patience_counter}/{patience}',
                    'Best Model': '✓'
                })
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_pbar.set_description(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # After training is complete, print a summary
        print(f"\nTraining completed: {epoch+1}/{epochs} epochs")
        print(f"Best smoothed validation loss: {best_val_loss:.6f}")

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        # Calculate additional metrics for validation predictions
        # Convert predictions to numpy and reshape for metrics calculation
        val_predictions_np = val_predictions.cpu().numpy()
        val_targets_np = val_targets.cpu().numpy()
        
        # Reshape to match expected format
        predictions_reshaped = val_predictions_np.reshape(-1, 1)
        targets_reshaped = val_targets_np.reshape(-1, 1)
        
        # Convert back to original scale for meaningful metrics
        predictions_original = self.preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped).flatten()
        targets_original = val_targets_np.flatten()  # Targets are already in the right scale for comparison
        
        # Calculate additional performance metrics
        performance_metrics = self.evaluate_predictions(
            predictions_original, 
            targets_original,
            data_index=val_data.index if hasattr(val_data, 'index') else None
        )
        
        # Add metrics to history
        self.history['metrics'] = performance_metrics
        
        # Print metrics summary
        print("\nValidation Performance Metrics:")
        print(f"RMSE: {performance_metrics['rmse']:.4f}")
        print(f"MAE: {performance_metrics['mae']:.4f}")
        print(f"R²: {performance_metrics['r2']:.4f}")
        if not np.isnan(performance_metrics['peak_rmse']):
            print(f"Peak RMSE: {performance_metrics['peak_rmse']:.4f}")
            print(f"Peak MAE: {performance_metrics['peak_mae']:.4f}")

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
            print("Using time features for prediction")
            
        X, y = self.preprocessor.prepare_data(data_copy, is_training=False)
        
        with torch.no_grad():
            # Make predictions in smaller chunks if needed
            predictions = self.model(X).cpu().numpy()
            y = y.cpu().numpy()
            
            # Preserve temporal order during inverse transform
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_original = self.preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped)
            
            # Reshape back maintaining temporal order
            predictions_original = predictions_original.reshape(predictions.shape)
            
            # Print information about the model and prediction process
            print("\nPrediction Information:")
            print(f"Model: LSTM with {self.config['num_layers']} layers, {self.config['hidden_size']} hidden units")
            print(f"Sequence length: {self.config['sequence_length']}")
            print(f"Features used: {len(self.preprocessor.feature_cols)}")
            
            # Print loss function information
            if self.config.get('use_dynamic_weighting', False):
                print(f"Loss function: Dynamic weighted loss")
            elif self.config.get('use_peak_weighted_loss', False):
                print(f"Loss function: Peak weighted loss (weight: {self.peak_weight})")
            else:
                print(f"Loss function: Standard MSE loss")
                
            print(f"Time features: {self.config.get('use_time_features', False)}")
            print(f"Smoothing: {self.config.get('use_smoothing', False)}")
            
            return predictions_original, predictions, y
    