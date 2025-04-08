import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from experiments.Improved_model_structure.model import LSTMModel
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
            for batch_idx, (batch_X, batch_y) in enumerate(tqdm(data_loader, desc="Training" if training else "Validating", leave=False)):
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

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}, Smoothed Val Loss: {smoothed_val_loss:.6f}")
            
            # Call epoch callback if provided (for hyperparameter tuning)
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch, train_loss, val_loss)
                except Exception as e:
                    print(f"Callback raised an exception: {e}")
                    break

            # Early stopping based on smoothed validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered! No improvement in validation loss for {patience} epochs.")
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
            print("Using time features for prediction")
            
        X, y = self.preprocessor.prepare_data(data_copy, is_training=False)
        
        with torch.no_grad():
            # Make predictions in smaller chunks if needed
            predictions = self.model(X).cpu().numpy()
            y = y.cpu().numpy()
            
            # Preserve temporal order during inverse transform
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_original = self.preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped)
            
            # Apply smoothing if configured
            if self.config.get('use_smoothing', False):
                alpha = self.config.get('smoothing_alpha', 0.2)  # EMA alpha parameter
                print(f"Applying exponential smoothing with alpha={alpha}")
                predictions_original = self._apply_exponential_smoothing(predictions_original, alpha)
            
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