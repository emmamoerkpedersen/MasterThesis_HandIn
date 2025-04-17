import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from experiments.Improved_model_structure.model import LSTMModel
#from _3_lstm_model.model import LSTMModel
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
        
        # Load the objective function
        from _3_lstm_model.objective_functions import get_objective_function
        obj_func_name = config.get('objective_function', 'smoothL1_loss')
        self.objective_function = get_objective_function(obj_func_name)
        print(f"Using loss function: {obj_func_name}")

        # Initialize LSTM Model using parameters from config
        self.model = LSTMModel(
            input_size=len(preprocessor.feature_cols),
            sequence_length=config.get('sequence_length', None),
            hidden_size=config['hidden_size'],
            output_size=len(config['output_features']),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        
        # Initialize history
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        
        # Get peak weight for the custom loss function (default to 2.0 if not specified)
        self.peak_weight = config.get('peak_weight', 2.0)
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        print(f"Using learning rate scheduler with patience {self.scheduler.patience}, factor {self.scheduler.factor}")
    

    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation.
        
        Args:
            data_loader: DataLoader for batch iteration
            training: Whether this is a training or validation epoch
            
        Returns:
            float: Mean loss for this epoch
            Optional: For validation, also returns predictions and targets
        """
        self.model.train() if training else self.model.eval()
        total_loss = 0
        
        # Lists to store validation predictions and targets
        all_predictions = []
        all_targets = []

        with torch.set_grad_enabled(training):
            for batch_X, batch_y in tqdm(data_loader, desc="Training" if training else "Validating", leave=False):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Create warm-up mask
                warmup_length = self.config.get('warmup_length', 0)
                if warmup_length > 0:
                    warmup_mask = torch.ones_like(batch_y, dtype=torch.bool)
                    warmup_mask[:, :warmup_length, :] = False
                else:
                    warmup_mask = torch.ones_like(batch_y, dtype=torch.bool)

                # Combine warm-up mask with NaN mask
                non_nan_mask = ~torch.isnan(batch_y)
                valid_mask = non_nan_mask & warmup_mask

                valid_outputs = outputs[valid_mask]
                valid_targets = batch_y[valid_mask]
                
                if valid_targets.size(0) == 0:
                    continue

                # Use the configured objective function
                loss = self.objective_function(valid_outputs, valid_targets)
                
                if training:
                    loss.backward()
                    
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
        patience_counter = 0
        best_model_state = None
        
        # Reset history for new training run
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        
        # Training loop with progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            # Run training epoch
            train_loss = self._run_epoch(train_loader, training=True)
            
            # Run validation epoch
            val_loss, val_predictions, val_targets = self._run_epoch(val_loader, training=False)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            # Remove smoothed validation loss calculation and storage
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Step the learning rate scheduler based on raw validation loss
            self.scheduler.step(val_loss)

            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.6f}',
                'Val Loss': f'{val_loss:.6f}',
                'LR': f'{current_lr:.6f}',
                'Patience': f'{patience_counter}/{patience}'
            })
            
            # Call epoch callback if provided (for hyperparameter tuning)
            if epoch_callback is not None:
                try:
                    epoch_callback(epoch, train_loss, val_loss)
                except Exception as e:
                    print(f"Callback raised an exception: {e}")
                    break

            # Early stopping based on raw validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.6f}',
                    'Val Loss': f'{val_loss:.6f}',
                    'LR': f'{current_lr:.6f}',
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
        print(f"Best validation loss: {best_val_loss:.6f}")

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
            
            # Reshape back maintaining temporal order
            predictions_original = predictions_original.reshape(predictions.shape)
            
            # Print information about the model and prediction process
            print("\nPrediction Information:")
            print(f"Model: LSTM with {self.config['num_layers']} layers, {self.config['hidden_size']} hidden units")
            print(f"Sequence length: {self.config['sequence_length']}")
            print(f"Features used: {len(self.preprocessor.feature_cols)}")
            print(f"Loss function: {self.config['objective_function']}")
                
            print(f"Time features: {self.config.get('use_time_features', False)}")
            print(f"Smoothing: {self.config.get('use_smoothing', False)}")
            
            return predictions_original, predictions, y
    