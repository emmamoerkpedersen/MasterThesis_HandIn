import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.objective_functions import get_objective_function
from experiments.iterative_forecaster.iterative_forecast_model import ForecastingLSTM
from tqdm import tqdm

class IterativeForecastTrainer:
    def __init__(self, config, preprocessor):
        """
        Initialize the trainer and Iterative LSTM model.
        
        Args:
            config: Dictionary containing model and training parameters
            preprocessor: Instance of DataPreprocessor
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = preprocessor

        # Initialize LSTM Model using parameters from config
        self.model = ForecastingLSTM(
            input_size=len(preprocessor.feature_cols),
            hidden_size=config['hidden_size'],
            output_size=config['prediction_window'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate'))
        self.criterion = get_objective_function(config.get('objective_function'))

    def fit_scaler(self, train_data):
        """
        Fit the feature scaler on the entire training dataset.
        
        Args:
            train_data: Training data (either DataFrame or preprocessed tuple)
        """
        print("\nFitting feature scaler on training data...")
        
        # Check if data is already preprocessed (tuple of tensors)
        if isinstance(train_data, tuple):
            print("Data is already preprocessed, skipping scaler fitting.")
            return
            
        # Extract features and target from DataFrame
        features = train_data[self.preprocessor.feature_cols]
        target = pd.DataFrame(train_data[self.preprocessor.output_features])
        
        # Fit the scaler on all training data
        self.preprocessor.feature_scaler.fit(features, target)
        print("Feature scaler fitted successfully.")

    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation with iterative forecasting.
        Includes warmup period where loss is not calculated for initial timesteps.
        """
        self.model.train() if training else self.model.eval()
        total_loss = 0
        num_batches_with_loss = 0
        
        # Lists to store validation predictions and targets
        all_predictions = []
        all_targets = []

        # Get warmup length from config
        warmup_length = self.config.get('warmup_length', 0)

        with torch.set_grad_enabled(training):
            for batch_idx, (batch_X, batch_y) in enumerate(tqdm(data_loader, desc="Training" if training else "Validating", leave=False)):
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if training:
                    self.optimizer.zero_grad()
                
                # Initialize previous predictions as None for first sequence
                prev_predictions = None
                
                # Get predictions for current sequence
                outputs = self.model(batch_X, seq_idx=batch_idx, prev_predictions=prev_predictions, is_training=training)
                
                # Store predictions for next iteration if not in training mode
                if not training:
                    prev_predictions = outputs.detach()
                    # Always store predictions and targets for validation
                    all_predictions.append(outputs.cpu().detach())
                    all_targets.append(batch_y.cpu())
                
                # Create mask for valid predictions (non-NaN values)
                non_nan_mask = ~torch.isnan(batch_y)
                
                # Add warmup mask - don't calculate loss for warmup period
                # Only apply warmup to the first batch
                if warmup_length > 0 and batch_idx == 0:
                    warmup_mask = torch.ones_like(non_nan_mask, dtype=torch.bool)
                    warmup_mask[:, :warmup_length] = False
                    valid_mask = non_nan_mask & warmup_mask
                else:
                    valid_mask = non_nan_mask
                
                # Only proceed with loss calculation if we have valid points
                if valid_mask.any():
                    valid_outputs = outputs[valid_mask]
                    valid_target = batch_y[valid_mask]
                    
                    if valid_target.size(0) > 0:
                        loss = self.criterion(valid_outputs, valid_target)
                        
                        if training:
                            loss.backward()
                            self.optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches_with_loss += 1

        if training:
            # Return average loss only for batches where loss was calculated
            return total_loss / max(1, num_batches_with_loss)
        else:
            # For validation, always return predictions and targets
            if len(all_predictions) == 0:
                # Return empty tensors with correct shape if no predictions
                return 0.0, torch.empty(0, device=self.device), torch.empty(0, device=self.device)
            
            val_predictions = torch.cat(all_predictions, dim=0)
            val_targets = torch.cat(all_targets, dim=0)
            avg_loss = total_loss / max(1, num_batches_with_loss)
            return avg_loss, val_predictions, val_targets

    def train(self, train_data, val_data, epochs, batch_size, patience):
        """
        Train the iterative LSTM model.
        """
        # Check if data needs preprocessing
        if not isinstance(train_data, tuple):
            # First, fit the scaler on all training data
            self.fit_scaler(train_data)
            
            # Now prepare the sequences for training and validation
            print("\nPreparing training sequences...")
            train_sequences = self.preprocessor.prepare_iterative_data(
                train_data,
                sequence_length=self.config['sequence_length'],
                prediction_window=self.config['prediction_window'],
                is_training=False  # Use transform instead of fit_transform since we already fitted
            )
            
            print("Preparing validation sequences...")
            val_sequences = self.preprocessor.prepare_iterative_data(
                val_data,
                sequence_length=self.config['sequence_length'],
                prediction_window=self.config['prediction_window'],
                is_training=False
            )
        else:
            # Data is already preprocessed
            train_sequences = train_data
            val_sequences = val_data
        
        # Print data preparation info
        print(f"\nTraining sequences shape: {train_sequences[0].shape}")
        print(f"Validation sequences shape: {val_sequences[0].shape}")
        
        # Create data loaders with num_workers for parallel data loading
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_sequences[0], train_sequences[1]), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_sequences[0], val_sequences[1]), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}

        # Training loop with progress bar
        current_lr = self.optimizer.param_groups[0]['lr']
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss, val_predictions, val_targets = self._run_epoch(val_loader, training=False)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return history, val_predictions, val_targets

    def predict(self, data):
        """
        Make predictions using the iterative forecasting approach.
        """
        self.model.eval()
        X, y = data
        
        with torch.no_grad():
            predictions = []
            prev_predictions = None
            
            # Iterate through sequences
            for seq_idx in range(len(X)):
                # Get current sequence
                current_X = X[seq_idx:seq_idx+1]  # Keep batch dimension
                
                # Make prediction
                pred = self.model(current_X, seq_idx=seq_idx, prev_predictions=prev_predictions, is_training=False)
                
                # Store prediction
                predictions.append(pred)
                
                # Update previous predictions for next iteration
                prev_predictions = pred.detach()
            
            # Concatenate all predictions
            predictions = torch.cat(predictions, dim=0).cpu().numpy()
            y = y.cpu().numpy()
            
            # Inverse transform predictions if needed
            predictions_original = self.preprocessor.feature_scaler.inverse_transform_target(predictions)
            
            return predictions_original, predictions, y

    def train_epoch(self, train_loader, optimizer, criterion):
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            
            batch_size = data.size(0)
            sequence_predictions = []
            prev_predictions = None
            
            # Make predictions for each sequence
            for seq_idx in range(self.num_sequences):
                # Get predictions for current sequence
                predictions = self.model(data, seq_idx=seq_idx, prev_predictions=prev_predictions, is_training=True)
                sequence_predictions.append(predictions)
                
                # Update previous predictions for next iteration
                prev_predictions = predictions.detach()
            
            # Stack all predictions
            all_predictions = torch.stack(sequence_predictions, dim=1)
            
            # Calculate loss only on non-NaN targets
            loss = 0
            valid_sequences = 0
            
            for seq_idx in range(self.num_sequences):
                seq_targets = targets[:, seq_idx, :]
                seq_predictions = all_predictions[:, seq_idx, :]
                
                # Only calculate loss for non-NaN targets
                valid_mask = ~torch.isnan(seq_targets)
                if valid_mask.any():
                    seq_loss = criterion(
                        seq_predictions[valid_mask],
                        seq_targets[valid_mask]
                    )
                    loss += seq_loss
                    valid_sequences += 1
            
            if valid_sequences > 0:
                loss = loss / valid_sequences
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Average Loss = {total_loss / (batch_idx + 1):.4f}")
        
        return total_loss / len(train_loader) 