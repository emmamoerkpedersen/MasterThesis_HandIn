import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from tqdm.auto import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # Add this import
import torch.optim as optim

# Add the necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import preprocessing module
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

class ForecastingLSTM(nn.Module):
    """
    LSTM model for water level forecasting with configurable components.
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
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Add storage for previous predictions
        self.previous_preds = None
  
    def forward(self, x, seq_idx=0, prev_predictions=None, is_training=False):
        """
        Forward pass with sequence-based iterative prediction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            seq_idx: Index of the current sequence
            prev_predictions: Optional tensor of previous predictions
            
        Returns:
            predictions: Tensor of shape (batch_size, prediction_window)
        """
        batch_size, seq_len, num_features = x.size()
        
        # If we have previous predictions, incorporate them into input
        if prev_predictions is not None:
            # Create updated input using previous predictions
            x_updated = x.clone()
            
            # Calculate how many original values to keep
            original_values_to_keep = max(0, seq_len - (seq_idx * self.output_size))
            
            # Calculate how many predictions to use
            predictions_to_use = min(seq_len, prev_predictions.size(1))
            
            # Replace the appropriate portion of the input with predictions
            if predictions_to_use > 0:
                # Keep original values at the start, use predictions for the rest
                x_updated[:, original_values_to_keep:original_values_to_keep + predictions_to_use, 0] = prev_predictions[:, -predictions_to_use:]
            
            # Use the updated input
            x = x_updated
        
        # Process through LSTM
        main_out, _ = self.main_lstm(x)
        final_output = main_out[:, -1, :]

        if is_training:
            final_output = self.dropout(final_output)
        
        # Generate forecast for multiple steps
        forecast = self.fc(final_output)  # Shape: (batch_size, prediction_window)
        
        return forecast
    
    def train_iteratively(self, x, y, optimizer, criterion, convergence_threshold=0.01):
        """
        Train the model by processing all sequences and using predictions from previous sequences.
        
        Args:
            x: Input tensor of shape (sequence_length, num_features)
            y: Target tensor of shape (prediction_window, 1)
            optimizer: Optimizer instance
            criterion: Loss function
            convergence_threshold: Threshold for prediction convergence
            
        Returns:
            tuple: (final predictions, training loss)
        """
        self.train()
        best_loss = float('inf')
        best_predictions = None
        
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        if len(y.shape) == 2:
            y = y.unsqueeze(0)  # Add batch dimension
        
        # Create mask for valid targets
        valid_mask = ~torch.isnan(y)
        
        # Calculate number of sequences
        sequence_length = x.size(1)
        prediction_window = self.output_size
        num_sequences = (sequence_length - prediction_window) // prediction_window + 1
        
        # Initialize storage for all predictions
        all_predictions = []
        current_predictions = None
        
        # Process each sequence
        for seq_idx in range(num_sequences):
            # Get current sequence
            start_idx = seq_idx * prediction_window
            end_idx = start_idx + sequence_length
            current_x = x[:, start_idx:end_idx, :]
            
            # Initialize predictions for this sequence
            sequence_predictions = None
            sequence_loss = float('inf')
            
            # Process current sequence
            optimizer.zero_grad()
            
            # Make predictions using current state and previous sequence predictions
            predictions = self(current_x, seq_idx=seq_idx, prev_predictions=current_predictions, is_training=True)
            
            # Only calculate loss on valid targets
            valid_predictions = predictions[valid_mask.squeeze(-1)]
            valid_targets = y[valid_mask].squeeze(-1)
            
            if len(valid_predictions) > 0:
                # Ensure shapes match by unsqueezing if needed
                if valid_targets.dim() == 0:
                    valid_targets = valid_targets.unsqueeze(0)
                if valid_predictions.dim() == 0:
                    valid_predictions = valid_predictions.unsqueeze(0)
                
                # Calculate loss only on valid targets and take mean to get scalar
                loss = criterion(valid_predictions, valid_targets).mean()
                
                # Backpropagate the scalar loss
                loss.backward()
                optimizer.step()
                
                # Store predictions for this sequence
                sequence_predictions = predictions.detach()
                sequence_loss = loss.item()
                
                # Store predictions for this sequence
                if sequence_predictions is not None:
                    all_predictions.append(sequence_predictions)
                    current_predictions = sequence_predictions
                
                # Update best overall predictions
                if sequence_loss < best_loss:
                    best_loss = sequence_loss
                    best_predictions = sequence_predictions
            else:
                print(f"Warning: No valid targets in sequence {seq_idx + 1}")
        
        return best_predictions, best_loss

class WaterLevelForecaster:
    """
    Main class to handle water level forecasting.
    """
    def __init__(self, config):
        """
        Initialize the water level forecasting model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.prediction_window = config.get('prediction_window', 24)
    
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
            dropout=dropout
        ).to(self.device)
        
        self.model = model
        return model
    
    def train(self, train_data, val_data, project_root, station_id):
        """
        Train the model using the provided training and validation data.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            project_root: Path to project root directory
            station_id: ID of the station being processed
            
        Returns:
            Trained model
        """
        # Initialize preprocessor
        preprocessor = DataPreprocessor(self.config)
        
        # Prepare training data
        train_sequences = preprocessor.prepare_data(train_data, is_training=True)
        val_sequences = preprocessor.prepare_data(val_data, is_training=False)
        
        # Initialize model
        input_size = train_sequences[0].shape[2]  # Get number of features
        self.model = self.build_model(input_size)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = nn.SmoothL1Loss(reduction='none')  # Use 'none' to handle masking
        
        # Training loop
        num_epochs = self.config.get('epochs', 10)
        best_val_loss = float('inf')
        patience = self.config.get('patience', 5)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            # Process each sequence in the training data
            for i in range(len(train_sequences[0])):
                # Get current sequence and target
                batch_x = train_sequences[0][i:i+1]  # Add batch dimension
                batch_y = train_sequences[1][i:i+1]  # Add batch dimension
                
                # Create mask for valid targets (non-NaN)
                valid_mask = ~torch.isnan(batch_y)
                
                # Skip if all targets are NaN
                if not valid_mask.any():
                    continue
                
                # Train iteratively
                predictions, loss = self.model.train_iteratively(
                    batch_x, batch_y, optimizer, criterion,
                    convergence_threshold=self.config.get('convergence_threshold', 0.01)
                )
                
                # Apply mask to loss and calculate mean only over valid targets
                masked_loss = (loss * valid_mask.float()).sum()
                num_valid = valid_mask.float().sum()
                
                if num_valid > 0:
                    train_loss += masked_loss.item()
                    train_samples += num_valid.item()
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for i in range(len(val_sequences[0])):
                    batch_x = val_sequences[0][i:i+1]  # Add batch dimension
                    batch_y = val_sequences[1][i:i+1]  # Add batch dimension
                    
                    # Create mask for valid targets
                    valid_mask = ~torch.isnan(batch_y)
                    
                    # Skip if all targets are NaN
                    if not valid_mask.any():
                        continue
                    
                    predictions = self.model(batch_x)
                    loss = criterion(predictions, batch_y.squeeze(-1))
                    
                    # Apply mask to loss
                    masked_loss = (loss * valid_mask.squeeze(-1).float()).sum()
                    num_valid = valid_mask.squeeze(-1).float().sum()
                    
                    if num_valid > 0:
                        val_loss += masked_loss.item()
                        val_samples += num_valid.item()
            
            # Calculate average validation loss
            avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Training Loss: {avg_train_loss:.6f} (over {train_samples} valid samples)")
            print(f"  Validation Loss: {avg_val_loss:.6f} (over {val_samples} valid samples)")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model(os.path.join(project_root, f"models/{station_id}_best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        return self.model
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.eval()
        
        # Prepare data
        features, targets = self.preprocessor.prepare_data(data, is_training=False)
        
        # Initialize storage for predictions
        all_predictions = []
        
        with torch.no_grad():
            # Process each sequence
            for i in range(features.size(0)):
                sequence = features[i:i+1]  # Add batch dimension
                
                # Make prediction
                pred = self.model(sequence)
                
                # Store results
                all_predictions.append(pred.cpu().numpy())
        
        # Combine predictions
        predictions = np.concatenate(all_predictions)
        
        # Create results dictionary
        results = {
            'forecasts': predictions,
            'clean_data': data[self.preprocessor.output_features].values,
            'timestamps': data.index  # Add timestamps from the original data
        }
        
        return results
    
    def save_model(self, path):
        """
        Save the trained model to a file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
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
