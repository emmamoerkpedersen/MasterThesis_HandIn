import torch.nn as nn
from pathlib import Path
import sys


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
    