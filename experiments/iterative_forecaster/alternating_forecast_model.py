import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add the necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

class AlternatingForecastModel(nn.Module):
    """
    LSTM model for water level forecasting that explicitly alternates between
    using ground truth and its own predictions during training.
    
    Key features:
    - Uses LSTMCell for explicit control over hidden states
    - Alternates between 1 week of original data and 1 week of prediction data
    - Includes binary flag indicating if input is original (0) or predicted (1)
    - Handles additional engineered features (time features, cumulative features)
    """
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.25, config=None):
        super(AlternatingForecastModel, self).__init__()
        self.model_name = 'AlternatingForecastModel'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.debug_mode = True  # Add debug mode flag
        
        #print(f"\nModel initialization:")
        #print(f"Input size: {input_size}")
        #print(f"Hidden size: {hidden_size}")
        #print(f"Output size: {output_size}")
        
        # Single LSTM cell that takes input + binary flag
        self.lstm_cell = nn.LSTMCell(input_size + 1, hidden_size)  # +1 for binary flag
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (always predicts 1 time step)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Define the number of time steps in 1 week (168 hours = 672 time steps)
        # Use config value if provided
        if config and 'week_steps' in config:
            self.week_steps = config['week_steps']
        else:
            self.week_steps = 672

    
    def forward(self, x, hidden_state=None, cell_state=None, use_predictions=False, 
                weekly_mask=None, alternating_weeks=True):
        """
        Forward pass with explicit control over hidden states and alternating input strategy.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
            hidden_state: Hidden state for LSTM cell, or None to initialize
            cell_state: Cell state for LSTM cell, or None to initialize
            use_predictions: Whether to use the model's own predictions as input
            weekly_mask: Optional binary mask indicating which time steps should use 
                        original data (0) vs predictions (1)
            alternating_weeks: Whether to use 1-week alternating pattern for training
        
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, output_size)
            hidden_state: Updated hidden state
            cell_state: Updated cell state
        """
        batch_size, seq_len, feature_dim = x.size()
        device = x.device
        
        # Initialize hidden and cell states if not provided
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
            cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        
        # Initialize the alternating week pattern if needed
        if alternating_weeks and weekly_mask is None:
            # Create weekly mask for alternating pattern
            weekly_mask = torch.zeros(seq_len, device=device)
            
            # Set every other week to use predictions (1)
            for i in range(self.week_steps, seq_len, self.week_steps * 2):
                if i + self.week_steps <= seq_len:
                    weekly_mask[i:i+self.week_steps] = 1
            
            #print("\nInitial weekly mask setup:")
            #print(f"Total sequence length: {seq_len}")
            #print(f"Week steps: {self.week_steps}")
            #print(f"Number of weeks where predictions will be used: {weekly_mask.sum() / self.week_steps:.0f}")
            #print(f"use_predictions flag value: {use_predictions}")
                    
        elif weekly_mask is None:
            weekly_mask = torch.zeros(seq_len, device=device)
        
        # Loop through sequence
        current_input = x[:, 0, :]
        
        # Track statistics for debugging
        n_used_original = 0
        n_used_prediction = 0
        last_data_type = None
        
        for t in range(seq_len):
            # Determine if we should use original data or prediction
            use_original = (weekly_mask[t].item() == 0) or not use_predictions
            
            current_data_type = 'original' if use_original else 'prediction'
            if current_data_type != last_data_type:
                #print(f"\nSwitching at timestep {t} to using {current_data_type} data")
                last_data_type = current_data_type
            
            if not use_original and t > 0:
                n_used_prediction += 1
                binary_flag = torch.ones(batch_size, 1, device=device)
                # Use the last prediction as input for the current timestep
                pred_input = x[:, t, :].clone()
                pred_input[:, 0] = outputs[:, t-1, 0]  # Water level is always first feature
                current_input = torch.cat([pred_input, binary_flag], dim=1)
            else:
                n_used_original += 1
                binary_flag = torch.zeros(batch_size, 1, device=device)
                current_input = torch.cat([x[:, t, :], binary_flag], dim=1)
            
            # Process through LSTM cell
            hidden_state, cell_state = self.lstm_cell(current_input, (hidden_state, cell_state))
            
            # Apply dropout to hidden state
            final_hidden = self.dropout(hidden_state)
            
            # Generate prediction for current timestep only
            outputs[:, t, :] = self.output_layer(final_hidden)
            
            # Print debug information for first batch of first epoch
            #if self.debug_mode and t % 672 == 0:
                #print(f"\nTimestep {t}:")
                #print(f"Using {'original' if use_original else 'prediction'} data")
                #print(f"Current input shape: {current_input.shape}")
                #print(f"Output shape: {outputs[:, t, :].shape}")
                #if t > 0:
                #    print(f"Previous prediction: {outputs[:, t-1, 0].mean().item():.2f}")
                #    print(f"Current prediction: {outputs[:, t, 0].mean().item():.2f}")
                
        
        #print(f"\nFinal usage statistics:")
        #print(f"Total timesteps: {seq_len}")
        #print(f"Used original data: {n_used_original} times")
        #print(f"Used predictions: {n_used_prediction} times")
        #print(f"Original data percentage: {n_used_original/seq_len*100:.1f}%")
        #print(f"Predictions percentage: {n_used_prediction/seq_len*100:.1f}%")
    
        return outputs, hidden_state, cell_state
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state and cell state."""
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state 