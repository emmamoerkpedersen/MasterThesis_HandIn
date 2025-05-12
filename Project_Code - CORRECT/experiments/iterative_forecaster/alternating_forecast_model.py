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
        
        print(f"\nModel initialization:")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Output size: {output_size}")
        print(f"Number of layers: {num_layers}")
        
        # LSTMCell layers for explicit state control
        self.lstm_cells = nn.ModuleList()
        
        # First layer takes input + binary flag
        self.lstm_cells.append(nn.LSTMCell(input_size + 1, hidden_size))  # +1 for binary flag
        
        # Additional layers (if num_layers > 1)
        for i in range(1, num_layers):
            self.lstm_cells.append(nn.LSTMCell(hidden_size, hidden_size))
        
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
    
    def forward(self, x, hidden_states=None, cell_states=None, use_predictions=False, 
                weekly_mask=None, alternating_weeks=True):
        """
        Forward pass with explicit control over hidden states and alternating input strategy.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
            hidden_state: Hidden state for LSTM cell, or None to initialize
            cell_state: Cell state for LSTM cell, or None to initialize
            use_predictions: Whether to use the model's own predictions as input
            weekly_mask: Optional binary mask indicating which time steps should use 
                          original data (1) vs predictions (0)
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
        
        if hidden_states is None or cell_states is None:
            hidden_states = []
            cell_states = []
            for i in range(self.num_layers):
                hidden_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
                cell_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
            
        # Storage for outputs
        outputs = torch.zeros(batch_size, seq_len, self.output_size, device=device)
    
        
        # Initialize the alternating week pattern if needed
        if alternating_weeks and weekly_mask is None:
            # Create forcing mask for alternating weeks pattern
            # As per professor's suggestion:
            # 0 for weeks where we use original observations (first week)
            # 1 for weeks where we use model predictions (second week)
            weekly_mask = torch.ones(seq_len, device=device)  # Default to using predictions
            
            # Set first week and every other week to use observations (0)
            for i in range(0, seq_len, self.week_steps * 2):
                if i + self.week_steps <= seq_len:
                    weekly_mask[i:i+self.week_steps] = 0
            
           
        elif weekly_mask is None:
            # Default to using original data if no pattern specified
            weekly_mask = torch.zeros(seq_len, device=device)  # All zeros = all observations
        
        # Loop through sequence
        current_input = x[:, 0, :]
        
        # Track statistics for debugging
        n_used_original = 0
        n_used_prediction = 0
        last_switch = 0
        
        for t in range(seq_len):
            use_original = (weekly_mask[t].item() == 0) or t == 0 or not use_predictions
            
            if not use_original and t > 0:
                n_used_prediction += 1
                binary_flag = torch.ones(batch_size, 1, device=device)
                pred_input = x[:, t, :].clone()
                pred_input[:, 0] = outputs[:, t-1, 0]
                current_input = torch.cat([pred_input, binary_flag], dim=1)
                
                # Print when we switch to using predictions
                if t - last_switch >= self.week_steps:
                    print(f"\nSwitching to predictions at t={t}")
                    print(f"Original value: {x[:, t, 0].item():.4f}")
                    print(f"Using prediction: {outputs[:, t-1, 0].item():.4f}")
                    last_switch = t
            else:
                n_used_original += 1
                binary_flag = torch.zeros(batch_size, 1, device=device)
                current_input = torch.cat([x[:, t, :], binary_flag], dim=1)
                
                # Print when we switch to using original data
                if t - last_switch >= self.week_steps:
                    print(f"\nSwitching to original data at t={t}")
                    print(f"Using original value: {x[:, t, 0].item():.4f}")
                    last_switch = t
            
            # Process through LSTM cells
            for i in range(self.num_layers):
                if i == 0:
                    hidden_states[i], cell_states[i] = self.lstm_cells[i](
                        current_input, (hidden_states[i], cell_states[i])
                    )
                else:
                    layer_input = self.dropout(hidden_states[i-1])
                    hidden_states[i], cell_states[i] = self.lstm_cells[i](
                        layer_input, (hidden_states[i], cell_states[i])
                    )
                
                if t == 0 and i == 0:  # Print shapes only for first timestep and layer
                    print(f"Layer {i} output: {hidden_states[i].shape}")
            
            final_hidden = self.dropout(hidden_states[-1])
            outputs[:, t, :] = self.output_layer(final_hidden)
            test  = 1

        print(f"\nFinal usage statistics:")
        print(f"Total timesteps: {seq_len}")
        print(f"Used original data: {n_used_original} ({n_used_original/seq_len*100:.1f}%)")
        print(f"Used predictions: {n_used_prediction} ({n_used_prediction/seq_len*100:.1f}%)")
        
        return outputs, hidden_state, cell_state
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state and cell state."""
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state 