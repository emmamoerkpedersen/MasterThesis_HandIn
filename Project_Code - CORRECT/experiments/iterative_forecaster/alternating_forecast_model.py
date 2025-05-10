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
    - Can be configured for single or multiple LSTM cells
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout=0.25, config=None):
        super(AlternatingForecastModel, self).__init__()
        self.model_name = 'AlternatingForecastModel'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        print(f"Initializing model with input_size={input_size}, adding 1 for binary flag = {input_size+1}")
        
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
        
        # Define the number of time steps in 1 week (168 hours)
        # Use config value if provided
        if config and 'week_steps' in config:
            self.week_steps = config['week_steps']
        else:
            self.week_steps = 672
    
    def forward(self, x, hidden_states=None, cell_states=None, use_predictions=False, 
                forcing_mask=None, alternating_weeks=True):
        """
        Forward pass with explicit control over hidden states and alternating input strategy.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
            hidden_states: List of hidden states for each layer, or None to initialize
            cell_states: List of cell states for each layer, or None to initialize
            use_predictions: Whether to use the model's own predictions as input
            forcing_mask: Optional binary mask indicating which time steps should use 
                          original data (1) vs predictions (0)
            alternating_weeks: Whether to use 1-week alternating pattern for training
        
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, output_size)
            hidden_states: Updated hidden states for each layer
            cell_states: Updated cell states for each layer
        """
        batch_size, seq_len, feature_dim = x.size()
        device = x.device
        
        # The water level feature is always the first feature
        water_level_idx = 0
        
        # Initialize hidden and cell states if not provided
        if hidden_states is None or cell_states is None:
            hidden_states = []
            cell_states = []
            for i in range(self.num_layers):
                hidden_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
                cell_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        
        # Initialize the alternating week pattern if needed
        if alternating_weeks and forcing_mask is None:
            # Create forcing mask for alternating weeks pattern
            # As per professor's suggestion:
            # - 0 for weeks where we use original observations (first week)
            # - 1 for weeks where we use model predictions (second week)
            forcing_mask = torch.ones(seq_len, device=device)  # Default to using predictions
            
            # Set first week and every other week to use observations (0)
            for i in range(0, seq_len, self.week_steps * 2):
                if i + self.week_steps <= seq_len:
                    forcing_mask[i:i+self.week_steps] = 0
        elif forcing_mask is None:
            # Default to using original data if no pattern specified
            forcing_mask = torch.zeros(seq_len, device=device)  # All zeros = all observations
        
        # Loop through sequence
        current_input = x[:, 0, :]
        
        for t in range(seq_len):
            # Determine if we use original observations or predicted data for this step
            # If forcing_mask[t] == 0, use original observations
            # If forcing_mask[t] == 1, use predictions (if available)
            use_original = (forcing_mask[t].item() == 0) or t == 0 or not use_predictions
            
            if not use_original and t > 0:
                # Use previous prediction as input
                # Create binary flag (1 for predicted input)
                binary_flag = torch.ones(batch_size, 1, device=device)  # 1 = using prediction
                
                # Replace target feature with previous prediction
                pred_input = x[:, t, :].clone()
                pred_input[:, water_level_idx] = outputs[:, t-1, 0]  # Use last prediction for water level
                
                # Add binary flag
                current_input = torch.cat([pred_input, binary_flag], dim=1)
            else:
                # Use original data
                # Create binary flag (0 for original input)
                binary_flag = torch.zeros(batch_size, 1, device=device)  # 0 = using observation
                
                # Use original input with flag
                current_input = torch.cat([x[:, t, :], binary_flag], dim=1)
            
            # Process through LSTM cells
            for i in range(self.num_layers):
                if i == 0:
                    # First layer gets current input
                    hidden_states[i], cell_states[i] = self.lstm_cells[i](
                        current_input, (hidden_states[i], cell_states[i])
                    )
                else:
                    # Subsequent layers get hidden state from previous layer
                    # Apply dropout between layers
                    layer_input = self.dropout(hidden_states[i-1])
                    hidden_states[i], cell_states[i] = self.lstm_cells[i](
                        layer_input, (hidden_states[i], cell_states[i])
                    )
            
            # Apply dropout to final hidden state
            final_hidden = self.dropout(hidden_states[-1])
            
            # Generate prediction for current timestep
            outputs[:, t, :] = self.output_layer(final_hidden)
        
        return outputs, hidden_states, cell_states
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states and cell states."""
        hidden_states = []
        cell_states = []
        for i in range(self.num_layers):
            hidden_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
            cell_states.append(torch.zeros(batch_size, self.hidden_size, device=device))
        return hidden_states, cell_states 