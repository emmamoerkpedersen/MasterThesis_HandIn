import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

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
    - Statistical anomaly detection with z-scores and anomaly flags (MAD-based)
    - Feeds back anomaly information as part of the internal state
    - Handles additional engineered features (time features, cumulative features)
    - Option to use either MAD-based or synthetic error flags
    """
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.25, config=None):
        super(AlternatingForecastModel, self).__init__()
        self.model_name = 'AlternatingForecastModel'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.debug_mode = True  # Add debug mode flag
        
        # Anomaly detection parameters
        self.window_size = config.get('window_size', 16) if config else 16
        self.threshold = config.get('threshold', 3.0) if config else 3.0
        
        # Buffer to store recent residuals for z-score calculation
        self.residual_buffer = None
        self.buffer_initialized = False
        
        #print(f"\nModel initialization:")
        #print(f"Input size: {input_size}")
        #print(f"Hidden size: {hidden_size}")
        #print(f"Output size: {output_size}")
        
        # LSTM cell that takes input + binary flag + z_score + anomaly_flag
        # input_size + 1 (binary flag) + 1 (z_score) + 1 (anomaly_flag)
        self.lstm_cell = nn.LSTMCell(input_size + 3, hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (only predicts water level, no anomaly probability)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Define the number of time steps in 1 week (168 hours = 672 time steps)
        # Use config value if provided
        if config and 'week_steps' in config:
            self.week_steps = config['week_steps']
        else:
            self.week_steps = 672

    def _calculate_z_score_mad_torch(self, residuals_buffer, current_residual):
        """
        Calculate z-score using MAD method in PyTorch for differentiability.
        
        Args:
            residuals_buffer: Tensor of shape (window_size,) containing recent residuals
            current_residual: Current residual value
            
        Returns:
            z_score: MAD-based z-score
        """
        # Remove NaN values from buffer
        valid_residuals = residuals_buffer[~torch.isnan(residuals_buffer)]
        
        if len(valid_residuals) < self.window_size * 0.5:
            return torch.tensor(0.0, device=residuals_buffer.device, dtype=torch.float32)
        
        # Calculate median and MAD
        median = torch.median(valid_residuals)
        mad = torch.median(torch.abs(valid_residuals - median))
        
        # Avoid division by zero
        if mad < 1e-8:
            return torch.tensor(0.0, device=residuals_buffer.device, dtype=torch.float32)
        
        # Robust z-score calculation
        z_score = (current_residual - median) / (mad * 1.4826)
        
        return z_score

    def _initialize_residual_buffer(self, batch_size, device):
        """Initialize the residual buffer for z-score calculation."""
        self.residual_buffer = torch.full((batch_size, self.window_size), 
                                         torch.nan, device=device, dtype=torch.float32)
        self.buffer_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.buffer_initialized = True

    def _update_residual_buffer(self, batch_idx, residual):
        """Update the residual buffer with a new residual value."""
        ptr = self.buffer_ptr[batch_idx]
        self.residual_buffer[batch_idx, ptr] = residual
        self.buffer_ptr[batch_idx] = (ptr + 1) % self.window_size
    
    def forward(self, x, hidden_state=None, cell_state=None, use_predictions=False, 
                weekly_mask=None, alternating_weeks=True, y_true=None):
        """
        Forward pass with explicit control over hidden states, alternating input strategy,
        and statistical anomaly detection only.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
            hidden_state: Hidden state for LSTM cell, or None to initialize
            cell_state: Cell state for LSTM cell, or None to initialize
            use_predictions: Whether to use the model's own predictions as input
            weekly_mask: Optional binary mask indicating which time steps should use 
                        original data (0) vs predictions (1)
            alternating_weeks: Whether to use 1-week alternating pattern for training
            y_true: True target values for anomaly detection (optional)
        
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, output_size)
            hidden_state: Updated hidden state
            cell_state: Updated cell state
            z_scores: Z-scores for anomaly detection
            anomaly_flags: Binary anomaly flags
            anomaly_probs: None (placeholder for compatibility)
        """
        batch_size, seq_len, feature_dim = x.size()
        device = x.device

        # Initialize hidden and cell states if not provided
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
            cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Initialize residual buffer
        if not self.buffer_initialized or self.residual_buffer.size(0) != batch_size:
            self._initialize_residual_buffer(batch_size, device)
        
        # Storage for outputs and anomaly information
        outputs = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        z_scores = torch.zeros(batch_size, seq_len, device=device)
        anomaly_flags = torch.zeros(batch_size, seq_len, device=device)
        
        # Initialize the alternating week pattern if needed
        if alternating_weeks and weekly_mask is None:
            # Create weekly mask for alternating pattern
            weekly_mask = torch.zeros(seq_len, device=device)
            
            # Set every other week to use predictions (1)
            for i in range(self.week_steps, seq_len, self.week_steps * 2):
                if i + self.week_steps <= seq_len:
                    weekly_mask[i:i+self.week_steps] = 1
                    
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
                last_data_type = current_data_type
            
            # Prepare input with anomaly information from PREVIOUS timestep
            if not use_original and t > 0:
                n_used_prediction += 1
                binary_flag = torch.ones(batch_size, 1, device=device)
                # Use the last prediction as input for the current timestep
                pred_input = x[:, t, :].clone()
                pred_input[:, 0] = outputs[:, t-1, 0]  # Water level is always first feature
            else:
                n_used_original += 1
                binary_flag = torch.zeros(batch_size, 1, device=device)
                pred_input = x[:, t, :]
            
            # Use anomaly information from previous timestep for current input
            prev_z_score = z_scores[:, t-1] if t > 0 else torch.zeros(batch_size, device=device)
            prev_anomaly_flag = anomaly_flags[:, t-1] if t > 0 else torch.zeros(batch_size, device=device)
            
            # Create augmented input with previous anomaly information
            # [features, binary_flag, prev_z_score, prev_anomaly_flag]
            current_input = torch.cat([
                pred_input, 
                binary_flag, 
                prev_z_score.unsqueeze(1),
                prev_anomaly_flag.unsqueeze(1)
            ], dim=1)
            
            # Process through LSTM cell
            hidden_state, cell_state = self.lstm_cell(current_input, (hidden_state, cell_state))
            
            # Apply dropout to hidden state
            final_hidden = self.dropout(hidden_state)
            
            # Generate prediction for current timestep (only water level)
            outputs[:, t, :] = self.output_layer(final_hidden)
            
            # NOW calculate anomaly detection for current timestep (after we have the prediction)
            current_z_score = torch.zeros(batch_size, device=device)
            current_anomaly_flag = torch.zeros(batch_size, device=device)
            
            if t >= self.window_size and y_true is not None:
                for b in range(batch_size):
                    # Compare current true value with current prediction
                    true_val = y_true[b, t, 0] if len(y_true.shape) == 3 else y_true[b, t]
                    pred_val = outputs[b, t, 0]
                    
                    if not torch.isnan(true_val) and not torch.isnan(pred_val):
                        # Calculate residual for current timestep
                        residual = (true_val - pred_val).detach()
                        self._update_residual_buffer(b, residual)
                        
                        # Calculate z-score
                        z_score = self._calculate_z_score_mad_torch(
                            self.residual_buffer[b], residual
                        )
                        current_z_score[b] = z_score.detach()
                        
                        # Set anomaly flag for current timestep
                        if torch.abs(z_score) > self.threshold:
                            current_anomaly_flag[b] = 1.0
            
            # Store z-scores and flags for current timestep
            z_scores[:, t] = current_z_score
            anomaly_flags[:, t] = current_anomaly_flag
        
        # Return None for anomaly_probs to maintain interface compatibility
        return outputs, hidden_state, cell_state, z_scores, anomaly_flags, None 