import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add the necessary paths
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

class AlternatingForecastModel(nn.Module):
    """
    LSTM model for water level forecasting that uses real-time z-score anomaly detection
    to decide whether to use original data or model predictions as input.
    
    Key features:
    - Uses LSTMCell for explicit control over hidden states
    - Real-time z-score calculation for anomaly detection
    - Switches between original data and predictions based on anomaly flags
    - Includes warmup period with alternating pattern
    - Feeds anomaly flags back to the model as additional input
    """
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.25, config=None):
        super(AlternatingForecastModel, self).__init__()
        self.model_name = 'AlternatingForecastModel_ZScore'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.debug_mode = True  # Add debug mode flag
        
        # Z-score anomaly detection parameters from config
        self.config = config or {}
        self.zscore_threshold = self.config.get('zscore_threshold', 3.0)
        self.zscore_window_size = self.config.get('zscore_window_size', 168)  # 1 week default
        self.warmup_steps = self.config.get('zscore_warmup_steps', 672)  # 1 month default
        self.use_mad = self.config.get('use_mad_zscore', True)  # Use MAD-based z-score by default
        
        print(f"\nZ-Score Anomaly Detection Configuration:")
        print(f"  Threshold: {self.zscore_threshold}")
        print(f"  Window size: {self.zscore_window_size}")
        print(f"  Warmup steps: {self.warmup_steps}")
        print(f"  Use MAD: {self.use_mad}")
        
        # Define the number of time steps in 1 week for warmup alternating pattern
        if config and 'week_steps' in config:
            self.week_steps = config['week_steps']
        else:
            self.week_steps = 672
        
        # LSTM cell that takes input + binary flag + anomaly flag
        self.lstm_cell = nn.LSTMCell(input_size + 2, hidden_size)  # +2 for binary flag and anomaly flag
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (always predicts 1 time step)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Buffers for storing residuals for z-score calculation
        self.residual_buffer = None
        self.buffer_initialized = False
    
    def initialize_residual_buffer(self, seq_len, device):
        """Initialize the residual buffer for z-score calculation."""
        buffer_size = max(self.zscore_window_size * 2, seq_len)
        self.residual_buffer = torch.full((buffer_size,), float('nan'), device=device)
        self.buffer_ptr = 0
        self.buffer_initialized = True
        
    def calculate_zscore_torch(self, residual, device):
        """
        Calculate z-score for the current residual using the buffer of previous residuals.
        
        Args:
            residual: Current residual value (torch tensor)
            device: Device for computation
            
        Returns:
            z_score: Z-score value (torch tensor)
            is_anomaly: Boolean indicating if residual is anomalous (torch tensor)
        """
        if not self.buffer_initialized:
            return torch.tensor(0.0, device=device), torch.tensor(False, device=device)
        
        # Add current residual to buffer
        self.residual_buffer[self.buffer_ptr] = residual
        self.buffer_ptr = (self.buffer_ptr + 1) % len(self.residual_buffer)
        
        # Get valid residuals from buffer for z-score calculation
        valid_mask = ~torch.isnan(self.residual_buffer)
        valid_residuals = self.residual_buffer[valid_mask]
        
        # Need at least half the window size of valid data
        min_required = max(10, self.zscore_window_size // 2)
        if len(valid_residuals) < min_required:
            return torch.tensor(0.0, device=device), torch.tensor(False, device=device)
        
        # Use most recent valid residuals up to window size
        recent_residuals = valid_residuals[-self.zscore_window_size:]
        
        if self.use_mad:
            # MAD-based robust z-score
            median = torch.median(recent_residuals)
            mad = torch.median(torch.abs(recent_residuals - median))
            
            if mad == 0:
                return torch.tensor(0.0, device=device), torch.tensor(False, device=device)
            
            z_score = (residual - median) / (mad * 1.4826)
        else:
            # Standard z-score
            mean = torch.mean(recent_residuals)
            std = torch.std(recent_residuals)
            
            if std == 0:
                return torch.tensor(0.0, device=device), torch.tensor(False, device=device)
            
            z_score = (residual - mean) / std
        
        is_anomaly = torch.abs(z_score) > self.zscore_threshold
        
        return z_score, is_anomaly
    
    def forward(self, x, hidden_state=None, cell_state=None, use_predictions=False, 
                weekly_mask=None, alternating_weeks=True):
        """
        Forward pass with z-score anomaly detection for adaptive input selection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
            hidden_state: Hidden state for LSTM cell, or None to initialize
            cell_state: Cell state for LSTM cell, or None to initialize
            use_predictions: Whether to enable the z-score switching mechanism
            weekly_mask: Not used in this version (kept for compatibility)
            alternating_weeks: Whether to use alternating pattern during warmup
        
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, output_size)
            hidden_state: Updated hidden state
            cell_state: Updated cell state
            anomaly_info: Dictionary containing anomaly detection information from forward pass
        """
        batch_size, seq_len, feature_dim = x.size()
        device = x.device
        
        # Initialize hidden and cell states if not provided
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
            cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Initialize residual buffer
        self.initialize_residual_buffer(seq_len, device)
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        
        # Initialize warmup alternating pattern
        warmup_mask = torch.zeros(seq_len, device=device)
        if alternating_weeks and self.warmup_steps > 0:
            # Create alternating pattern for warmup period
            warmup_end = min(self.warmup_steps, seq_len)
            for i in range(self.week_steps, warmup_end, self.week_steps * 2):
                if i + self.week_steps <= warmup_end:
                    warmup_mask[i:i+self.week_steps] = 1
        
        # Track statistics and anomaly information for debugging and visualization
        n_used_original = 0
        n_used_prediction = 0
        n_anomalies_detected = 0
        n_warmup_steps = 0
        
        # Storage for anomaly detection information
        z_scores_list = []
        anomaly_flags_list = []
        usage_decisions_list = []  # 0 = original, 1 = prediction
        in_warmup_list = []
        
        if self.debug_mode:
            print(f"\nZ-Score Alternating Forward Pass:")
            print(f"Sequence length: {seq_len}")
            print(f"Warmup steps: {min(self.warmup_steps, seq_len)}")
            print(f"Z-score threshold: {self.zscore_threshold}")
        
        # Loop through sequence
        for t in range(seq_len):
            # Determine if we're in warmup period
            in_warmup = t < self.warmup_steps
            in_warmup_list.append(in_warmup)
            
            if in_warmup:
                # Use alternating pattern during warmup
                use_original = (warmup_mask[t].item() == 0) or not use_predictions
                anomaly_flag = torch.zeros(batch_size, 1, device=device)  # No anomaly detection during warmup
                n_warmup_steps += 1
                z_score_value = 0.0  # No z-score during warmup
                
                if self.debug_mode and t % 672 == 0:
                    print(f"Warmup at t={t}: using {'original' if use_original else 'prediction'}")
            else:
                # Use z-score anomaly detection after warmup
                if t == 0 or not use_predictions:
                    # First timestep or not using predictions - use original data
                    use_original = True
                    anomaly_flag = torch.zeros(batch_size, 1, device=device)
                    z_score_value = 0.0
                else:
                    # Calculate residual between observation and previous prediction
                    current_observation = x[:, t, 0]  # Water level is first feature
                    previous_prediction = outputs[:, t-1, 0]
                    residual = current_observation - previous_prediction
                    
                    # Calculate z-score and anomaly flag for each sample in batch
                    # For simplicity, we'll use the first sample's residual for buffer management
                    z_score, is_anomaly = self.calculate_zscore_torch(residual[0], device)
                    z_score_value = z_score.item()
                    
                    # Use anomaly detection to decide input
                    use_original = not is_anomaly.item()
                    anomaly_flag = is_anomaly.float().unsqueeze(0).unsqueeze(1).expand(batch_size, 1)
                    
                    if is_anomaly.item():
                        n_anomalies_detected += 1
                        if self.debug_mode and t % 672 == 0:
                            print(f"Anomaly detected at t={t}: z-score={z_score:.2f}, using prediction")
                    elif self.debug_mode and t % 672 == 0:
                        print(f"Normal at t={t}: z-score={z_score:.2f}, using original")
            
            # Store anomaly detection information
            z_scores_list.append(z_score_value)
            anomaly_flags_list.append(anomaly_flag[0, 0].item() if isinstance(anomaly_flag, torch.Tensor) else 0.0)
            usage_decisions_list.append(0 if use_original else 1)
            
            # Prepare input based on decision
            if not use_original and t > 0:
                n_used_prediction += 1
                # Use the last prediction as input for the current timestep
                pred_input = x[:, t, :].clone()
                pred_input[:, 0] = outputs[:, t-1, 0]  # Water level is always first feature
                binary_flag = torch.ones(batch_size, 1, device=device)
                current_input = torch.cat([pred_input, binary_flag, anomaly_flag], dim=1)
            else:
                n_used_original += 1
                # Use original data
                binary_flag = torch.zeros(batch_size, 1, device=device)
                current_input = torch.cat([x[:, t, :], binary_flag, anomaly_flag], dim=1)
            
            # Process through LSTM cell
            hidden_state, cell_state = self.lstm_cell(current_input, (hidden_state, cell_state))
            
            # Apply dropout to hidden state
            final_hidden = self.dropout(hidden_state)
            
            # Generate prediction for current timestep
            outputs[:, t, :] = self.output_layer(final_hidden)
        
        if self.debug_mode:
            print(f"\nFinal Z-Score Usage Statistics:")
            print(f"Total timesteps: {seq_len}")
            print(f"Warmup steps: {n_warmup_steps}")
            print(f"Used original data: {n_used_original} times ({n_used_original/seq_len*100:.1f}%)")
            print(f"Used predictions: {n_used_prediction} times ({n_used_prediction/seq_len*100:.1f}%)")
            print(f"Anomalies detected: {n_anomalies_detected} ({n_anomalies_detected/max(1,seq_len-self.warmup_steps)*100:.1f}% of post-warmup)")
        
        # Prepare anomaly information dictionary
        anomaly_info = {
            'z_scores': z_scores_list,
            'anomaly_flags': anomaly_flags_list,
            'usage_decisions': usage_decisions_list,  # 0 = original, 1 = prediction
            'in_warmup': in_warmup_list,
            'n_used_original': n_used_original,
            'n_used_prediction': n_used_prediction,
            'n_anomalies_detected': n_anomalies_detected,
            'warmup_steps': self.warmup_steps,
            'zscore_threshold': self.zscore_threshold
        }
        
        return outputs, hidden_state, cell_state, anomaly_info
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state and cell state."""
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state 