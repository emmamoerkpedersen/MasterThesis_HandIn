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
    - Handles additional engineered features (time features, cumulative features)
    - NEW: Accepts anomaly flags as input features
    """
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.25, config=None):
        super(AlternatingForecastModel, self).__init__()
        self.model_name = 'AlternatingForecastModel'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.config = config or {}
        
        # Main LSTM cell
        self.lstm_cell = nn.LSTMCell(input_size + 1, hidden_size)  # +1 for binary flag
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Week steps configuration for alternating training
        self.week_steps = config.get('week_steps', 672) if config else 672
        
        # Anomaly-aware training parameters
        self.use_weighted_loss = config.get('use_weighted_loss', False) if config else False
        self.anomaly_weight = config.get('anomaly_weight', 0.3) if config else 0.3
        
        # Store anomaly flag column name
        self.anomaly_flag_column = config.get('anomaly_flag_column', 'anomaly_flag') if config else 'anomaly_flag'
        self.anomaly_flag_idx = None  # Will be set during forward pass
        
        # ===============================
        # EXPERIMENT 8: SLIDING WINDOW MEMORY
        # ===============================
        # Instead of single last_good_cell_state, keep buffer of recent good states
        self.memory_window_size = config.get('memory_window_size', 10) if config else 10
        self.good_memory_buffer = []  # Will store recent good cell states
        self.use_sliding_window = config.get('use_sliding_window', True) if config else True
    
    def forward(self, x, hidden_state=None, cell_state=None, use_predictions=False, 
                weekly_mask=None, alternating_weeks=True, feature_names=None):
        """
        Forward pass with memory protection against anomalous data.
        
        MEMORY PROTECTION STRATEGY:
        1. During normal periods: Update LSTM memory normally and save as "good backup"
        2. During anomalous periods: Protect memory by blending mostly old good memory (90%) 
           with only a little new information (10%)
        3. This prevents corruption from bad data while maintaining model functionality
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_features)
            hidden_state: Hidden state for LSTM cell, or None to initialize
            cell_state: Cell state for LSTM cell, or None to initialize
            use_predictions: Whether to use the model's own predictions as input
            weekly_mask: Optional binary mask for alternating training pattern
            alternating_weeks: Whether to use 1-week alternating pattern for training
            feature_names: Optional list of feature names to find anomaly flag column
        
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
            weekly_mask = torch.zeros(seq_len, device=device)
            for i in range(self.week_steps, seq_len, self.week_steps * 2):
                if i + self.week_steps <= seq_len:
                    weekly_mask[i:i+self.week_steps] = 1
        elif weekly_mask is None:
            weekly_mask = torch.zeros(seq_len, device=device)
        
        # ===============================
        # MEMORY PROTECTION INITIALIZATION
        # ===============================
        # EXPERIMENT 8: Initialize sliding window memory buffer
        if self.use_sliding_window:
            # Reset buffer for new sequence
            self.good_memory_buffer = []
            # Add initial cell state as first good memory
            self.good_memory_buffer.append(cell_state.clone())
        else:
            # Original approach: Store a backup of "clean" memory state
            last_good_cell_state = cell_state.clone()
        
        # memory_decay = 0.3 means: during anomalies, accept only 30% new info, keep 70% old good memory
        memory_decay = 0.3
        
        # Find which column contains the anomaly flags (0=normal, 1=anomaly)
        if feature_names is not None and self.anomaly_flag_idx is None:
            try:
                self.anomaly_flag_idx = feature_names.index(self.anomaly_flag_column)
            except ValueError:
                self.anomaly_flag_idx = -1  # Default to last feature if not found
        
        # ===============================
        #      MAIN PROCESSING LOOP     #
        # ===============================
        for t in range(seq_len):
            # -----------------------------------------------
            # STEP 1: Prepare input (alternating training logic)
            # -----------------------------------------------
            use_original = (weekly_mask[t].item() == 0) or not use_predictions
            
            if not use_original and t > 0:
                # Use model's own prediction as input (with flag=1)
                binary_flag = torch.ones(batch_size, 1, device=device)
                pred_input = x[:, t, :].clone()
                pred_input[:, 0] = outputs[:, t-1, 0]  # Replace with previous prediction
                current_input = torch.cat([pred_input, binary_flag], dim=1)
            else:
                # Use original data as input (with flag=0)
                binary_flag = torch.zeros(batch_size, 1, device=device)
                current_input = torch.cat([x[:, t, :], binary_flag], dim=1)
            
            # -----------------------------------------------
            # STEP 2: Check if current timestep is anomalous
            # -----------------------------------------------
            if self.anomaly_flag_idx is not None and self.anomaly_flag_idx >= 0:
                is_anomalous = x[:, t, self.anomaly_flag_idx].bool()
            else:
                is_anomalous = x[:, t, -1].bool()  # Fallback to last feature
            
            # -----------------------------------------------
            # STEP 3: Normal LSTM processing
            # -----------------------------------------------
            new_hidden_state, new_cell_state = self.lstm_cell(current_input, (hidden_state, cell_state))
            
            # -----------------------------------------------
            # STEP 4: MEMORY PROTECTION DECISION
            # -----------------------------------------------
            if is_anomalous.any():
                # ANOMALY DETECTED: Use sliding window memory protection
                if self.use_sliding_window and len(self.good_memory_buffer) > 0:
                    # Calculate weighted average of recent good states
                    # Weight recent states more heavily (exponential decay)
                    weights = torch.softmax(torch.arange(len(self.good_memory_buffer), device=device).float(), dim=0)
                    
                    # Weighted combination of good memory states
                    weighted_good_state = torch.zeros_like(cell_state)
                    for i, good_state in enumerate(self.good_memory_buffer):
                        weighted_good_state += weights[i] * good_state
                    
                    # Blend with new state: 70% weighted good memory + 30% new info
                    new_cell_blend = weighted_good_state * (1 - memory_decay) + new_cell_state * memory_decay
                    
                    # Add exponential smoothing for gradual transitions
                    smoothing_factor = 0.1
                    cell_state = cell_state * (1 - smoothing_factor) + new_cell_blend * smoothing_factor
                else:
                    # Fallback to original method if buffer empty
                    if 'last_good_cell_state' in locals():
                        new_cell_blend = last_good_cell_state * (1 - memory_decay) + new_cell_state * memory_decay
                        smoothing_factor = 0.1
                        cell_state = cell_state * (1 - smoothing_factor) + new_cell_blend * smoothing_factor
                    else:
                        cell_state = new_cell_state
                
                # Update hidden state normally (less critical for long-term memory)
                hidden_state = new_hidden_state
            else:
                # NORMAL OPERATION: Update everything and save as new "good" backup
                hidden_state = new_hidden_state
                cell_state = new_cell_state
                
                if self.use_sliding_window:
                    # Add current good state to sliding window buffer
                    self.good_memory_buffer.append(cell_state.clone())
                    
                    # Maintain buffer size limit
                    if len(self.good_memory_buffer) > self.memory_window_size:
                        self.good_memory_buffer.pop(0)  # Remove oldest
                else:
                    # Original approach: update single backup
                    last_good_cell_state = cell_state.clone()
            
            # -----------------------------------------------
            # STEP 5: Generate prediction
            # -----------------------------------------------
            final_hidden = self.dropout(hidden_state)
            outputs[:, t, :] = self.output_layer(final_hidden)
        
        return outputs, hidden_state, cell_state
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state and cell state."""
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state
    
    def anomaly_aware_loss(self, predictions, targets, anomaly_flags, base_criterion):
        """
        Improved anomaly-aware loss function that helps model learn what data SHOULD 
        look like during anomalous periods.
        
        Components:
        1. Weighted base loss: Reduced weight during anomalies (corrupted targets)
        2. Recovery guidance: Use post-anomaly data to guide predictions
        3. Pattern consistency: Maintain normal rates of change during anomalies
        4. Statistical consistency: Keep predictions within normal bounds
        
        Args:
            predictions: Model predictions [seq_len, 1]
            targets: Ground truth (corrupted during anomalies) [seq_len, 1]
            anomaly_flags: Binary flags (1=anomaly, 0=normal) [seq_len]
            base_criterion: Base loss function
        
        Returns:
            Combined loss encouraging realistic predictions during anomalies
        """
        if not self.use_weighted_loss:
            return base_criterion(predictions, targets)
        
        device = predictions.device
        seq_len = predictions.shape[0]
        
        # ===============================
        # 1. WEIGHTED BASE LOSS
        # ===============================
        pointwise_loss = torch.abs(predictions - targets)
        
        # Check for NaN in basic loss first
        if torch.isnan(pointwise_loss).any():
            return base_criterion(predictions, targets)
        
        # Reduce weight during anomalies (don't trust corrupted targets)
        weights = torch.where(anomaly_flags.bool(), 
                            torch.tensor(self.anomaly_weight, device=device),  # 0.3 for anomalies
                            torch.tensor(1.0, device=device))                 # 1.0 for normal
        
        weighted_base_loss = torch.mean(weights * pointwise_loss)
        
        # ===============================
        # 2. RECOVERY GUIDANCE
        # ===============================
        recovery_loss = torch.tensor(0.0, device=device)
        
        if seq_len > 12:  # Only if sequence is long enough
            try:
                # Find where anomalies end (transition from 1 to 0)
                anomaly_diff = torch.diff(anomaly_flags.float())
                anomaly_ends = torch.where(anomaly_diff < 0)[0] + 1
                
                if len(anomaly_ends) > 0:
                    recovery_window = min(12, seq_len - anomaly_ends.max().item())
                    recovery_losses = []
                    
                    for end_idx in anomaly_ends:
                        if end_idx + recovery_window <= seq_len:
                            recovery_target = targets[end_idx:end_idx + recovery_window]
                            recovery_pred = predictions[end_idx:end_idx + recovery_window]
                            
                            # Check for valid data
                            if not torch.isnan(recovery_target).any() and not torch.isnan(recovery_pred).any():
                                recovery_losses.append(torch.mean(torch.abs(recovery_pred - recovery_target)))
                    
                    if recovery_losses:
                        recovery_loss = torch.mean(torch.stack(recovery_losses))
            except:
                recovery_loss = torch.tensor(0.0, device=device)
        
        # ===============================
        # 3. PATTERN CONSISTENCY
        # ===============================
        pattern_loss = torch.tensor(0.0, device=device)
        
        if seq_len > 2:  # Need at least 3 points for meaningful differences
            try:
                # Calculate normal patterns (rate of change during normal periods)
                normal_mask = ~anomaly_flags.bool()[:-1]  # Exclude last element for diff
                
                if normal_mask.sum() > 1:  # Need at least 2 normal differences
                    normal_diffs = torch.diff(targets.squeeze())[normal_mask]
                    
                    # Check for valid normal differences
                    if not torch.isnan(normal_diffs).any() and len(normal_diffs) > 0:
                        normal_mean_change = torch.mean(normal_diffs)
                        normal_std_change = torch.std(normal_diffs)
                        
                        # Only proceed if we have meaningful variance
                        if normal_std_change > 1e-6:
                            # During anomalies, encourage predictions to follow normal patterns
                            anomaly_mask = anomaly_flags.bool()[:-1]
                            
                            if anomaly_mask.sum() > 0:  # Have anomalous differences
                                pred_diffs = torch.diff(predictions.squeeze())[anomaly_mask]
                                
                                if not torch.isnan(pred_diffs).any() and len(pred_diffs) > 0:
                                    # Penalize deviations beyond 2 standard deviations from normal
                                    deviation = torch.abs(pred_diffs - normal_mean_change)
                                    pattern_loss = torch.mean(torch.relu(deviation - 2 * normal_std_change))
            except:
                pattern_loss = torch.tensor(0.0, device=device)
        
        # ===============================
        # 4. STATISTICAL CONSISTENCY
        # ===============================
        stats_loss = torch.tensor(0.0, device=device)
        
        try:
            normal_data = targets[~anomaly_flags.bool()]
            anomaly_predictions = predictions[anomaly_flags.bool()]
            
            # Need sufficient data for meaningful statistics
            if len(normal_data) > 2 and len(anomaly_predictions) > 1:
                # Check for valid data
                if not torch.isnan(normal_data).any() and not torch.isnan(anomaly_predictions).any():
                    normal_mean = torch.mean(normal_data)
                    normal_std = torch.std(normal_data)
                    
                    pred_mean = torch.mean(anomaly_predictions)
                    pred_std = torch.std(anomaly_predictions)
                    
                    # Only proceed if we have meaningful variance
                    if normal_std > 1e-6:
                        # Normalize differences by normal std to make scale-invariant
                        mean_diff = torch.abs(pred_mean - normal_mean) / (normal_std + 1e-6)
                        std_diff = torch.abs(pred_std - normal_std) / (normal_std + 1e-6)
                        
                        stats_loss = mean_diff + std_diff
        except:
            stats_loss = torch.tensor(0.0, device=device)
        
        # ===============================
        # 5. COMBINE ALL COMPONENTS
        # ===============================
        # Check all components for NaN before combining
        if torch.isnan(weighted_base_loss):
            weighted_base_loss = torch.tensor(1.0, device=device)
        if torch.isnan(recovery_loss):
            recovery_loss = torch.tensor(0.0, device=device)
        if torch.isnan(pattern_loss):
            pattern_loss = torch.tensor(0.0, device=device)
        if torch.isnan(stats_loss):
            stats_loss = torch.tensor(0.0, device=device)
        
        total_loss = weighted_base_loss + \
                    0.3 * recovery_loss + \
                    0.3 * pattern_loss + \
                    0.2 * stats_loss
        
        # Final NaN check - fallback to base criterion
        if torch.isnan(total_loss):
            return base_criterion(predictions, targets)
        
        return total_loss 

    def anomaly_aware_loss_simple(self, predictions, targets, anomaly_flags, base_criterion):
        """
        Anomaly-aware loss function with two key components:
        
        1. WEIGHTED LOSS: Pay less attention to errors during anomalous periods
            - Normal periods: weight = 1.0 (learn fully from errors)
            - Anomalous periods: weight = 0.3 (learn less, targets might be wrong)
        
        2. PATTERN PENALTY: Punish the model if it copies anomalous patterns
            - If corrupted data spikes up and model follows → penalty
            - Encourages model to maintain reasonable predictions during anomalies
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets (may be corrupted during anomalous periods)
            anomaly_flags: Binary flags (1 = anomaly, 0 = normal)
            base_criterion: Base loss function (e.g., MSE, MAE)
        
        Returns:
            Combined loss = weighted_base_loss + pattern_penalty
        """
        if not self.use_weighted_loss:
            return base_criterion(predictions, targets)
        
        # ===============================
        # COMPONENT 1: BASIC ERROR CALCULATION
        # ===============================
        pointwise_loss = torch.abs(predictions - targets)  # MAE: |prediction - target|
        
        # ===============================
        # COMPONENT 2: PATTERN ANALYSIS
        # ===============================
        # Calculate how predictions and targets change over time
        pred_diff = torch.diff(predictions, dim=0)    # pred_diff[i] = predictions[i+1] - predictions[i]
        target_diff = torch.diff(targets, dim=0)      # target_diff[i] = targets[i+1] - targets[i]
        
        # Initialize pattern penalty (same size as differences)
        pattern_penalty = torch.zeros_like(pointwise_loss[:-1])  # One shorter due to diff
        anomaly_mask = anomaly_flags[:-1].bool()  # Match diff length
        
        if anomaly_mask.any():
            # ===============================
            # PATTERN PENALTY CALCULATION
            # ===============================
            # During anomalous periods, check: "Are my predictions following the same pattern as corrupted targets?"
            # Example: If corrupted data suddenly spikes up, and model prediction also spikes up → penalty
            # Goal: Model should maintain reasonable predictions, not copy anomalous behavior
            pattern_similarity = torch.abs(pred_diff[anomaly_mask] - target_diff[anomaly_mask])
            pattern_penalty[anomaly_mask] = pattern_similarity
        
        # ===============================
        # COMPONENT 3: WEIGHTED COMBINATION
        # ===============================
        normal_weight = 1.0      # Full weight during normal periods
        anomaly_weight = self.anomaly_weight  # Reduced weight (0.3) during anomalies
        pattern_weight = 0.5     # REDUCED from 2.0 to 0.5 to prevent oscillations
        
        # Create weight mask: anomalous periods get lower weight
        weights = torch.where(anomaly_flags.bool(), 
                            torch.tensor(anomaly_weight, device=predictions.device),  # 0.3 for anomalies
                            torch.tensor(normal_weight, device=predictions.device))   # 1.0 for normal
        
        # ===============================
        # FINAL LOSS CALCULATION
        # ===============================
        weighted_base_loss = torch.mean(weights * pointwise_loss)
        pattern_loss = torch.mean(pattern_penalty) if pattern_penalty.numel() > 0 else torch.tensor(0.0, device=predictions.device)
        
        total_loss = weighted_base_loss + pattern_weight * pattern_loss
        
        return total_loss 