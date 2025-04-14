import torch
import torch.nn as nn   

# Dictionary to store objective functions
OBJECTIVE_FUNCTIONS = {}

def register_objective(name):
    """
    Decorator to register an objective function.
    
    Args:
        name (str): Name to register the function under
        
    Returns:
        function: Decorator function
    """
    def decorator(func):
        OBJECTIVE_FUNCTIONS[name] = func
        return func
    return decorator

@register_objective('mse_loss')
def mse_loss(outputs, targets):
    """
    Mean Squared Error loss function.
    """
    return nn.functional.mse_loss(outputs, targets)

@register_objective('smoothL1_loss')
def smoothL1_loss(outputs, targets):
    """
    Smooth L1 loss function.
    """
    return nn.functional.smooth_l1_loss(outputs, targets)

@register_objective('dynamic_weighted_loss')
def dynamic_weighted_loss(outputs, targets):
    """
    Loss function that applies dynamic weighting based on the magnitude of target values.
    This helps the model focus more on areas with larger water levels.
    
    Args:
        outputs: Model predictions
        targets: Target values
        
    Returns:
        Tensor: Mean of the dynamically weighted MSE loss
    """
    # Calculate MSE loss
    mse_loss = nn.functional.mse_loss(outputs, targets, reduction='none')
    
    # Calculate the dynamic weighting based on the magnitude of target values
    # Higher weights for larger water levels
    min_val = targets.min()
    max_val = targets.max()
    if max_val > min_val:  # Avoid division by zero
        normalized_targets = (targets - min_val) / (max_val - min_val)
        
        # Create weighting that increases with target magnitude
        # Using a sigmoid-like scaling to keep values in a reasonable range
        scaled_weighting = 2.0 * torch.sigmoid(normalized_targets) - 1.0
        
        # Multiply the loss by the dynamic weighting
        weighted_loss = mse_loss * (1.0 + scaled_weighting)
    else:
        weighted_loss = mse_loss
    
    return weighted_loss.mean()


@register_objective('peak_weighted_loss')
def peak_weighted_loss(outputs, targets, peak_weight=1.5):
    """
    Custom loss function that gives higher weight to errors during peak water levels
    and mid-range values where the model tends to struggle.
    """
    # Calculate normal MSE
    mse_loss = nn.functional.mse_loss(outputs, targets, reduction='none')
    
    # Create weights based on target values
    min_val = targets.min()
    max_val = targets.max()
    if max_val > min_val:  # Avoid division by zero
        normalized_targets = (targets - min_val) / (max_val - min_val)
        
        # Create tailored weighting function specifically for the water level data pattern
        # Higher weights for both peaks (>0.7) and troughs (<0.3)
        peak_weight = torch.pow(normalized_targets, 2) * peak_weight
        
        # Targeted boost for mid-range values (0.3-0.7) 
        # Using a Gaussian with max at 0.5 (mid-range)
        mid_boost = 2.0 * torch.exp(-40.0 * torch.pow(normalized_targets - 0.5, 2))
        
        # Combined weights with higher mid-range emphasis
        weights = 1.0 + peak_weight + mid_boost
    else:
        weights = torch.ones_like(targets)
    
    # Apply weights to the MSE loss
    weighted_loss = mse_loss * weights
    
    # Return mean of weighted loss
    return weighted_loss.mean()

@register_objective('smoothL1_dynamic_weighted_loss')
def smoothL1_dynamic_weighted_loss(outputs, targets, beta=1.0, weight_factor=1.5):
    """
    Combined loss function that uses smoothL1 loss with dynamic weighting.
    This provides the robustness of smoothL1 loss against outliers while
    still giving higher weights to peaks and troughs.
    
    Args:
        outputs: Model predictions
        targets: Target values
        beta: smoothL1 beta parameter (default: 1.0)
        weight_factor: Controls the strength of dynamic weighting (default: 1.5)
        
    Returns:
        Tensor: Mean of the dynamically weighted smoothL1 loss
    """
    # Calculate smoothL1 loss with reduction='none' to keep per-element losses
    smooth_l1 = nn.functional.smooth_l1_loss(outputs, targets, reduction='none', beta=beta)
    
    # Calculate the dynamic weighting based on the magnitude of target values
    min_val = targets.min()
    max_val = targets.max()
    
    if max_val > min_val:  # Avoid division by zero
        normalized_targets = (targets - min_val) / (max_val - min_val)
        
        # Create weighting that increases with target magnitude
        # Using a sigmoid-like scaling to keep values in a reasonable range
        scaled_weighting = weight_factor * torch.sigmoid(normalized_targets * 2 - 1.0)
        
        # Enhance weights specifically for peaks (upper quartile)
        peak_mask = normalized_targets > 0.75
        scaled_weighting = torch.where(peak_mask, scaled_weighting * 1.5, scaled_weighting)
        
        # Enhance weights for troughs (lower quartile) as well
        trough_mask = normalized_targets < 0.25
        scaled_weighting = torch.where(trough_mask, scaled_weighting * 1.3, scaled_weighting)
        
        # Multiply the loss by the dynamic weighting
        weighted_loss = smooth_l1 * (1.0 + scaled_weighting)
    else:
        weighted_loss = smooth_l1
    
    return weighted_loss.mean()

@register_objective('adaptive_smoothL1_loss')
def adaptive_smoothL1_loss(outputs, targets):
    """
    SmoothL1 loss with adaptive beta based on target magnitude.
    Uses different sensitivity for peak vs normal values to improve peak predictions.
    
    Args:
        outputs: Model predictions
        targets: Target values
        
    Returns:
        Tensor: Weighted mean of the adaptive smoothL1 loss
    """
    # Calculate target percentiles for adaptive thresholds
    peak_threshold = torch.quantile(targets, 0.9)  # 90th percentile
    
    # Use smaller beta for peaks (more L1-like) and larger beta for normal values (more MSE-like)
    beta_peaks = 0.2  # More sensitive to peaks
    beta_normal = 1.0  # Standard smoothing
    
    # Create peak mask
    peak_mask = targets > peak_threshold
    
    # Initialize combined loss
    combined_loss = torch.zeros_like(targets)
    
    # Calculate losses separately for peaks and normal values
    if peak_mask.any():
        combined_loss[peak_mask] = nn.functional.smooth_l1_loss(
            outputs[peak_mask], 
            targets[peak_mask], 
            beta=beta_peaks,
            reduction='none'
        ) * 2.0  # Higher weight for peaks
    
    if (~peak_mask).any():
        combined_loss[~peak_mask] = nn.functional.smooth_l1_loss(
            outputs[~peak_mask], 
            targets[~peak_mask], 
            beta=beta_normal,
            reduction='none'
        )
    
    return combined_loss.mean()

@register_objective('momentum_smoothL1_loss')
def momentum_smoothL1_loss(outputs, targets):
    """
    SmoothL1 loss that considers the momentum of changes to better capture rapid changes
    in water levels. This helps reduce prediction lag during sudden level changes.
    
    Args:
        outputs: Model predictions
        targets: Target values
        
    Returns:
        Tensor: Combined loss incorporating both value and change-rate errors
    """
    # Basic smoothL1 loss
    base_loss = nn.functional.smooth_l1_loss(outputs, targets, reduction='none')
    
    # Ensure tensors are properly shaped for diff operation
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(0)
        outputs = outputs.unsqueeze(0)
    
    # Calculate momentum terms (rate of change)
    target_diff = torch.diff(targets, n=1, dim=-1)
    output_diff = torch.diff(outputs, n=1, dim=-1)
    
    # Pad the diff tensors to match original size
    target_diff = torch.cat([target_diff, target_diff[:,-1:]], dim=-1)
    output_diff = torch.cat([output_diff, output_diff[:,-1:]], dim=-1)
    
    # Momentum loss (helps with predicting changes)
    momentum_loss = nn.functional.smooth_l1_loss(
        output_diff,
        target_diff,
        reduction='none'
    )
    
    # Calculate adaptive weights based on rate of change
    change_magnitude = torch.abs(target_diff)
    max_change = torch.max(change_magnitude)
    if max_change > 0:  # Avoid division by zero
        momentum_weight = change_magnitude / max_change
        momentum_weight = torch.clamp(momentum_weight, min=0.5, max=2.0)  # Limit weight range
    else:
        momentum_weight = torch.ones_like(change_magnitude)
    
    # Combine losses with momentum getting higher weight during rapid changes
    combined_loss = base_loss + momentum_weight * momentum_loss
    
    return combined_loss.mean()

@register_objective('peak_focused_loss')
def peak_focused_loss(outputs, targets):
    """
    Highly specialized loss function focused on accurate peak prediction.
    Uses extremely aggressive weighting for peaks and rapid changes.
    
    Args:
        outputs: Model predictions
        targets: Target values
        
    Returns:
        Tensor: Mean of the combined loss with strong peak emphasis
    """
    # Calculate the basic smooth L1 loss
    basic_loss = nn.functional.smooth_l1_loss(outputs, targets, reduction='none')
    
    # Identify peaks using a higher threshold (top 20%)
    min_val = targets.min()
    max_val = targets.max()
    if max_val <= min_val:  # Handle edge case
        return basic_loss.mean()
    
    # Normalize targets to 0-1 range
    normalized_targets = (targets - min_val) / (max_val - min_val)
    
    # Create peak mask for top 20% of values
    peak_threshold = 0.8
    peak_mask = normalized_targets > peak_threshold
    
    # Calculate the rate of change in target values
    if len(targets.shape) == 1:
        targets_temp = targets.unsqueeze(0)
    else:
        targets_temp = targets
        
    target_diffs = torch.cat([
        torch.zeros_like(targets_temp[:,:1]), 
        torch.diff(targets_temp, dim=1)
    ], dim=1)
    
    # Identify rapid changes using the rate of change
    change_threshold = torch.quantile(torch.abs(target_diffs), 0.9)  # Top 10% of changes
    rapid_change_mask = torch.abs(target_diffs) > change_threshold
    
    # Create the final weight matrix based on different conditions:
    weights = torch.ones_like(targets)
    
    # Extreme weight for peaks (5x)
    weights = torch.where(peak_mask, 5.0 * torch.ones_like(weights), weights)
    
    # High weight for rapid changes (3x)
    weights = torch.where(rapid_change_mask, 3.0 * torch.ones_like(weights), weights)
    
    # Enhanced weights for "approach to peak" (values in upper half)
    approach_mask = (normalized_targets > 0.5) & (~peak_mask)
    weights = torch.where(approach_mask, 2.0 * torch.ones_like(weights), weights)
    
    # Apply weights to the basic loss
    weighted_loss = basic_loss * weights
    
    return weighted_loss.mean()

def get_objective_function(name):
    """
    Get an objective function by name.
    
    Args:
        name (str): Name of the objective function
        
    Returns:
        function: The objective function
        
    Raises:
        ValueError: If the objective function is not found
    """
    if name not in OBJECTIVE_FUNCTIONS:
        raise ValueError(f"Objective function '{name}' not found. Available functions: {list(OBJECTIVE_FUNCTIONS.keys())}")
    
    return OBJECTIVE_FUNCTIONS[name]