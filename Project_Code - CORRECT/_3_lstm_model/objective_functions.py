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