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

@register_objective('handle_dynamic_weighting')
def handle_dynamic_weighting(loss, previous, targets):
    """
    Apply dynamic weighting to the loss based on the magnitude of change between consecutive targets.
    This helps the model focus more on areas with larger changes in water levels.
    
    Args:
        loss: The base loss tensor
        previous: Tensor of previous target values
        targets: Tensor of current target values
        
    Returns:
        Tensor: Loss with dynamic weighting applied
    """
    # Calculate the dynamic weighting; assign higher weights to larger differences
    dynamic_weighting = torch.abs(targets - previous)
    
    # Scale the dynamic weighting to avoid extreme values
    # Using a sigmoid-like scaling to keep values in a reasonable range
    scaled_weighting = 2.0 * torch.sigmoid(dynamic_weighting) - 1.0
    
    # Multiply the loss by the dynamic weighting
    loss = loss * (1.0 + scaled_weighting)
    
    return loss

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