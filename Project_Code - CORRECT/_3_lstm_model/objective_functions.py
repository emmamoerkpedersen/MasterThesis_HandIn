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


@register_objective('mae_loss')
def mae_loss(outputs, targets):
    """
    Mean Absolute Error loss function.
    """
    return nn.functional.l1_loss(outputs, targets)


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