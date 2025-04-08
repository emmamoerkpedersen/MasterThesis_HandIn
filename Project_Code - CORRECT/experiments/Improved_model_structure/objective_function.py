

def dynamic_weighted_loss(self, outputs, targets):
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

def handle_dynamic_weighting(self, loss, previous, targets):
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