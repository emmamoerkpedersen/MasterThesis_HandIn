"""
Model utilities for the water level prediction pipeline.
This module contains functions to handle model creation, training, and prediction.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def create_lstm_model(input_size, model_config, model_class):
    """
    Create an LSTM model with the specified configuration.
    
    Args:
        input_size: Number of input features
        model_config: Model configuration dictionary
        model_class: LSTM model class to instantiate
        
    Returns:
        Instantiated LSTM model
    """
    sequence_length = model_config.get('sequence_length', 150)
    print(f"Creating model with sequence_length: {sequence_length}")
    
    model = model_class(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=model_config['hidden_size'],
        output_size=len(model_config['output_features']),
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    return model

def print_model_params(model_config):
    """
    Print model hyperparameters for verification.
    
    Args:
        model_config: Model configuration dictionary
    """
    print("\nVerifying hyperparameters:")
    expected_params = [
        'hidden_size', 'num_layers', 'dropout', 'learning_rate', 
        'batch_size', 'sequence_length'
    ]
    for param in expected_params:
        print(f"  {param}: {model_config.get(param)}")

def train_model(trainer, train_data, val_data, model_config):
    """
    Train an LSTM model using the specified trainer and data.
    
    Args:
        trainer: LSTM trainer instance
        train_data: Training data DataFrame
        val_data: Validation data DataFrame
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (training history, validation predictions, validation targets)
    """
    print("\nTraining model...")
    history, val_predictions, val_targets = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        patience=model_config['patience']
    )
    
    print("\nTraining Results:")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    return history, val_predictions, val_targets

def save_model(model, path='final_model.pth'):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        path: Path to save the model
        
    Returns:
        Path where the model was saved
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path

def process_val_predictions(val_predictions, preprocessor, validation_data, model_config=None):
    """
    Process validation predictions and align with validation data.
    
    Args:
        val_predictions: Raw validation predictions tensor
        preprocessor: Data preprocessor with inverse transform capability
        validation_data: Validation data DataFrame
        model_config: Model configuration dictionary
        
    Returns:
        DataFrame with processed validation predictions
    """
    from utils.pipeline_utils import prepare_prediction_dataframe
    
    # Convert validation predictions to numpy and reshape
    val_predictions_np = val_predictions.cpu().numpy() if isinstance(val_predictions, torch.Tensor) else val_predictions
    
    # Print debug info about original shape
    print(f"Original validation predictions shape: {val_predictions_np.shape}")
    
    # Handle multidimensional arrays (more than 2D)
    if len(val_predictions_np.shape) > 2:
        print(f"Flattening {len(val_predictions_np.shape)}-dimensional array")
        # For 3D or higher, flatten all but the first dimension first
        flattened_shape = (val_predictions_np.shape[0], -1)
        val_predictions_np = val_predictions_np.reshape(flattened_shape)
        print(f"Reshaped to: {val_predictions_np.shape}")
    
    # Now flatten to 1D
    predictions_flattened = val_predictions_np.flatten()
    print(f"Final flattened shape: {predictions_flattened.shape}")
    
    # Inverse transform the predictions
    predictions_reshaped = predictions_flattened.reshape(-1, 1)
    predictions_original = preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped)
    predictions_flattened = predictions_original.flatten()  # Ensure 1D array
    
    # Print data lengths for debugging
    print(f"Validation predictions length: {len(predictions_flattened)}")
    print(f"Validation data length: {len(validation_data)}")
    
    # Create DataFrame with aligned predictions
    return prepare_prediction_dataframe(
        predictions_flattened,
        validation_data.index,
        len(validation_data)
    )

def process_test_predictions(test_predictions, test_data, model_config=None):
    """
    Process test predictions and align with test data.
    
    Args:
        test_predictions: Raw test predictions array
        test_data: Test data DataFrame
        model_config: Model configuration dictionary
        
    Returns:
        DataFrame with processed test predictions
    """
    from utils.pipeline_utils import prepare_prediction_dataframe
    
    # Ensure test_predictions is a numpy array
    if isinstance(test_predictions, torch.Tensor):
        test_predictions = test_predictions.cpu().numpy()
    else:
        test_predictions = np.array(test_predictions)
    
    # Print debug info about original shape
    print(f"Original test predictions shape: {test_predictions.shape}")
    
    # Handle multidimensional arrays (more than 2D)
    if len(test_predictions.shape) > 2:
        print(f"Flattening {len(test_predictions.shape)}-dimensional array")
        # For 3D or higher, flatten all but the first dimension first
        flattened_shape = (test_predictions.shape[0], -1)
        test_predictions = test_predictions.reshape(flattened_shape)
        print(f"Reshaped to: {test_predictions.shape}")
    
    # Now flatten to 1D
    test_predictions = test_predictions.flatten()
    print(f"Final flattened shape: {test_predictions.shape}")
    
    # Print data lengths for debugging
    print(f"Test predictions length: {len(test_predictions)}")
    print(f"Test data length: {len(test_data)}")
    
    # Convert test predictions to proper format for DataFrame creation
    return prepare_prediction_dataframe(
        test_predictions, 
        test_data.index, 
        len(test_data)
    ) 