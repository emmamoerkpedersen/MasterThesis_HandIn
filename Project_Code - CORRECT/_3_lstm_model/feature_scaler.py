import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
import torch


class FeatureScaler:
    """
    A class for scaling features and saving/loading scalers.
    This can be used independently or as part of the DataPreprocessor.
    """
    
    def __init__(self, feature_cols, output_features, device=None):
        """
        Initialize the FeatureScaler.
        
        Args:
            feature_cols (list): List of feature column names
            output_features (str): Name of the target/output feature
            device (torch.device, optional): Device to use for tensors. Defaults to None.
        """
        self.feature_cols = feature_cols
        self.output_features = output_features
        self.scalers = {}
        self.is_fitted = False
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    
    def fit(self, features, target):
        """
        Fit the scalers to the data.
        
        Args:
            features (pd.DataFrame): DataFrame containing feature columns
            target (pd.DataFrame): DataFrame containing target column
        """
        # Reset the fitted state
        self.is_fitted = False
        
        # Call scale_data to fit the scalers
        _, _ = self.scale_data(features, target)
        
        # The scalers are now fitted
        self.is_fitted = True
    
    def transform(self, features, target):
        """
        Transform the data using the fitted scalers.
        
        Args:
            features (pd.DataFrame): DataFrame containing feature columns
            target (pd.DataFrame): DataFrame containing target column
            
        Returns:
            tuple: (scaled_features, scaled_target)
        """
        if not self.is_fitted:
            raise ValueError("Scalers have not been fitted. Call fit() first.")
        
        # Call scale_data to transform the data
        return self.scale_data(features, target)
    
    def fit_transform(self, features, target):
        """
        Fit the scalers and transform the data in one step.
        
        Args:
            features (pd.DataFrame): DataFrame containing feature columns
            target (pd.DataFrame): DataFrame containing target column
            
        Returns:
            tuple: (scaled_features, scaled_target)
        """
        # Reset the fitted state
        self.is_fitted = False
        
        # Call scale_data to fit the scalers and transform the data
        return self.scale_data(features, target)
    
    def inverse_transform_target(self, scaled_target):
        """
        Inverse transform the scaled target back to original scale.
        
        Args:
            scaled_target (np.ndarray): Scaled target values
            
        Returns:
            np.ndarray: Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Scalers have not been fitted. Call fit() first.")
        
        # Reshape if needed
        if scaled_target.ndim == 1:
            scaled_target = scaled_target.reshape(-1, 1)
        
        return self.scalers['target'].inverse_transform(scaled_target)
    
    def save_scalers(self, filepath):
        """
        Save the fitted scalers to a file.
        
        Args:
            filepath (str): Path to save the scalers
        """
        if not self.is_fitted:
            raise ValueError("Scalers have not been fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
    
    def load_scalers(self, filepath):
        """
        Load scalers from a file.
        
        Args:
            filepath (str): Path to the saved scalers
        """
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
        
        self.is_fitted = True

    def scale_data(self, features, target):
        """
        Scale features and target using StandardScaler.
        For target variables, NaN values are excluded from scaling calculations.
        """
        if not self.is_fitted:
            # Ensure we have all feature columns
            missing_cols = [col for col in self.feature_cols if col not in features.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
            
            # Create a single scaler for all features
            self.scalers = {
                'features': StandardScaler(),
                'target': StandardScaler()
            }
            
            # Extract features in the correct order
            feature_matrix = features[self.feature_cols].values
            
            # Fit the feature scaler on all features at once
            self.scalers['features'].fit(feature_matrix)
            
            # Fit target scaler - exclude NaN values
            target_array = target.values if isinstance(target, pd.DataFrame) else target
            
            # Create a mask for non-NaN values
            non_nan_mask = ~np.isnan(target_array)
            
            # Fit the scaler only on non-NaN values
            if np.any(non_nan_mask):
                self.scalers['target'].fit(target_array[non_nan_mask].reshape(-1, 1))
            else:
                # If all values are NaN, create a dummy scaler
                self.scalers['target'].fit(np.array([[0]]))
                
            self.is_fitted = True

        # Scale features - ensure we scale all features in the correct order
        feature_matrix = features[self.feature_cols].values
        scaled_features = self.scalers['features'].transform(feature_matrix)

        # Scale target - handle NaN values
        target_array = target.values if isinstance(target, pd.DataFrame) else target
        
        # Create a copy of the target array to avoid modifying the original
        scaled_target = np.full_like(target_array, np.nan)
        
        # Create a mask for non-NaN values
        non_nan_mask = ~np.isnan(target_array)
        
        # Scale only non-NaN values
        if np.any(non_nan_mask):
            scaled_target[non_nan_mask] = self.scalers['target'].transform(
                target_array[non_nan_mask].reshape(-1, 1)
            ).flatten()
        
        # Ensure scaled_target has the same shape as before (1D array)
        if scaled_target.ndim > 1:
            scaled_target = scaled_target.flatten()
        
        return scaled_features, scaled_target 