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
        """
        # Handle NaN values in target
        target_filled = target.copy()
        target_filled = target_filled.fillna(target_filled.mean())
        
        if not self.is_fitted:
            self.scalers = {
                'features': {col: StandardScaler() for col in self.feature_cols},
                'target': StandardScaler()
            }
            
            # Fit each feature scaler
            for col in self.feature_cols:
                # Handle NaN values in features
                feature_data = features[[col]].fillna(features[col].mean())
                self.scalers['features'][col].fit(feature_data)
            
            # Fit target scaler
            self.scalers['target'].fit(target_filled)
            self.is_fitted = True

        # Scale features
        scaled_features_list = []
        for col in self.feature_cols:
            # Handle NaN values in features
            feature_data = features[[col]].fillna(features[col].mean())
            scaled_feature = self.scalers['features'][col].transform(feature_data)
            scaled_features_list.append(scaled_feature)
        
        scaled_features = np.hstack(scaled_features_list)

        # Scale target
        scaled_target = self.scalers['target'].transform(target_filled).flatten()
        
        return scaled_features, scaled_target 