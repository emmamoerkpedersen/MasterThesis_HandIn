"""
Feature Engineering Module for Water Level Forecasting

This module provides functionality for creating derived features from raw water level data.
The system is designed to be modular and extensible, allowing for easy addition of new feature types.

Key components:
- FeatureEngineer class: Main class for feature engineering
- add_lagged_features: Adds lag features (values from previous timesteps)
- add_custom_features: Extensible system for adding custom feature types

Example usage for extending with custom features:
```python
# Define custom feature specifications
custom_features = [
    # Add a moving average feature for 24-hour window
    {
        'type': 'ma',
        'params': {'hours': 24}
    },
    # Add a rate of change feature for 12-hour period
    {
        'type': 'roc',
        'params': {'hours': 12}
    },
    # Example of a custom feature type (would need implementation in add_custom_features)
    {
        'type': 'wavelet', 
        'params': {'level': 3, 'wavelet': 'db4'}
    }
]

# Add these features to your configuration
config['custom_features'] = custom_features

# When you want to add a new feature type, extend the add_custom_features method with a new case:
# elif feature_type == 'new_type':
#     # Code to generate the new feature
```

When adding new feature types:
1. Add a new case to the add_custom_features method
2. Make sure to consistently name features and update feature_cols
3. Ensure features are properly handled during error injection
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FeatureEngineer:
    def __init__(self, config):
        # Initialize feature columns list with base features
        self.feature_cols = config['feature_cols'].copy()
        
        # Add feature station columns dynamically
        for station in config['feature_stations']:
            for feature in station['features']:
                feature_name = f"feature_station_{station['station_id']}_{feature}"
                self.feature_cols.append(feature_name)
                
        self.output_features = config['output_features'][0]
    
    def _add_time_features(self, data):
            """
            Add time-based features to better capture temporal patterns.
            Uses sin/cos encoding for cyclical features (month, day of year).
            """
            # Skip if the data already has time features
            if 'month_sin' in data.columns:
                return data
                
            # Create a copy of the data to avoid SettingWithCopyWarning
            data = data.copy()
                
            # Extract datetime components
            data.loc[:, 'month'] = data.index.month
            data.loc[:, 'day'] = data.index.day
            data.loc[:, 'day_of_year'] = data.index.dayofyear
        
          
            # Create cyclical features for month (period = 12)
            data.loc[:, 'month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data.loc[:, 'month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Create cyclical features for day of year (period = 365.25)
            data.loc[:, 'day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
            data.loc[:, 'day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
            
            
            # Add these features to feature_cols (check to avoid duplicates)
            time_features = ['month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
            for feature in time_features:
                if feature not in self.feature_cols:
                    self.feature_cols.append(feature)
            
            # Remove the raw time features that we don't want to use directly
            data = data.drop(['month', 'day', 'day_of_year'], axis=1, errors='ignore')
            
            return data

    def _add_cumulative_features(self, data):
        """
        Add cumulative rainfall features to better capture long-term patterns.
        
        NaN handling:
        - Rainfall: Convert -1 (missing) to 0 before calculating cumulative sums
        - Feature station rainfall: Convert -1 (missing) to 0 before calculating cumulative sums
        - Cumulative features: Forward fill, backward fill after calculation
        
        Features added:
        - 1-month (30-day) cumulative rainfall
        - 3-month (90-day) cumulative rainfall
        - 6-month (180-day) cumulative rainfall
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            DataFrame with added cumulative features
        """
        # Create a copy of the data to avoid SettingWithCopyWarning
        data = data.copy()
        
        # First, convert rainfall -1 values (missing) to 0 for cumulative calculations
        rainfall_fixed = data['rainfall'].copy()
        rainfall_fixed[rainfall_fixed == -1] = 0
        
        # Similarly for feature stations
        feature_1_rain = data['feature_station_21006845_rainfall'].copy()
        feature_1_rain[feature_1_rain == -1] = 0
        
        feature_2_rain = data['feature_station_21006847_rainfall'].copy()
        feature_2_rain[feature_2_rain == -1] = 0
        
        # Calculate cumulative rainfall for different windows
        # Using rolling windows with different sizes
        # Short windows
        data.loc[:, 'rainfall_1hour'] = rainfall_fixed.rolling(window=1*4, min_periods=1).sum()
        data.loc[:, 'rainfall_7hour'] = rainfall_fixed.rolling(window=7*4, min_periods=1).sum()
        data.loc[:, 'rainfall_48hour'] = rainfall_fixed.rolling(window=48*4, min_periods=1).sum()
        data.loc[:, 'rainfall_90hour'] = rainfall_fixed.rolling(window=90*4, min_periods=1).sum()
        # Longer windows
        # data.loc[:, 'rainfall_1month'] = rainfall_fixed.rolling(window=30*24*4, min_periods=1).sum()
        # data.loc[:, 'rainfall_3months'] = rainfall_fixed.rolling(window=90*24*4, min_periods=1).sum()
        # data.loc[:, 'rainfall_6months'] = rainfall_fixed.rolling(window=180*24*4, min_periods=1).sum()
        # data.loc[:, 'rainfall_1year'] = rainfall_fixed.rolling(window=365*24*4, min_periods=1).sum()

        # Calculate cumulative rainfall for feature stations as well
        # Short windows
        data.loc[:, 'feature1_rain_1hour'] = feature_1_rain.rolling(window=1*4, min_periods=1).sum()
        data.loc[:, 'feature2_rain_1hour'] = feature_2_rain.rolling(window=1*4, min_periods=1).sum()
        data.loc[:, 'feature1_rain_7hour'] = feature_1_rain.rolling(window=7*4, min_periods=1).sum()
        data.loc[:, 'feature2_rain_7hour'] = feature_2_rain.rolling(window=7*4, min_periods=1).sum()
        data.loc[:, 'feature1_rain_48hour'] = feature_1_rain.rolling(window=48*4, min_periods=1).sum()
        data.loc[:, 'feature2_rain_48hour'] = feature_2_rain.rolling(window=48*4, min_periods=1).sum()
        data.loc[:, 'feature1_rain_90hour'] = feature_1_rain.rolling(window=90*4, min_periods=1).sum()
        data.loc[:, 'feature2_rain_90hour'] = feature_2_rain.rolling(window=90*4, min_periods=1).sum()
       
        # # Longer windows
        # data.loc[:, 'feature1_rain_1month'] = feature_1_rain.rolling(window=30*24*4, min_periods=1).sum()
        # data.loc[:, 'feature2_rain_1month'] = feature_2_rain.rolling(window=30*24*4, min_periods=1).sum()
        # data.loc[:, 'feature1_rain_3months'] = feature_1_rain.rolling(window=90*24*4, min_periods=1).sum()
        # data.loc[:, 'feature2_rain_3months'] = feature_2_rain.rolling(window=90*24*4, min_periods=1).sum()
        # data.loc[:, 'feature1_rain_6months'] = feature_1_rain.rolling(window=180*24*4, min_periods=1).sum()
        # data.loc[:, 'feature2_rain_6months'] = feature_2_rain.rolling(window=180*24*4, min_periods=1).sum()
        # data.loc[:, 'feature1_rain_1year'] = feature_1_rain.rolling(window=365*24*4, min_periods=1).sum()
        # data.loc[:, 'feature2_rain_1year'] = feature_2_rain.rolling(window=365*24*4, min_periods=1).sum()

        # Fill potential NaN values created by rolling operations with forward fill then backward fill
        cumulative_cols = [
            'rainfall_1hour', 'rainfall_7hour', 'rainfall_48hour', 'rainfall_90hour',
            #'rainfall_1month', 'rainfall_3months', 'rainfall_6months', 'rainfall_1year',
            'feature1_rain_1hour', 'feature2_rain_1hour', 'feature1_rain_7hour', 'feature2_rain_7hour', 'feature1_rain_48hour', 'feature2_rain_48hour', 'feature1_rain_90hour', 'feature2_rain_90hour',
            #'feature1_rain_1month', 'feature2_rain_1month', 'feature1_rain_3months', 'feature2_rain_3months', 'feature1_rain_6months', 'feature2_rain_6months', 'feature1_rain_1year', 'feature2_rain_1year'
        ]
        
        for col in cumulative_cols:
            data.loc[:, col] = data[col].ffill().bfill()
            
            # Add new columns to feature_cols list if they're not already there
            if col not in self.feature_cols:
                self.feature_cols.append(col)
        
        return data

 