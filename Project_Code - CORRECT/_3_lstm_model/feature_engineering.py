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
        data.loc[:, 'rainfall_30day'] = rainfall_fixed.rolling(window=30, min_periods=1).sum()
        data.loc[:, 'rainfall_180day'] = rainfall_fixed.rolling(window=180, min_periods=1).sum()
        data.loc[:, 'rainfall_365day'] = rainfall_fixed.rolling(window=365, min_periods=1).sum()
        
        # Calculate cumulative rainfall for feature stations as well
        data.loc[:, 'feature1_rain_30day'] = feature_1_rain.rolling(window=30, min_periods=1).sum()
        data.loc[:, 'feature2_rain_30day'] = feature_2_rain.rolling(window=30, min_periods=1).sum()
        data.loc[:, 'feature1_rain_180day'] = feature_1_rain.rolling(window=180, min_periods=1).sum()
        data.loc[:, 'feature2_rain_180day'] = feature_2_rain.rolling(window=180, min_periods=1).sum()
        data.loc[:, 'feature1_rain_365day'] = feature_1_rain.rolling(window=365, min_periods=1).sum()
        data.loc[:, 'feature2_rain_365day'] = feature_2_rain.rolling(window=365, min_periods=1).sum()

        # Fill potential NaN values created by rolling operations with forward fill then backward fill
        cumulative_cols = [
            'rainfall_30day', 'rainfall_180day', 'rainfall_365day',
            'feature1_rain_30day', 'feature2_rain_30day', 'feature1_rain_180day', 'feature2_rain_180day', 'feature1_rain_365day', 'feature2_rain_365day'
        ]
        
        for col in cumulative_cols:
            data.loc[:, col] = data[col].ffill().bfill()
            
            # Add new columns to feature_cols list if they're not already there
            if col not in self.feature_cols:
                self.feature_cols.append(col)
        
        return data
    