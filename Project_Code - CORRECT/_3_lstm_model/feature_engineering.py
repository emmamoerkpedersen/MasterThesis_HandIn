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

    def add_lagged_features(self, data, target_col, lags=None):
        """Add lagged features and derived features to the dataset."""
        if lags is None:
            lags = [1, 2, 4, 8, 12, 24]  # Default lags in hours
        
        df = data.copy()
        
        # Convert lags from hours to timesteps (4 timesteps per hour for 15-min data)
        timesteps = [lag * 4 for lag in lags]
        
        # Add lagged features
        for lag in timesteps:
            lag_name = f'water_level_lag_{lag//4}h'
            df[lag_name] = df[target_col].shift(lag)
            print(f"Added lag feature: {lag_name}")  # Debug print
            self.feature_cols.append(lag_name)
        
        # Add moving averages at different windows with more intermediate values
        ma_windows = [12, 24, 48, 72, 96, 144, 192, 384]  # 3h, 6h, 12h, 18h, 24h, 36h, 48h, 96h
        for window in ma_windows:
            ma_name = f'water_level_ma_{window//4}h'
            df[ma_name] = df[target_col].rolling(window=window, min_periods=1).mean()
            print(f"Added MA feature: {ma_name}")  # Debug print
            self.feature_cols.append(ma_name)
        
        # Add more granular rate of change features
        # Very short-term changes
        df['water_level_roc_15min'] = df[target_col].diff()
        df['water_level_roc_30min'] = df[target_col].diff(2)
        
        # Short to medium-term changes
        df['water_level_roc_1h'] = df[target_col].diff(4)
        df['water_level_roc_2h'] = df[target_col].diff(8)
        df['water_level_roc_4h'] = df[target_col].diff(16)
        
        # Longer-term changes
        df['water_level_roc_6h'] = df[target_col].diff(24)
        df['water_level_roc_12h'] = df[target_col].diff(48)
        
        # Rate of change of moving averages to capture trend changes
        df['water_level_ma_roc_6h'] = df[f'water_level_ma_6h'].diff()   # 6h MA change
        df['water_level_ma_roc_24h'] = df[f'water_level_ma_24h'].diff() # 24h MA change
        
        # Add acceleration (change in rate of change)
        df['water_level_acc_15min'] = df['water_level_roc_15min'].diff()
        df['water_level_acc_1h'] = df['water_level_roc_1h'].diff()
        
        # Add these new features to feature_cols
        roc_features = [
            'water_level_roc_15min', 'water_level_roc_30min', 
            'water_level_roc_1h', 'water_level_roc_2h', 'water_level_roc_4h',
            'water_level_roc_6h', 'water_level_roc_12h',
            'water_level_ma_roc_6h', 'water_level_ma_roc_24h',
            'water_level_acc_15min', 'water_level_acc_1h'
        ]
        print(f"Added ROC features: {roc_features}")  # Debug print
        self.feature_cols.extend(roc_features)
        
        # Forward fill any NaN values created by lagging/differencing
        df = df.fillna(method='ffill')
        # Backward fill any remaining NaN values at the start
        df = df.fillna(method='bfill')
        
        print(f"Total features after adding all: {len(self.feature_cols)}")  # Debug print
        return df
