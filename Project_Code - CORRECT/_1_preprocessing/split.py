from pathlib import Path
import sys
project_root = Path(__file__).parent.parent  # Go up one level from current file
sys.path.append(str(project_root))

import pandas as pd
from typing import Dict
from datetime import timedelta
from config import LSTM_CONFIG  # Import the config

def split_data_with_combined_windows(preprocessed_data: dict,
                                   val_years: int = 1,
                                   test_years: int = 2) -> dict:
    """
    Split data into training, validation, and test sets based on years.
    Training data will be all remaining data after validation and test periods.
    
    Args:
        preprocessed_data: Dictionary containing station data
        val_years: Number of years for validation (default 1)
        test_years: Number of years for test (default 2)
    
    Returns:
        Dictionary with structure:
        {
            'train': {station_id: {feature: pd.Series}},
            'validation': {station_id: {feature: pd.Series}},
            'test': {station_id: {feature: pd.Series}}
        }
    """
    result = {
        'train': {},
        'validation': {},
        'test': {}
    }
    
    # Get feature columns and target feature from config
    feature_cols = LSTM_CONFIG.get('feature_cols', [])
    target_feature = LSTM_CONFIG.get('output_features', ['vst_raw'])[0]
    all_features = list(set(feature_cols + [target_feature]))
    
    for station_id, station_data in preprocessed_data.items():
        result['train'][station_id] = {}
        result['validation'][station_id] = {}
        result['test'][station_id] = {}
        
        # Get the full date range from the target feature
        if target_feature not in station_data:
            print(f"Warning: Target feature '{target_feature}' not found for station {station_id}")
            continue
            
        target_data = station_data[target_feature]
        if isinstance(target_data, pd.DataFrame):
            target_data = target_data.iloc[:, 0]
        
        # Calculate split dates
        end_date = target_data.index.max()
        test_start = end_date - pd.DateOffset(years=test_years)
        val_start = test_start - pd.DateOffset(years=val_years)
        
        print(f"\nSplitting data for station {station_id}:")
        print(f"Training period: up to {val_start}")
        print(f"Validation period: {val_start} to {test_start}")
        print(f"Testing period: {test_start} to {end_date}")
        
        # Split each feature
        for feature in all_features:
            if feature in station_data:
                data = station_data[feature]
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
                
                # Split the data
                train_data = data[data.index < val_start].copy()
                val_data = data[(data.index >= val_start) & (data.index < test_start)].copy()
                test_data = data[data.index >= test_start].copy()
                
                # Resample to 15-minute frequency and handle missing values
                for split_data in [train_data, val_data, test_data]:
                    split_data = split_data.resample('15T').interpolate(method='time').ffill().bfill()
                
                # Store the processed data
                result['train'][station_id][feature] = train_data
                result['validation'][station_id][feature] = val_data
                result['test'][station_id][feature] = test_data
                
                # Print data sizes
                print(f"\n{feature}:")
                print(f"  Training: {len(train_data)} points")
                print(f"  Validation: {len(val_data)} points")
                print(f"  Test: {len(test_data)} points")
            else:
                print(f"Warning: Feature '{feature}' not found for station {station_id}")
    
    return result
