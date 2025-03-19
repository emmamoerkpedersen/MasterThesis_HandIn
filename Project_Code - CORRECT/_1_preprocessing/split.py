import pandas as pd
from typing import Dict
from datetime import timedelta
from config import LSTM_CONFIG  # Import the config

def split_data_with_combined_windows(preprocessed_data: dict,
                                   val_size: float = 0.2,
                                   test_size: float = 0.2) -> dict:
    """
    Split data into training, validation, and test sets based on percentages.
    Training: 60%, Validation: 20%, Test: 20% (by default)
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
        # Initialize result dictionaries
        for split in result:
            result[split][station_id] = {}
            
        # Get target data and calculate split points
        target_data = station_data.get(target_feature)
        if target_data is None:
            print(f"Warning: Target feature '{target_feature}' not found for station {station_id}")
            continue
            
        if isinstance(target_data, pd.DataFrame):
            target_data = target_data.iloc[:, 0]
        
        # Calculate split points
        dates = sorted(target_data.index)
        val_start = dates[int(len(dates) * (1 - test_size - val_size))]
        test_start = dates[int(len(dates) * (1 - test_size))]
        
        # Split and resample each feature
        for feature in all_features:
            if feature not in station_data:
                print(f"Warning: Feature '{feature}' not found for station {station_id}")
                continue
                
            data = station_data[feature]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
                
            # Split data
            splits = {
                'train': data[data.index < val_start],
                'validation': data[(data.index >= val_start) & (data.index < test_start)],
                'test': data[data.index >= test_start]
            }
            
            # Resample and fill based on feature type
            for split_name, split_data in splits.items():
                resampled = split_data.resample('15min')
                if feature in ['rainfall']:
                    # First resample, then replace NaN with -1
                    result[split_name][station_id][feature] = resampled.asfreq().fillna(-1)
                else:  # temperature
                    result[split_name][station_id][feature] = resampled.ffill().bfill()
    
    return result
