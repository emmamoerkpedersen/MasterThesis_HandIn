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
        
        # Split 
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
            
    
    # Add debug print to check NaN values
    print("\nData Statistics after splitting:")
    for split_name in ['train', 'validation', 'test']:
        print(f"\n{split_name.upper()} SET:")
        for feature in all_features:
            for station_id in result[split_name]:
                data = result[split_name][station_id][feature]
                nan_count = data.isna().sum()
                total_points = len(data)
                print(f"{feature}:")
                print(f"  NaN values: {nan_count} ({(nan_count/total_points)*100:.2f}% of {total_points} points)")
                print(f"  Unique values: {data.nunique()}")
                print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    
    return result
