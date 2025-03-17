from pathlib import Path
import sys
project_root = Path(__file__).parent.parent  # Go up one level from current file
sys.path.append(str(project_root))

import pandas as pd
from typing import Dict
from datetime import timedelta
from config import LSTM_CONFIG  # Import the config



def split_data(preprocessed_data: dict, 
               train_ratio: float = 0.6,
               validation_ratio: float = 0.2) -> dict:
    """
    Split preprocessed data into training, validation and test sets chronologically.
    Uses features specified in LSTM_CONFIG['feature_cols'].
    
    Args:
        preprocessed_data: Dictionary containing station data
        train_ratio: Proportion of data for training (default 0.6)
        validation_ratio: Proportion of data for validation (default 0.2)
        Note: test_ratio will be 1 - (train_ratio + validation_ratio)
    
    Returns:
        Dictionary with structure:
        {
            'train': {
                'station1': {
                    'vst_raw': pd.DataFrame,
                    'vinge': pd.DataFrame,
                    'rainfall': pd.DataFrame
                },
                'station2': {...}
            },
            'validation': {
                'station1': {
                    'vst_raw': pd.DataFrame,
                    'vinge': pd.DataFrame,
                    'rainfall': pd.DataFrame
                },
                'station2': {...}
            },
            'test': {
                'station1': {
                    'vst_raw': pd.DataFrame,
                    'vinge': pd.DataFrame,
                    'rainfall': pd.DataFrame
                },
                'station2': {...}
            }
        }
    """
    split_data = {
        'train': {},
        'validation': {},
        'test': {}
    }
    
    test_ratio = 1 - (train_ratio + validation_ratio)
    feature_cols = LSTM_CONFIG.get('feature_cols', ['vst_raw'])  # Get features from config
    
    print(f"Splitting data for features: {feature_cols}")
    
    for station_name, station_data in preprocessed_data.items():
        # Initialize empty dictionaries for this station in each split
        split_data['train'][station_name] = {}
        split_data['validation'][station_name] = {}
        split_data['test'][station_name] = {}
        
        # Check if we have the primary feature (usually vst_raw) for determining time range
        primary_feature = feature_cols[0]
        if station_data[primary_feature] is not None:
            primary_data = station_data[primary_feature]
            
            # Ensure we're working with datetime index
            if not isinstance(primary_data.index, pd.DatetimeIndex):
                primary_data.set_index('Date', inplace=True)
            
            # Get the date range (ensure timezone-naive)
            start_date = primary_data.index.min().tz_localize(None)
            end_date = primary_data.index.max().tz_localize(None)
            date_range = end_date - start_date
            
            # Calculate split dates
            train_end = start_date + pd.Timedelta(days=date_range.days * train_ratio)
            val_end = start_date + pd.Timedelta(days=date_range.days * (train_ratio + validation_ratio))
            
            print(f"\nSplitting {station_name} data:")
            print(f"Training period: {start_date.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"Validation period: {train_end.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
            print(f"Testing period: {val_end.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Split each required feature using the same date ranges
            for feature in feature_cols:
                if feature in station_data and station_data[feature] is not None:
                    data = station_data[feature]
                    
                    # Ensure datetime index and convert to timezone-naive
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.set_index('Date', inplace=True)
                    
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Split based on dates
                    train_data = data[data.index <= train_end].copy()
                    val_data = data[(data.index > train_end) & (data.index <= val_end)].copy()
                    test_data = data[data.index > val_end].copy()
                    
                    split_data['train'][station_name][feature] = train_data
                    split_data['validation'][station_name][feature] = val_data
                    split_data['test'][station_name][feature] = test_data
                else:
                    print(f"Warning: Feature '{feature}' not found for station {station_name}")
                    # Initialize empty DataFrames for missing features
                    split_data['train'][station_name][feature] = pd.DataFrame()
                    split_data['validation'][station_name][feature] = pd.DataFrame()
                    split_data['test'][station_name][feature] = pd.DataFrame()
    
    return split_data

def split_data_yearly(preprocessed_data: dict, window_size: int = 365) -> dict:
    """
    Splits the preprocessed data into yearly windows for unsupervised anomaly detection.
    Uses features specified in LSTM_CONFIG['feature_cols'].
    """
    split_result = {'windows': {}}
    feature_cols = LSTM_CONFIG.get('feature_cols') # Get features from config
    
    print(f"Creating yearly windows for features: {feature_cols}")
    
    for station_name, station_data in preprocessed_data.items():
        # Check if we have the primary feature for determining time range
        primary_feature = feature_cols[0]
        if station_data[primary_feature] is not None and not station_data[primary_feature].empty:
            primary_data = station_data[primary_feature]
            
            # Ensure datetime index
            if not isinstance(primary_data.index, pd.DatetimeIndex):
                primary_data.set_index('Date', inplace=True)
            
            if primary_data.index.tz is not None:
                primary_data.index = primary_data.index.tz_localize(None)
            
            start_date = primary_data.index.min()
            end_date = primary_data.index.max()
            current_start = start_date
            
            while current_start + timedelta(days=window_size) <= end_date:
                window_end = current_start + timedelta(days=window_size)
                year_key = str(current_start.year)
                
                # Initialize the year and station in the results if needed
                if year_key not in split_result['windows']:
                    split_result['windows'][year_key] = {}
                if station_name not in split_result['windows'][year_key]:
                    split_result['windows'][year_key][station_name] = {}
                
                # Process each feature
                for feature in feature_cols:
                    if feature in station_data and station_data[feature] is not None:
                        feature_data = station_data[feature]
                        
                        # Ensure datetime index
                        if not isinstance(feature_data.index, pd.DatetimeIndex):
                            feature_data.set_index('Date', inplace=True)
                        
                        if feature_data.index.tz is not None:
                            feature_data.index = feature_data.index.tz_localize(None)
                        
                        # Get data for this window
                        window_data = feature_data[
                            (feature_data.index >= current_start) & 
                            (feature_data.index < window_end)
                        ]
                        
                        if not window_data.empty:
                            split_result['windows'][year_key][station_name][feature] = window_data
                        else:
                            print(f"Warning: No data for feature '{feature}' in window {year_key} for station {station_name}")
                            split_result['windows'][year_key][station_name][feature] = pd.DataFrame()
                    else:
                        print(f"Warning: Feature '{feature}' not found for station {station_name}")
                        split_result['windows'][year_key][station_name][feature] = pd.DataFrame()
                
                current_start = current_start + timedelta(days=window_size)
    
    return split_result

def split_data_rolling(preprocessed_data: dict, 
                      train_years: int = 3,
                      val_years: int = 1,
                      test_years: int = 2) -> dict:
    """
    Split data into rolling windows of training and validation sets, with final test set.
    Uses features specified in LSTM_CONFIG['feature_cols'] and includes the target feature.
    
    Returns:
        Dictionary with structure:
        {
            'windows': {
                0: {
                    'train': {...},
                    'validation': {...}
                },
                1: {...},
                ...
            },
            'test': {...}
        }
    """
    split_result = {
        'windows': {},  # Changed from list to dictionary
        'test': {}
    }
    
    # Get feature columns from config
    feature_cols = LSTM_CONFIG.get('feature_cols')
    
    # Get target feature (default to vst_raw if not specified)
    target_feature = LSTM_CONFIG.get('output_features', ['vst_raw'])[0]
    
    # Make sure target feature is included in processing
    all_features = list(feature_cols)
    if target_feature not in all_features:
        all_features.append(target_feature)
        print(f"Added target feature '{target_feature}' to features list for splitting")
    
    for station_name, station_data in preprocessed_data.items():
        # Check if target feature exists
        if target_feature not in station_data or station_data[target_feature] is None:
            print(f"Warning: Target feature '{target_feature}' missing for station {station_name}")
            print(f"Available features: {list(station_data.keys())}")
            continue
            
        primary_feature = target_feature  # Use target feature as primary
        primary_data = station_data[primary_feature]
        
        # Ensure datetime index
        if not isinstance(primary_data.index, pd.DatetimeIndex):
            primary_data.set_index('Date', inplace=True)
        
        # Convert to timezone-naive if needed
        if primary_data.index.tz is not None:
            primary_data.index = primary_data.index.tz_localize(None)
        
        start_date = primary_data.index.min()
        end_date = primary_data.index.max()
        
        # Reserve last test_years for testing
        test_start = end_date - pd.DateOffset(years=test_years)
        
        # Create rolling windows
        current_start = start_date
        window_size = pd.DateOffset(years=train_years + val_years)
        window_idx = 0  # Initialize window counter
        
        while current_start + window_size <= test_start:
            train_end = current_start + pd.DateOffset(years=train_years)
            val_end = train_end + pd.DateOffset(years=val_years)
            
            window_data = {
                'train': {station_name: {}},
                'validation': {station_name: {}}
            }
            
            # Process each feature for this window
            for feature in all_features:
                if feature in station_data and station_data[feature] is not None:
                    data = station_data[feature]
                    
                    # Ensure datetime index
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.set_index('Date', inplace=True)
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Split for current window
                    train_data = data[(data.index >= current_start) & 
                                    (data.index < train_end)].copy()
                    val_data = data[(data.index >= train_end) & 
                                  (data.index < val_end)].copy()
                    
                    window_data['train'][station_name][feature] = train_data
                    window_data['validation'][station_name][feature] = val_data
            
            split_result['windows'][window_idx] = window_data  # Use dictionary assignment
            window_idx += 1  # Increment window counter
            current_start = train_end  # Move to next window
        
        # Process final test set
        if station_name not in split_result['test']:
            split_result['test'][station_name] = {}
        
        for feature in all_features:
            if feature in station_data and station_data[feature] is not None:
                data = station_data[feature]
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                test_data = data[data.index >= test_start].copy()
                split_result['test'][station_name][feature] = test_data
    
    # Verify that target feature is included in all splits
    for window_idx, window_data in split_result['windows'].items():
        for split_type in ['train', 'validation']:
            for station_name in window_data[split_type]:
                if target_feature not in window_data[split_type][station_name]:
                    print(f"Warning: Target feature '{target_feature}' missing in window {window_idx}, {split_type} split for station {station_name}")
    
    for station_name in split_result['test']:
        if target_feature not in split_result['test'][station_name]:
            print(f"Warning: Target feature '{target_feature}' missing in test split for station {station_name}")
    
    return split_result

'''
1. Unsupervised Learning Context:
In unsupervised anomaly detection, the model learns what "normal" looks like by modeling patterns within the data. 
Training on a complete year and then applying the model to that same year can be valid because anomalies will be the values
that deviate from the learned normal patterns—even if those patterns are also used in training.

Imputation and Anomaly Detection:
Since your model is not forecasting but rather detecting anomalies and imputing missing or corrupt values, you care more about
understanding the relationships within the existing data rather than predicting unseen future values. 
In such cases, testing on the same period is acceptable because you want to gauge how well the model can flag 
anomalies and perform imputation in context.
Avoiding Data Leakage:
Although training and testing on the same year might raise concerns about data leakage in a supervised forecasting
context, with unsupervised or self-supervised methods (where there are no labels for "good" vs. "bad" outcomes)
this is less of an issue. The model learns the underlying structure of the data, and then you evaluate how well
it identifies deviations.

Windowed Evaluation for Robustness:
Even within the same year, you can consider further segmenting your analysis (for example, using a sliding window approach) to cross-check
the performance and robustness of the model. This can help ensure that the model is not overfitting to specific segments of the data.

Practical Use-Case Alignment:
If your use-case always operates on a single year of data (i.e., the model is retrained and executed on a yearly basis), then evaluating on that year makes sense. Over time, you might accumulate performance metrics across different years, which can later be used to refine your model further.
In summary, training and testing on the same yearly period is a valid approach here, provided you understand that this strategy aims to capture the normal behavior and then detect deviations from it. It's especially appropriate if your model is unsupervised and if the approach aligns with your operational workflow. Do keep in mind that, as you refine your model, you could also experiment with sliding windows or multiple-year evaluations to strengthen robustness.
'''