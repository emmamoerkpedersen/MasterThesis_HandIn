import pandas as pd
from typing import Dict
from datetime import timedelta

def split_data(preprocessed_data: dict, 
               train_ratio: float = 0.6,
               validation_ratio: float = 0.2) -> dict:
    """
    Split preprocessed data into training, validation and test sets chronologically.
    
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
    
    for station_name, station_data in preprocessed_data.items():
        # Initialize empty dictionaries for this station in each split
        split_data['train'][station_name] = {}
        split_data['validation'][station_name] = {}
        split_data['test'][station_name] = {}
        
        # Get the time range for this station from VST_RAW data
        if station_data['vst_raw'] is not None:
            vst_data = station_data['vst_raw']
            
            # Ensure we're working with datetime index
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
            
            # Get the date range (ensure timezone-naive)
            start_date = vst_data.index.min().tz_localize(None)
            end_date = vst_data.index.max().tz_localize(None)
            date_range = end_date - start_date
            
            # Calculate split dates
            train_end = start_date + pd.Timedelta(days=date_range.days * train_ratio)
            val_end = start_date + pd.Timedelta(days=date_range.days * (train_ratio + validation_ratio))
            
            print(f"\nSplitting {station_name} data:")
            print(f"Training period: {start_date.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"Validation period: {train_end.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
            print(f"Testing period: {val_end.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Split each data type using the same date ranges
            for data_type, data in station_data.items():
                if data is not None and isinstance(data, pd.DataFrame):
                    # Ensure datetime index and convert to timezone-naive
                    if not isinstance(data.index, pd.DatetimeIndex):
                        data.set_index('Date', inplace=True)
                    
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Split based on dates
                    train_data = data[data.index <= train_end].copy()
                    val_data = data[(data.index > train_end) & (data.index <= val_end)].copy()
                    test_data = data[data.index > val_end].copy()
                    
                    split_data['train'][station_name][data_type] = train_data
                    split_data['validation'][station_name][data_type] = val_data
                    split_data['test'][station_name][data_type] = test_data
                else:
                    # Initialize empty DataFrames for missing data types
                    split_data['train'][station_name][data_type] = pd.DataFrame()
                    split_data['validation'][station_name][data_type] = pd.DataFrame()
                    split_data['test'][station_name][data_type] = pd.DataFrame()
    
    return split_data

def split_data_yearly(preprocessed_data: dict, window_size: int = 365) -> dict:
    """
    Splits the preprocessed data into yearly windows for unsupervised anomaly detection.
    Each window is used for both training and evaluation.
    
    The model learns normal patterns from the entire window and identifies
    anomalies as deviations from these patterns.
    """
    split_result = {'windows': {}}
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None and not station_data['vst_raw'].empty:
            vst_data = station_data['vst_raw']
            
            # Ensure datetime index
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
            
            if vst_data.index.tz is not None:
                vst_data.index = vst_data.index.tz_localize(None)
            
            start_date = vst_data.index.min()
            end_date = vst_data.index.max()
            current_start = start_date
            
            while current_start + timedelta(days=window_size) <= end_date:
                window_end = current_start + timedelta(days=window_size)
                year_key = str(current_start.year)
                
                # Get data for this window
                window_data = vst_data[(vst_data.index >= current_start) & 
                                     (vst_data.index < window_end)]
                
                # Only create window if there's actually data in it
                if not window_data.empty:
                    if year_key not in split_result['windows']:
                        split_result['windows'][year_key] = {}
                    if station_name not in split_result['windows'][year_key]:
                        split_result['windows'][year_key][station_name] = {}
                    
                    # Store the window data
                    split_result['windows'][year_key][station_name]['vst_raw'] = window_data
                    
                current_start = current_start + timedelta(days=window_size)
    
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