import pandas as pd
from typing import Dict

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
