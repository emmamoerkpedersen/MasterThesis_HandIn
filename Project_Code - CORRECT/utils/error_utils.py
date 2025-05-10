"""
Error injection utilities for the water level prediction pipeline.
This module contains functions to handle synthetic error injection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import copy

def configure_error_params(base_error_config, error_multiplier):
    """
    Configure error parameters based on the specified error multiplier.
    
    Args:
        base_error_config: Base error configuration dictionary
        error_multiplier: Multiplier for error counts per year
        
    Returns:
        Modified error configuration with adjusted counts
    """
    # Create a deep copy of the base configuration
    error_config = copy.deepcopy(base_error_config)
    
    # Apply multiplier to existing count_per_year values
    for error_type in error_config:
        if isinstance(error_config[error_type], dict) and 'count_per_year' in error_config[error_type]:
            # Apply multiplier to the existing value, not overriding with hardcoded values
            error_config[error_type]['count_per_year'] *= error_multiplier
    
    return error_config

def print_error_frequencies(error_config):
    """
    Print the error counts per year in the configuration.
    
    Args:
        error_config: Error configuration dictionary
    """
    print(f"Error counts per year by type:")
    for error_type, settings in error_config.items():
        if isinstance(settings, dict) and 'count_per_year' in settings:
            print(f"  {error_type.capitalize()}: {settings['count_per_year']:.1f}")

def inject_errors_into_dataset(dataset, error_generator, station_id, water_level_cols):
    """
    Inject synthetic errors into a dataset's water level columns.
    
    Args:
        dataset: DataFrame containing the data
        error_generator: SyntheticErrorGenerator instance
        station_id: Station identifier string
        water_level_cols: List of water level column names
        
    Returns:
        Tuple of (modified_dataset, error_periods_dict)
    """
    # Make a copy of the dataset to modify
    dataset_with_errors = dataset.copy()
    
    # Dictionary to store error periods
    station_results = {}
    
    # Process each water level column
    for column in water_level_cols:
        print(f"\nProcessing {column}...")
        
        # Create a single-column DataFrame for the error generator
        column_data = pd.DataFrame({
            'vst_raw': dataset[column]  # Error generator expects 'vst_raw' column
        })
        
        # Skip if column contains only NaN values
        if column_data['vst_raw'].isna().all():
            print(f"Column {column} contains only NaN values, skipping")
            continue
        
        # Generate synthetic errors
        print(f"Generating synthetic errors for {column}...")
        try:
            modified_data, ground_truth = error_generator.inject_all_errors(column_data)
            
            # Store the error periods
            station_key = f"{station_id}_{column}"
            station_results[station_key] = {
                'modified_data': modified_data,
                'ground_truth': ground_truth,
                'error_periods': error_generator.error_periods.copy()  # Important to copy!
            }
            
            print(f"Injected {len(error_generator.error_periods)} error periods into {column}")
            
            # Update the dataset with errors
            dataset_with_errors[column] = modified_data['vst_raw']
            
        except Exception as e:
            print(f"Error generating synthetic errors for {column}: {str(e)}")
            traceback.print_exc()
    
    return dataset_with_errors, station_results

def identify_water_level_columns(dataset, station_id=None):
    """
    Identify columns in the dataset that contain water level data.
    
    Args:
        dataset: DataFrame containing the data
        station_id: Optional station identifier to filter columns
        
    Returns:
        List of water level column names
    """
    # Method 1: Look for columns with 'vst_raw' in the name
    vst_cols = [col for col in dataset.columns if 'vst_raw' in col.lower()]
    
    # Method 2: If station_id is provided, look for columns with that station id
    if station_id and not vst_cols:
        vst_cols = [col for col in dataset.columns if f'station_{station_id}' in col.lower() and 'vst' in col.lower()]
    
    # Method 3: As a fallback, look for any column with 'water_level' or similar
    if not vst_cols:
        vst_cols = [col for col in dataset.columns if any(term in col.lower() for term in ['water_level', 'waterlevel', 'level', 'vst'])]
    
    return vst_cols 