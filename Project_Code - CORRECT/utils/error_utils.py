"""
Error injection utilities for the water level prediction pipeline.
This module contains functions to handle synthetic error injection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

def configure_error_params(base_error_config, error_frequency):
    """
    Configure error parameters based on the specified error frequency.
    
    Args:
        base_error_config: Base error configuration dictionary
        error_frequency: Target error frequency (0-1)
        
    Returns:
        Modified error configuration with adjusted frequencies
    """
    # Create a copy of the base configuration
    error_config = base_error_config.copy()
    
    # Set error frequencies based on the error_frequency parameter
    # Distribute the total error frequency across different error types
    error_config['offset']['frequency'] = error_frequency * 0.3     # 30% of errors are offsets
    error_config['drift']['frequency'] = error_frequency * 0.2      # 20% of errors are drifts
    error_config['flatline']['frequency'] = error_frequency * 0.2   # 20% of errors are flatlines
    error_config['spike']['frequency'] = error_frequency * 0.15     # 15% of errors are spikes
    error_config['noise']['frequency'] = error_frequency * 0.15     # 15% of errors are noise
    
    return error_config

def print_error_frequencies(error_config):
    """
    Print the error frequencies in the configuration.
    
    Args:
        error_config: Error configuration dictionary
    """
    print(f"Error frequencies by type:")
    print(f"  Offset: {error_config['offset']['frequency']:.5f}")
    print(f"  Drift: {error_config['drift']['frequency']:.5f}")
    print(f"  Flatline: {error_config['flatline']['frequency']:.5f}")
    print(f"  Spike: {error_config['spike']['frequency']:.5f}")
    print(f"  Noise: {error_config['noise']['frequency']:.5f}")

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