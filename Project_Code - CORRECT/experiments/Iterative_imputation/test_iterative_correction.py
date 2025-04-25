"""
Test script for the iterative anomaly correction pipeline.

This module tests a retrospective data cleaning approach for water level data. The system
is designed for batch processing of complete historical datasets after they've been collected
(e.g., cleaning a year of data at the beginning of the next year).

Key characteristics of this approach:

1. MODEL APPROACH:
   - Uses an LSTM-based model to generate expected water level values
   - Leverages neighboring station data as key predictive features
   - Iteratively improves by retraining on progressively cleaner data

2. ANOMALY DETECTION:
   - Primarily looks for negative drops and deviations from expected values
   - Compares model predictions with actual values to identify anomalies
   - Uses a custom anomaly detection algorithm focused on drops of interest
   - Employs a directional bias to prioritize detecting significant drops

3. CORRECTION PROCESS:
   - Replaces anomalous points with model predictions
   - Applies sophisticated smoothing to ensure natural transitions
   - Iteratively refines corrections until convergence
   - Uses both interpolation and exponential smoothing for natural results

4. DATA USAGE:
   - Uses complete historical context (not limited to past data only)
   - Takes advantage of neighboring station correlations
   - Works well when neighboring stations provide reliable reference points
   - Most effective when water levels are correlated across nearby stations

5. IMPORTANT CONSIDERATIONS:
   - Not designed for real-time anomaly detection (uses future context)
   - The model's effectiveness depends on having correlated neighboring station data
   - Extensive context window (sequence_length) allows for learning long-term patterns
   - May tend to align water levels with neighboring stations if they differ significantly

This retrospective approach prioritizes creating a cleaned dataset that maintains
natural water level patterns while removing specific anomalous readings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Add the parent directory to Python path
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Points to Project_Code - CORRECT
sys.path.append(str(project_root))

# Import configuration and modules
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from experiments.Iterative_imputation.config import LSTM_CONFIG, ANOMALY_CORRECTION_PARAMS, SYNTHETIC_ERROR_PARAMS
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from experiments.Iterative_imputation.iterative_anomaly_correction import run_iterative_correction_pipeline
from experiments.Iterative_imputation.visualization_utils import visualize_all_results

# Suppress ConvergenceWarnings from statsmodels
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def inject_controlled_errors(data):
    """
    Add specific synthetic errors for testing anomaly detection and correction.
    
    Args:
        data: DataFrame with water level data
        
    Returns:
        DataFrame with injected errors
    """
    # Create custom error configuration focusing on negative spikes and offsets
    error_config = {
        'spike': {
            'frequency': 0.01,  # Frequency of spike errors (1% of data points)
            'magnitude_range': [0.1, 0.4],  # 10-40% of value
            'negative_positiv_ratio': 0.5,  # Mostly negative spikes (90%)
            'recovery_time': 1,
        },
        'offset': {
            'frequency': 0.003,  # 0.3% chance for offset segments
            'magnitude_range': [0.2, 0.5],  # 20-50% of value
            'negative_positiv_ratio': 0.8,  # Mostly negative offsets
            'min_duration': 24,  # At least 24 hours
            'max_duration_multiplier': [1.5, 3.0],  # 1.5-3x the min duration
            'magnitude_multiplier': [0.8, 1.2],  # Local context adjustment
        },
        'PHYSICAL_LIMITS': {
            'min_value': 0,
            'max_value': 3000,
            'max_rate_of_change': 50
        },
        'use_context_aware': True,  # Use local context for error magnitudes
    }
    
    # Calculate and display expected number of spikes based on frequency
    expected_spikes = int(len(data) * error_config['spike']['frequency'])
    print(f"\nSpike frequency: {error_config['spike']['frequency']}")
    print(f"Data length: {len(data)} points")
    print(f"Expected number of spikes: {expected_spikes}")
    
    # Initialize error generator
    print("Creating synthetic errors for testing...")
    error_generator = SyntheticErrorGenerator(error_config)
    
    # To avoid the index/value length mismatch error, we'll try to inject only spikes first
    # and then add minimal offsets if that's successful
    try:
        # First try with spikes only
        data_with_errors, ground_truth = error_generator.inject_all_errors(
            data, error_types=['spike']
        )
        
        # If that worked, cautiously try to add one offset
        if error_config['offset']['frequency'] > 0:
            try:
                # Create a temporary error config with just a single offset
                temp_config = error_config.copy()
                temp_config['offset']['frequency'] = min(0.001, 1/len(data))  # Just one offset
                
                # Create a new generator for this
                temp_generator = SyntheticErrorGenerator(temp_config)
                
                # Add a single offset to the already-spiked data
                data_with_errors, _ = temp_generator.inject_all_errors(
                    data_with_errors, error_types=['offset']
                )
                
                # Combine error periods from both generators
                error_generator.error_periods.extend(temp_generator.error_periods)
            except Exception as e:
                print(f"Warning: Could not add offset errors: {e}")
                # Continue with spike-only data
    except Exception as e:
        print(f"Error injecting errors: {e}")
        return data  # Return original data if error injection fails completely
    
    # Print summary of injected errors
    error_periods = error_generator.error_periods
    error_types = {}
    for period in error_periods:
        error_type = period.error_type
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    print("\nSummary of injected errors:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}")
    
    # Explicitly verify if the spike frequency affected the number of injected spikes
    if 'spike' in error_types:
        if error_types['spike'] == 0 and error_config['spike']['frequency'] > 0:
            print("WARNING: No spikes were injected despite a non-zero spike frequency!")
        elif error_types['spike'] > 0 and error_config['spike']['frequency'] == 0:
            print("WARNING: Spikes were injected despite a zero spike frequency!")
        else:
            print(f"Spike frequency validation: Expected ~{expected_spikes}, got {error_types.get('spike', 0)}")
    
    return data_with_errors

def apply_artificial_scaling_error(data, start_idx=1000, duration=50, scale_factor=0.2):
    """
    Apply a simple artificial scaling error to the data.
    
    Args:
        data: DataFrame with water level data
        start_idx: Index to start the error
        duration: Duration of the error in data points
        scale_factor: Factor to scale the data by (e.g., 0.2 = reduce to 20%)
        
    Returns:
        DataFrame with injected scaling error
    """
    # Make a copy to avoid modifying the original
    modified_data = data.copy()
    
    # Determine end index
    end_idx = min(start_idx + duration, len(modified_data) - 1)
    
    # Store original values
    original_values = modified_data.iloc[start_idx:end_idx]['vst_raw'].copy()
    
    # Apply scaling factor
    modified_data.iloc[start_idx:end_idx, modified_data.columns.get_loc('vst_raw')] = original_values * scale_factor
    
    print(f"Applied artificial scaling error at indices {start_idx}-{end_idx} with scale factor {scale_factor}")
    print(f"  Original range: {original_values.min():.1f}-{original_values.max():.1f}")
    print(f"  Modified range: {(original_values * scale_factor).min():.1f}-{(original_values * scale_factor).max():.1f}")
    
    return modified_data

def main():
    # Start timing
    start_time = time.time()
    
    print("Testing iterative anomaly correction...")
    
    # Set up paths
    output_path = current_dir / "results" / "iterative_correction_test"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Using project root: {project_root}")
    print(f"Saving results to: {output_path}")
    
    # Load configuration
    model_config = LSTM_CONFIG.copy()
    
    # Set parameters for faster testing
    test_config = model_config.copy()
    test_config['epochs'] = 40  # Use moderate number of epochs
    test_config['batch_size'] = 32  # Smaller batch size for more granular updates
    test_config['learning_rate'] = 0.001  # Standard learning rate
    
    print(f"Using configuration with {test_config['num_layers']} layers, hidden size {test_config['hidden_size']}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(test_config)
    
    # Load data for station 21006846
    station_id = '21006846'
    print(f"Loading data for station {station_id}...")
    # Pass project_root directly - preprocessor will construct the full data path
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # Option to inject synthetic errors for testing
    use_synthetic_errors = True
    
    # Choose error injection method
    # Set to "synthetic" to use SyntheticErrorGenerator, "scaling" to use direct scaling errors
    error_injection_method = "scaling"  # Options: "synthetic", "scaling"
    
    if use_synthetic_errors:
        # Save original data for comparison
        original_test_data = test_data.copy()
        
        # Focus on a shorter segment of data for clearer visualization
        test_segment = test_data.iloc[:8000].copy()  # Use first 8000 points only
        original_test_data = original_test_data.iloc[:8000].copy()
        
        if error_injection_method == "synthetic":
            print("\nUsing SyntheticErrorGenerator for error injection")
            print("Note: The spike frequency parameter should affect the number of injected spikes")
            
            # This will use the inject_controlled_errors function with the specified error_config
            test_segment = inject_controlled_errors(test_segment)
            
        elif error_injection_method == "scaling":
            print("\nUsing direct scaling errors")
            # Apply 3 significant scaling errors at different points with more dramatic scale factors
            test_segment = apply_artificial_scaling_error(test_segment, start_idx=1000, duration=120, scale_factor=0.3)
            test_segment = apply_artificial_scaling_error(test_segment, start_idx=3000, duration=100, scale_factor=0.4)
            test_segment = apply_artificial_scaling_error(test_segment, start_idx=6000, duration=150, scale_factor=0.25)
        
        # Use the modified segment for testing
        test_data = test_segment
    else:
        original_test_data = test_data.copy()
        test_data = test_data.copy()
    
    # Run iterative correction
    print("\nRunning iterative anomaly correction...")
    
    # Adjust parameters to better detect and correct scaling errors
    max_iterations = 5     
    contamination = 0.05   # Lower contamination for more selective anomaly detection
    smoothing_window = 5   # Larger smoothing window for better transitions
    magnitude_threshold = 1.0  # Lower threshold to detect more subtle anomalies
    direction_bias = -1    # Focus on negative drops
    
    results, diff_stats, correction_history, anomaly_masks = run_iterative_correction_pipeline(
        test_config,
        preprocessor,
        train_data,
        val_data,
        test_data,
        output_path=output_path,
        max_iterations=max_iterations,
        contamination=contamination,
        smoothing_window=smoothing_window,
        magnitude_threshold=magnitude_threshold,
        direction_bias=direction_bias
    )
    
    # Generate all visualizations
    error_data = test_data if use_synthetic_errors else None
    visualize_all_results(
        original_test_data, 
        error_data, 
        correction_history, 
        anomaly_masks, 
        results, 
        output_path
    )
    
    # Calculate RMSE between original and corrected
    valid_mask = ~np.isnan(original_test_data['vst_raw']) & ~np.isnan(results['corrected'])
    original_vs_corrected_rmse = np.sqrt(((original_test_data.loc[valid_mask, 'vst_raw'] - 
                                        results.loc[valid_mask, 'corrected']) ** 2).mean())
    print(f"RMSE between original and corrected: {original_vs_corrected_rmse:.4f}")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 