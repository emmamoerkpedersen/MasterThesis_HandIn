import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

from shared.anomaly_detection.z_score import calculate_z_scores_mad

class SimpleAnomalyDetector:
    """
    Simple anomaly detector using Z-score MAD method.
    Can flag anomalous periods and add flags as features to data.
    """
    
    def __init__(self, threshold=3.0, window_size=1500):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            window_size: Window size for MAD calculation
        """
        self.threshold = threshold
        self.window_size = window_size
        self.anomaly_flags = None
        
    def detect_anomalies(self, data, reference_data=None):
        """
        Detect anomalies in the data using Z-score MAD.
        
        Args:
            data: Data to check for anomalies (pandas Series or numpy array)
            reference_data: Reference data to compare against (if None, uses data itself)
            
        Returns:
            anomaly_flags: Boolean array indicating anomalous periods
            z_scores: Z-scores for each data point
        """
        if reference_data is None:
            reference_data = data
            
        # Convert to numpy arrays if needed
        if isinstance(data, pd.Series):
            data = data.values
        if isinstance(reference_data, pd.Series):
            reference_data = reference_data.values
            
        # Calculate Z-scores using MAD
        z_scores, anomaly_flags = calculate_z_scores_mad(
            reference_data, 
            data, 
            window_size=self.window_size,
            threshold=self.threshold
        )
        
        self.anomaly_flags = anomaly_flags
        return anomaly_flags, z_scores
    
    def create_perfect_flags(self, error_generator, data_length, data_index=None):
        """
        Create 'perfect' anomaly flags based on known synthetic error locations.
        
        Args:
            error_generator: SyntheticErrorGenerator object with error_periods
            data_length: Length of the data
            data_index: Optional pandas DatetimeIndex for matching timestamps
            
        Returns:
            anomaly_flags: Boolean array with True for known error locations
        """
        anomaly_flags = np.zeros(data_length, dtype=bool)
        
        if hasattr(error_generator, 'error_periods') and error_generator.error_periods:
            print(f"Creating perfect flags for {len(error_generator.error_periods)} error periods")
            
            for error_period in error_generator.error_periods:
                if data_index is not None:
                    # Convert timestamps to indices
                    try:
                        start_idx = data_index.get_loc(error_period.start_time)
                        end_idx = data_index.get_loc(error_period.end_time) + 1
                    except KeyError:
                        # Handle case where exact timestamp isn't found
                        start_idx = data_index.searchsorted(error_period.start_time)
                        end_idx = data_index.searchsorted(error_period.end_time) + 1
                else:
                    # If no index provided, try to extract indices from error_period
                    # This is a fallback - may not work in all cases
                    print("Warning: No data index provided, attempting to extract indices")
                    continue
                
                # Flag the entire error period
                if end_idx <= data_length and start_idx >= 0:
                    anomaly_flags[start_idx:end_idx] = True
                    print(f"  - Flagged {error_period.error_type} from {start_idx} to {end_idx} ({error_period.start_time} to {error_period.end_time})")
                else:
                    print(f"  - Warning: Error period {start_idx}-{end_idx} exceeds data bounds")
        else:
            print("No error periods found - error generator may not have tracked injection locations")
            
        return anomaly_flags
    
    def add_anomaly_flags_to_dataframe(self, df, anomaly_flags, flag_column_name='anomaly_flag'):
        """
        Add anomaly flags as a new column to the DataFrame.
        
        Args:
            df: DataFrame to add flags to
            anomaly_flags: Boolean array of anomaly flags
            flag_column_name: Name for the new flag column
            
        Returns:
            df_with_flags: DataFrame with added anomaly flag column
        """
        df_with_flags = df.copy()
        
        # Ensure the flags array matches the DataFrame length
        if len(anomaly_flags) != len(df):
            print(f"Warning: Anomaly flags length ({len(anomaly_flags)}) doesn't match DataFrame length ({len(df)})")
            # Truncate or pad as needed
            if len(anomaly_flags) > len(df):
                anomaly_flags = anomaly_flags[:len(df)]
            else:
                # Pad with False values
                padded_flags = np.zeros(len(df), dtype=bool)
                padded_flags[:len(anomaly_flags)] = anomaly_flags
                anomaly_flags = padded_flags
        
        # Add flags as new column (convert bool to int for easier handling)
        df_with_flags[flag_column_name] = anomaly_flags.astype(int)
        
        print(f"Added anomaly flags: {np.sum(anomaly_flags)} anomalous periods out of {len(df)} total")
        print(f"Anomaly percentage: {np.sum(anomaly_flags) / len(df) * 100:.2f}%")
        
        return df_with_flags
    
    def get_anomaly_statistics(self, anomaly_flags):
        """
        Get basic statistics about detected anomalies.
        
        Args:
            anomaly_flags: Boolean array of anomaly flags
            
        Returns:
            stats: Dictionary with anomaly statistics
        """
        total_points = len(anomaly_flags)
        anomalous_points = np.sum(anomaly_flags)
        normal_points = total_points - anomalous_points
        
        # Find anomalous periods (consecutive True values)
        anomalous_periods = []
        in_anomaly = False
        start_idx = None
        
        for i, is_anomaly in enumerate(anomaly_flags):
            if is_anomaly and not in_anomaly:
                # Start of new anomalous period
                start_idx = i
                in_anomaly = True
            elif not is_anomaly and in_anomaly:
                # End of anomalous period
                anomalous_periods.append((start_idx, i-1))
                in_anomaly = False
        
        # Handle case where data ends during anomalous period
        if in_anomaly:
            anomalous_periods.append((start_idx, len(anomaly_flags)-1))
        
        stats = {
            'total_points': total_points,
            'anomalous_points': anomalous_points,
            'normal_points': normal_points,
            'anomaly_percentage': (anomalous_points / total_points) * 100,
            'number_of_periods': len(anomalous_periods),
            'anomalous_periods': anomalous_periods
        }
        
        return stats 