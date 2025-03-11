"""
Iterative confidence interval-based anomaly detector.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pickle
from datetime import datetime

from _3_lstm_model.torch_lstm_model import evaluate_realistic, train_autoencoder

class ConfidenceIntervalAnomalyDetector:
    """Iterative anomaly detector using confidence intervals for individual stations."""
    
    def __init__(
        self, 
        config: Dict, 
        models_dir: Union[str, Path],
        confidence_level: float = 0.95,
        max_iterations: int = 3
    ):
        """
        Initialize the detector.
        
        Args:
            config: Model configuration
            models_dir: Directory for model storage
            confidence_level: Confidence level for anomaly detection (default: 0.95)
            max_iterations: Maximum number of refinement iterations (default: 3)
        """
        self.config = config
        self.models_dir = Path(models_dir)
        self.confidence_level = confidence_level
        self.max_iterations = max_iterations
        
        # Calculate z-score based on confidence level
        if confidence_level == 0.95:
            self.z_score = 1.96
        elif confidence_level == 0.99:
            self.z_score = 2.58
        elif confidence_level == 0.90:
            self.z_score = 1.645
        else:
            self.z_score = 1.96
            
    def train_station_model(
        self, 
        station_id: str,
        train_data: Dict, 
        validation_data: Optional[Dict] = None
    ) -> Tuple:
        """
        Train model iteratively for a single station, refining training data each iteration.
        
        Args:
            station_id: ID of the station
            train_data: Training data dictionary for this station
            validation_data: Validation data dictionary for this station (optional)
            
        Returns:
            Tuple of (model, history, anomalies_per_iteration, cleaned_train_data)
        """
        print(f"\nInitiating iterative training process for station {station_id}...")
        print(f"Confidence level: {self.confidence_level*100}% (z-score: {self.z_score})")
        print(f"Maximum iterations: {self.max_iterations}")
        
        # Step 1: Initial training on station data
        print(f"Initial training for station {station_id}...")
        model, history = train_autoencoder(train_data, validation_data, self.config)
        
        # Track training data refinement
        current_train_data = train_data.copy()
        anomalies_per_iteration = []
        
        # Track iterations for early stopping
        total_anomalies_found = 0
        consecutive_low_anomaly_iterations = 0
        min_anomalies_per_iteration = 10
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration+1}/{self.max_iterations} for station {station_id}")
            print(f"{'='*50}")
            
            # Progressively lower threshold
            percentile_threshold = max(90 - (iteration * 2), 75)
            print(f"Using percentile threshold: {percentile_threshold}% for anomaly detection")
            
            # Find anomalies in training data
            anomalies = self._find_anomalies_in_training(model, current_train_data, percentile_threshold)
            anomalies_per_iteration.append(anomalies)
            
            print(f"Found {len(anomalies)} potential anomalies in training data")
            
            # Early stopping criteria
            if len(anomalies) == 0:
                print("No anomalies found, stopping iterations")
                break
                
            if len(anomalies) < min_anomalies_per_iteration:
                consecutive_low_anomaly_iterations += 1
                print(f"Few anomalies found ({len(anomalies)} < {min_anomalies_per_iteration})")
                print(f"Low anomaly iterations: {consecutive_low_anomaly_iterations}/2")
                
                if consecutive_low_anomaly_iterations >= 2:
                    print("Two consecutive iterations with few anomalies, stopping")
                    break
            else:
                consecutive_low_anomaly_iterations = 0
                
            total_anomalies_found += len(anomalies)
            
            # Refine training data
            refined_train_data = self._refine_training_data(current_train_data, anomalies)
            
            # Retrain on refined data
            print(f"Retraining model for station {station_id} on refined data...")
            model, history = train_autoencoder(refined_train_data, validation_data, self.config)
            
            # Update for next iteration
            current_train_data = refined_train_data
            
            # Summary
            print(f"\nCompleted iteration {iteration+1} for station {station_id}:")
            print(f"  Anomalies found this iteration: {len(anomalies)}")
            print(f"  Total anomalies found so far: {total_anomalies_found}")
            print(f"  Remaining iterations: {self.max_iterations - (iteration+1)}")
        
        print(f"\nIterative training complete for station {station_id}.")
        print(f"Total iterations performed: {iteration+1}")
        total_anomalies = sum(len(a) for a in anomalies_per_iteration)
        print(f"Total anomalies identified and handled: {total_anomalies}")
        
        # Save the final model
        model_path = self.models_dir / f"{station_id}_model.pt"
        model.save(str(model_path))
        
        return model, history, anomalies_per_iteration, current_train_data
    
    def _find_anomalies_in_training(self, model, train_data, percentile_threshold=90):
        """
        Find anomalies in training data using multiple detection criteria.
        
        Args:
            model: Trained model
            train_data: Training data dictionary
            percentile_threshold: Percentile to use for error threshold (default: 90)
            
        Returns:
            List of anomalies with metadata
        """
        anomalies = []
        
        # Process each station's raw data
        for station_key, station_data in train_data.items():
            try:
                # Get the raw data
                if 'vst_raw' not in station_data or not isinstance(station_data['vst_raw'], pd.DataFrame):
                    print(f"Skipping {station_key} - no valid raw data")
                    continue
                
                raw_df = station_data['vst_raw']
                if raw_df.empty:
                    continue
                
                # Create a temporary structure for evaluation with the correct key
                # This is the key change: Map 'vst_raw' to 'vst_raw_modified' for evaluate_realistic
                eval_data = {station_key: {'vst_raw_modified': raw_df}}
                
                # Get predictions and errors
                results = evaluate_realistic(model, eval_data, {}, self.config)
                
                if station_key not in results:
                    continue
                    
                result = results[station_key]
                if 'reconstruction_errors' not in result:
                    continue
                
                errors = result['reconstruction_errors']
                timestamps = result['timestamps']
                reconstructions = result['reconstructions']
                original_values = raw_df['Value'].values
                
                # 1. Reconstruction Error-based Detection
                # Use configurable threshold based on iteration
                error_threshold = np.percentile(errors, percentile_threshold)
                
                # 2. Value-based Statistical Detection
                window_size = 48  # 12 hours with 15-min data
                values_series = pd.Series(original_values)
                rolling_mean = values_series.rolling(window=window_size, center=True).mean()
                rolling_std = values_series.rolling(window=window_size, center=True).std()
                
                # 3. Spike Detection
                # Calculate local variation with more sensitivity
                value_diff = np.abs(values_series - values_series.shift(1))
                spike_threshold = np.percentile(value_diff.dropna(), percentile_threshold + 5)  # More sensitive spike detection
                
                # Additional: Rate of Change Detection
                rate_of_change = value_diff / values_series.shift(1).abs()
                rate_threshold = np.percentile(rate_of_change.dropna(), percentile_threshold + 5)
                
                # Combine detection methods
                for i, (ts, error) in enumerate(zip(timestamps, errors)):
                    is_anomaly = False
                    anomaly_types = []
                    
                    # Skip if we don't have enough context
                    if i < window_size//2 or i >= len(original_values) - window_size//2:
                        continue
                    
                    # Check reconstruction error
                    if error > error_threshold:
                        is_anomaly = True
                        anomaly_types.append('reconstruction')
                    
                    # Check statistical outlier (more sensitive)
                    if i < len(rolling_mean) and i < len(rolling_std) and rolling_std[i] > 0:
                        # Make Z-score threshold progressively lower
                        z_threshold = max(2.5 - (100 - percentile_threshold) * 0.05, 1.5)
                        z_score = abs(original_values[i] - rolling_mean[i]) / rolling_std[i]
                        if z_score > z_threshold:
                            is_anomaly = True
                            anomaly_types.append('statistical')
                    
                    # Check for spikes
                    if i > 0 and i < len(value_diff):
                        if value_diff[i] > spike_threshold:
                            # Check if it's a significant relative change
                            if i < len(rate_of_change) and rate_of_change[i] > rate_threshold:
                                is_anomaly = True
                                anomaly_types.append('spike')
                    
                    if is_anomaly:
                        # This is a candidate anomaly in the training data
                        anomalies.append({
                            'station_key': station_key,
                            'timestamp': ts,
                            'index': i,
                            'error': error,
                            'error_threshold': error_threshold,
                            'original_value': original_values[i],
                            'reconstructed_value': reconstructions[i] if i < len(reconstructions) else None,
                            'local_mean': rolling_mean[i] if i < len(rolling_mean) else None,
                            'local_std': rolling_std[i] if i < len(rolling_std) else None,
                            'detection_methods': anomaly_types
                        })
                
                station_anomalies = sum(1 for a in anomalies if a['station_key'] == station_key)
                print(f"  {station_key}: Found {station_anomalies} anomalies")
                if station_anomalies > 0:
                    methods_count = {}
                    for a in anomalies:
                        if a['station_key'] == station_key:
                            for method in a['detection_methods']:
                                methods_count[method] = methods_count.get(method, 0) + 1
                    print(f"    Detection methods: {methods_count}")
                    
            except Exception as e:
                print(f"Error processing {station_key}: {e}")
                import traceback
                traceback.print_exc()
        
        return anomalies
    
    def _refine_training_data(self, train_data, anomalies):
        """
        Create refined training data by handling anomalies.
        
        Args:
            train_data: Original training data dictionary
            anomalies: List of anomalies to handle
            
        Returns:
            Refined training data dictionary
        """
        refined_data = {}
        
        # Group anomalies by station_key
        anomalies_by_station = {}
        for anomaly in anomalies:
            station_key = anomaly['station_key']
            if station_key not in anomalies_by_station:
                anomalies_by_station[station_key] = []
            anomalies_by_station[station_key].append(anomaly)
        
        # Process each station in the training data
        for station_key, data in train_data.items():
            if station_key in anomalies_by_station:
                # Create a copy of the data
                refined_station_data = {}
                
                for key, value in data.items():
                    if key == 'vst_raw' and isinstance(value, pd.DataFrame):
                        # Create a copy of the dataframe
                        df_copy = value.copy()
                        
                        # Create a mask of anomalous timestamps
                        anomalous_timestamps = [a['timestamp'] for a in anomalies_by_station[station_key]]
                        
                        # Replace with interpolated values
                        for ts in anomalous_timestamps:
                            if ts in df_copy.index:
                                # Use simple linear interpolation
                                idx = df_copy.index.get_loc(ts)
                                
                                # Get values before and after
                                before_idx = max(0, idx-5)  # Look back up to 5 points
                                after_idx = min(len(df_copy)-1, idx+5)  # Look ahead up to 5 points
                                
                                # Find valid points (not at the anomaly timestamp)
                                before_values = df_copy['Value'].iloc[before_idx:idx].values
                                after_values = df_copy['Value'].iloc[idx+1:after_idx+1].values
                                
                                if len(before_values) > 0 and len(after_values) > 0:
                                    # Take the mean of values before and after
                                    before_val = np.median(before_values)
                                    after_val = np.median(after_values)
                                    
                                    # Replace with interpolated value
                                    df_copy.at[ts, 'Value'] = (before_val + after_val) / 2
                                elif len(before_values) > 0:
                                    # Only use before values
                                    df_copy.at[ts, 'Value'] = np.median(before_values)
                                elif len(after_values) > 0:
                                    # Only use after values
                                    df_copy.at[ts, 'Value'] = np.median(after_values)
                        
                        refined_station_data[key] = df_copy
                    else:
                        refined_station_data[key] = value
                
                refined_data[station_key] = refined_station_data
                
                # Report refinement
                print(f"  Refined {station_key}: Handled {len(anomalies_by_station[station_key])} anomalies")
            else:
                # No anomalies, keep as is
                refined_data[station_key] = data
        
        return refined_data 