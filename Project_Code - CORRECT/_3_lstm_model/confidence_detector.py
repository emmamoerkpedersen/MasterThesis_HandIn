"""
Iterative confidence interval-based anomaly detector.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import pickle
from datetime import datetime

from _3_lstm_model.model_registry import ModelRegistry
from _3_lstm_model.torch_lstm_model import evaluate_realistic

class ConfidenceIntervalAnomalyDetector:
    """Iterative anomaly detector using confidence intervals."""
    
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
        self.registry = ModelRegistry(config, models_dir)
        self.confidence_level = confidence_level
        self.max_iterations = max_iterations
        
        # Calculate z-score based on confidence level
        # For 95% confidence, z-score is approximately 1.96
        if confidence_level == 0.95:
            self.z_score = 1.96
        elif confidence_level == 0.99:
            self.z_score = 2.58
        elif confidence_level == 0.90:
            self.z_score = 1.645
        else:
            # Default to 95% confidence
            self.z_score = 1.96
        
    def iterative_training(
        self, 
        train_data: Dict, 
        validation_data: Optional[Dict] = None
    ) -> Tuple:
        """
        Train model iteratively, refining training data each iteration.
        
        Args:
            train_data: Training data dictionary
            validation_data: Validation data dictionary (optional)
            
        Returns:
            Tuple of (model, history, anomalies_per_iteration, cleaned_train_data)
        """
        # Step 1: Initial training on all data
        print("\nInitiating iterative training process...")
        print(f"Confidence level: {self.confidence_level*100}% (z-score: {self.z_score})")
        print(f"Maximum iterations: {self.max_iterations}")
        print(f"Initial training on all data ({len(train_data)} station-years)...")
        
        model, history = self.registry.train_global_model(train_data, validation_data)
        
        # Track training data refinement
        current_train_data = train_data.copy()
        anomalies_per_iteration = []
        
        # Track iterations for early stopping
        total_anomalies_found = 0
        consecutive_low_anomaly_iterations = 0
        min_anomalies_per_iteration = 10  # Minimum meaningful number of anomalies
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration+1}/{self.max_iterations}")
            print(f"{'='*50}")
            
            # Progressively lower the anomaly detection threshold in later iterations
            # This helps find more subtle anomalies after obvious ones are removed
            percentile_threshold = max(90 - (iteration * 2), 75)  # Start at 90%, decrease by 2% each iteration, floor at 75%
            print(f"Using percentile threshold: {percentile_threshold}% for anomaly detection")
            
            # Step 2: Identify anomalies in training data using model
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
                
            # Step 3: Remove or downweight anomalies
            refined_train_data = self._refine_training_data(current_train_data, anomalies)
            
            # Step 4: Retrain on refined data
            print(f"Retraining on refined data ({len(refined_train_data)} station-years)...")
            model, history = self.registry.train_global_model(refined_train_data, validation_data)
            
            # Update for next iteration
            current_train_data = refined_train_data
            
            # Summary of this iteration
            print(f"\nCompleted iteration {iteration+1}:")
            print(f"  Anomalies found this iteration: {len(anomalies)}")
            print(f"  Total anomalies found so far: {total_anomalies_found}")
            print(f"  Remaining iterations: {self.max_iterations - (iteration+1)}")
        
        print("\nIterative training complete.")
        print(f"Total iterations performed: {iteration+1}")
        total_anomalies = sum(len(a) for a in anomalies_per_iteration)
        print(f"Total anomalies identified and handled: {total_anomalies}")
        
        # Save the final model as the global model
        self.registry.save_model(model, "global")
        
        return model, history, anomalies_per_iteration, current_train_data
    
    def _find_anomalies_in_training(self, model, train_data, percentile_threshold=95):
        """
        Find anomalies in training data using primarily prediction error.
        
        Args:
            model: Trained model
            train_data: Training data dictionary
            percentile_threshold: Percentile to use for error threshold (default: 95)
            
        Returns:
            List of anomalies with metadata
        """
        anomalies = []
        model_type = getattr(model, 'model_type', self.config.get('model_type', 'autoencoder'))
        
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
                
                # Create a temporary structure for evaluation
                eval_data = {station_key: {'vst_raw_modified': raw_df}}
                
                # Get predictions and errors based on model type
                if model_type == 'forecaster':
                    from _3_lstm_model.lstm_forecaster import evaluate_forecaster
                    results = evaluate_forecaster(model, eval_data, {}, self.config)
                else:
                    results = evaluate_realistic(model, eval_data, {}, self.config)
                
                if station_key not in results:
                    continue
                    
                result = results[station_key]
                
                # Get errors based on model type
                if model_type == 'forecaster':
                    if 'prediction_errors' not in result:
                        continue
                    errors = result['prediction_errors']
                    timestamps = result['timestamps']
                    predictions = result['predictions'] 
                    original_values = result['actual_values']
                else:
                    if 'reconstruction_errors' not in result:
                        continue
                    errors = result['reconstruction_errors']
                    timestamps = result['timestamps']
                    reconstructions = result['reconstructions']
                    original_values = raw_df['Value'].values
                
                # Use a higher percentile threshold to be more selective
                error_threshold = np.percentile(errors, percentile_threshold)
                
                # Calculate statistical metrics
                window_size = 48  # 12 hours with 15-min data
                values_series = pd.Series(original_values)
                rolling_mean = values_series.rolling(window=window_size, center=True).mean()
                rolling_std = values_series.rolling(window=window_size, center=True).std()
                
                # Process each point for anomaly detection
                for i, (ts, error) in enumerate(zip(timestamps, errors)):
                    # Skip if we don't have enough context
                    if i < window_size//2 or i >= len(original_values) - window_size//2:
                        continue
                    
                    # Start with a score of 0
                    anomaly_score = 0
                    anomaly_types = []
                    
                    # Add to score based on error (0-5 points)
                    if error > error_threshold:
                        # Calculate how many standard deviations above threshold
                        error_z = (error - np.mean(errors)) / (np.std(errors) + 1e-10)
                        # Score from 1-5 based on how extreme the error is
                        error_score = min(5, 1 + int(error_z))
                        anomaly_score += error_score
                        anomaly_types.append('prediction' if model_type == 'forecaster' else 'reconstruction')
                    
                    # Add to score based on statistical outlier (0-3 points)
                    if i < len(rolling_mean) and i < len(rolling_std) and rolling_std[i] > 0:
                        z_score = abs(original_values[i] - rolling_mean[i]) / rolling_std[i]
                        if z_score > 3.0:  # Only consider significant outliers
                            # Score from 1-3 based on how extreme the z-score is
                            stat_score = min(3, int(z_score / 2))
                            anomaly_score += stat_score
                            anomaly_types.append('statistical')
                    
                    # Only flag as anomaly if confidence score is high enough
                    if anomaly_score >= 4:  # Minimum threshold for anomaly detection
                        anomalies.append({
                            'station_key': station_key,
                            'timestamp': ts,
                            'index': i,
                            'error': error,
                            'error_threshold': error_threshold,
                            'confidence_score': anomaly_score,
                            'original_value': original_values[i],
                            'predicted_value': predictions[i] if model_type == 'forecaster' and i < len(predictions) else None,
                            'reconstructed_value': reconstructions[i] if model_type != 'forecaster' and i < len(reconstructions) else None,
                            'local_mean': rolling_mean[i] if i < len(rolling_mean) else None,
                            'local_std': rolling_std[i] if i < len(rolling_std) else None,
                            'detection_methods': anomaly_types
                        })
                
                station_anomalies = sum(1 for a in anomalies if a['station_key'] == station_key)
                print(f"  {station_key}: Found {station_anomalies} anomalies")
                
                # Limit maximum percentage of points flagged as anomalies
                if station_anomalies > 0:
                    max_anomaly_percentage = 0.05  # Max 5% of data can be anomalies
                    max_anomalies = int(len(timestamps) * max_anomaly_percentage)
                    
                    if station_anomalies > max_anomalies:
                        print(f"    Too many anomalies detected ({station_anomalies}), limiting to top {max_anomalies}")
                        # Sort anomalies by confidence score and keep only the top ones
                        station_anomalies_list = [a for a in anomalies if a['station_key'] == station_key]
                        station_anomalies_list.sort(key=lambda x: x['confidence_score'], reverse=True)
                        
                        # Remove excess anomalies
                        anomalies = [a for a in anomalies if a['station_key'] != station_key]
                        anomalies.extend(station_anomalies_list[:max_anomalies])
                        
                        print(f"    Kept {max_anomalies} highest-confidence anomalies")
                    
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