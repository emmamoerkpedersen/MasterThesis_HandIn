"""
Iterative anomaly correction utilities for water level data.

This module implements an iterative approach to detect and correct anomalies in water level data:
1. Train LSTM model on available data
2. Identify anomalies using Isolation Forest on residuals
3. Replace anomalous points with model predictions
4. Re-train model on partially corrected data
5. Repeat until convergence

This approach aims to produce a "cleaned" version of the data where anomalies
are replaced with values that represent what the data would look like without anomalies.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import copy
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from experiments.Iterative_imputation.anomaly_detection import detect_anomalies_from_residuals
from _3_lstm_model.feature_engineering import FeatureEngineer
from experiments.Improved_model_structure.train_model import LSTM_Trainer
from utils.model_utils import create_lstm_model, train_model
from experiments.Improved_model_structure.model import LSTMModel
from experiments.Iterative_imputation.visualization_utils import visualize_all_results

# Custom version of anomaly detection that handles NaN values
def detect_anomalies_with_nan_handling(actual, predicted, contamination=0.05, magnitude_threshold=0.5, direction_bias=0):
    """
    Detect anomalies in time series data while handling NaN values.
    Uses Isolation Forest as the primary method for anomaly detection.
    
    Args:
        actual: Series with actual values
        predicted: Series with predicted values
        contamination: Expected proportion of anomalies
        magnitude_threshold: Minimum relative deviation to consider as anomaly (as fraction of standard deviation)
        direction_bias: Focus on specific direction of anomalies (-1 for drops, 1 for spikes, 0 for both)
        
    Returns:
        DataFrame with actual, predicted, and anomaly indicators
    """
    # Calculate residuals
    residuals = actual - predicted
    
    # Create a mask for valid (non-NaN) data points
    valid_mask = ~np.isnan(residuals)
    
    # Extract valid data for anomaly detection
    valid_residuals = residuals[valid_mask].values.reshape(-1, 1)
    
    if len(valid_residuals) == 0:
        # Handle case with no valid data
        return pd.DataFrame({
            'actual': actual,
            'predicted': predicted,
            'residual': residuals,
            'anomaly': pd.Series(np.zeros(len(actual)), index=actual.index)
        })
    
    # Create result Series with same index as input
    result = pd.Series(np.zeros(len(actual)), index=actual.index)
    
    # Fit Isolation Forest on valid data
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    
    # Predict anomalies
    anomaly_labels = iso_forest.fit_predict(valid_residuals)
    
    # Convert to 0/1 indicator (1 for anomalies)
    anomaly_indicator = np.where(anomaly_labels == -1, 1, 0)
    
    # Apply magnitude threshold to filter out small deviations
    residual_std = np.std(valid_residuals)
    significant_deviation_mask = np.abs(valid_residuals.flatten()) >= magnitude_threshold * residual_std
    
    # Apply direction bias if specified
    if direction_bias < 0:
        # Focus on drops (negative residuals - actual is lower than predicted)
        significant_deviation_mask = significant_deviation_mask & (valid_residuals.flatten() < 0)
    elif direction_bias > 0:
        # Focus on spikes (positive residuals - actual is higher than predicted)
        significant_deviation_mask = significant_deviation_mask & (valid_residuals.flatten() > 0)
    
    # Adjust anomalies based on significance of deviation
    filtered_anomalies = anomaly_indicator * significant_deviation_mask
    
    # Set the result values from the Isolation Forest approach
    result.loc[valid_mask] = filtered_anomalies
    
    # Connect nearby anomalies - if multiple points within a short window are anomalous,
    # the points between them are likely part of the same anomalous segment
    if np.any(result):
        # Convert to numpy for faster operations
        result_np = result.values
        
        # Use a window to connect nearby anomalies
        connect_window = 3
        for i in range(len(result_np) - connect_window):
            if result_np[i] and result_np[i + connect_window]:
                result_np[i:i + connect_window + 1] = True
        
        # Convert back to Series
        result = pd.Series(result_np, index=result.index)
        
        # Print statistics about anomalies detected
        anomaly_count = result.sum()
        print(f"  Detected {anomaly_count} anomalies ({anomaly_count/len(result)*100:.2f}%)")
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'residual': residuals,
        'anomaly': result.astype(int)  # Ensure anomaly column is int type, not boolean
    })
    
    return result_df

class IterativeAnomalyCorrector:
    def __init__(self, config, preprocessor):
        """
        Initialize the iterative anomaly corrector.
        
        Args:
            config: Configuration dictionary
            preprocessor: DataPreprocessor instance
        """
        self.config = config
        self.preprocessor = preprocessor
        self.feature_engineer = FeatureEngineer(config)
        
    def iterative_correction(self, train_data, val_data, test_data, 
                           max_iterations=5, 
                           convergence_threshold=0.01,
                           contamination=0.05, 
                           smoothing_window=5,
                           magnitude_threshold=0.5,
                           direction_bias=-1,
                           anomaly_types=None,
                           output_path=None):
        """
        Execute iterative anomaly detection and correction approach.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            test_data: Test data DataFrame to correct
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for determining convergence
            contamination: Expected proportion of anomalies
            smoothing_window: Window size for smoothing corrected segments
            magnitude_threshold: Minimum relative deviation to consider as anomaly
            direction_bias: Focus on specific direction of anomalies (-1 for drops, 0 for both, 1 for spikes)
            anomaly_types: List of anomaly types to detect (for compatibility, no longer used)
            output_path: Path to save visualizations
            
        Returns:
            DataFrame with original and corrected values
        """
        print("\n--- ITERATIVE ANOMALY CORRECTION ---")
        print(f"Contamination: {contamination}, Magnitude threshold: {magnitude_threshold}")
        print(f"Direction bias: {direction_bias}, Smoothing window: {smoothing_window}")
        
        # Create output directory if needed
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Make a copy of the data to avoid modifying the original
        corrected_train = train_data.copy()
        corrected_val = val_data.copy()
        corrected_test = test_data.copy()
        
        # Add a column for corrected values (initially same as original)
        for df in [corrected_train, corrected_val, corrected_test]:
            df['vst_raw_corrected'] = df['vst_raw'].copy()
        
        # Store history of corrected values for test data
        correction_history = []
        correction_history.append(corrected_test['vst_raw'].copy())
        
        # Store anomaly masks for visualization
        anomaly_masks = []
        
        # Run iterative process
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration+1}/{max_iterations} ===")
            
            # 1. Train model on current corrected data
            print(f"Training model on {'corrected' if iteration > 0 else 'original'} data...")
            
            # Use corrected value as the target variable
            orig_output_feature = self.config['output_features'][0]
            self.config['output_features'] = ['vst_raw_corrected']
            
            # Create a new model for this iteration
            input_size = len(self.preprocessor.feature_cols)
            model = create_lstm_model(input_size, self.config, LSTMModel)
            print(f"Feature columns: {self.preprocessor.feature_cols}")
            print(f"Output feature: {self.config['output_features'][0]}")
            
            # Initialize trainer
            trainer = LSTM_Trainer(self.config, preprocessor=self.preprocessor)
            
            # Use smaller number of epochs for intermediate iterations
            iter_config = copy.deepcopy(self.config)
            if iteration < max_iterations - 1:
                iter_config['epochs'] = max(3, self.config['epochs'] // 3)
                
            # Train the model
            history, _, _ = train_model(trainer, corrected_train, corrected_val, iter_config)
            
            # 2. Make predictions on the test data
            print(f"Making predictions on test data...")
            test_predictions, _, _ = trainer.predict(corrected_test)
            
            # Flatten predictions to match data length if needed
            predictions_flat = np.array(test_predictions).flatten()
            if len(predictions_flat) > len(corrected_test):
                predictions_flat = predictions_flat[:len(corrected_test)]
            elif len(predictions_flat) < len(corrected_test):
                # Pad with NaN if predictions are shorter
                padding = np.full(len(corrected_test) - len(predictions_flat), np.nan)
                predictions_flat = np.concatenate([predictions_flat, padding])
            
            # Create predictions Series with same index as data
            predictions_series = pd.Series(predictions_flat, index=corrected_test.index)
            
            # 3. Detect anomalies using residuals between original and predicted values
            print(f"Detecting anomalies between original and predicted values...")
            anomaly_results = detect_anomalies_with_nan_handling(
                corrected_test['vst_raw'],  # Always compare against original data
                predictions_series,
                contamination=contamination,
                magnitude_threshold=magnitude_threshold,
                direction_bias=direction_bias
            )
            
            # Store anomaly mask for this iteration
            anomaly_masks.append(anomaly_results['anomaly'].copy().astype(int))  # Ensure integer type
            
            # Count anomalies
            anomaly_count = anomaly_results['anomaly'].sum()
            print(f"Detected {anomaly_count} potential anomalies "
                 f"({(anomaly_count / len(anomaly_results)) * 100:.2f}% of data)")
            
            # 4. Replace anomalous points with predictions
            previous_corrected = corrected_test['vst_raw_corrected'].copy()
            
            # Apply corrections where anomalies are detected and predictions are not NaN
            anomaly_mask = (anomaly_results['anomaly'] == 1) & (~np.isnan(predictions_series))
            anomaly_indices = corrected_test.index[anomaly_mask]
            
            if len(anomaly_indices) > 0:
                # Identify distinct segments for better visualization
                segments = []
                if len(anomaly_indices) > 0:
                    current_segment = [anomaly_indices[0]]
                    for i in range(1, len(anomaly_indices)):
                        # Check if consecutive
                        if (anomaly_indices[i] - anomaly_indices[i-1]).total_seconds() <= 3600:  # Within 1 hour
                            current_segment.append(anomaly_indices[i])
                        else:
                            segments.append(current_segment)
                            current_segment = [anomaly_indices[i]]
                    
                    # Add the last segment
                    if current_segment:
                        segments.append(current_segment)
                
                # Log information about segments
                print(f"Identified {len(segments)} anomalous segments")
                for i, segment in enumerate(segments[:3]):  # Show details for first 3 segments only
                    start_time = segment[0]
                    end_time = segment[-1]
                    duration = (end_time - start_time).total_seconds() / 3600  # hours
                    
                    # Calculate average correction magnitude for this segment
                    original_vals = corrected_test.loc[segment, 'vst_raw'].values
                    predicted_vals = predictions_series.loc[segment].values
                    mean_diff = (predicted_vals - original_vals).mean()
                    relative_diff = (mean_diff / np.abs(original_vals.mean())) * 100 if np.abs(original_vals.mean()) > 0 else 0
                    
                    print(f"  Segment {i+1}: {len(segment)} points ({duration:.1f} hours), "
                          f"avg correction: {mean_diff:.1f} ({relative_diff:.1f}%)")
                
                if len(segments) > 3:
                    print(f"  ... and {len(segments) - 3} more segments")
            
                # Apply corrections directly to detected anomalies
                corrected_test.loc[anomaly_indices, 'vst_raw_corrected'] = predictions_series.loc[anomaly_indices]
            
            # 5. Apply smoothing to corrected segments to ensure natural transitions
            if smoothing_window and smoothing_window > 1 and len(anomaly_indices) > 0:
                print(f"Applying smoothing with window size {smoothing_window}...")
                corrected_test['vst_raw_corrected'] = self._smooth_corrected_segments(
                    corrected_test['vst_raw_corrected'],
                    anomaly_mask,
                    smoothing_window
                )
            
            # 6. Also process training data if needed
            if iteration > 0:
                # Make predictions on training data
                print("Making predictions on training data...")
                train_predictions, _, _ = trainer.predict(corrected_train)
                
                # Process predictions
                train_preds_flat = np.array(train_predictions).flatten()
                if len(train_preds_flat) > len(corrected_train):
                    train_preds_flat = train_preds_flat[:len(corrected_train)]
                elif len(train_preds_flat) < len(corrected_train):
                    padding = np.full(len(corrected_train) - len(train_preds_flat), np.nan)
                    train_preds_flat = np.concatenate([train_preds_flat, padding])
                
                train_preds_series = pd.Series(train_preds_flat, index=corrected_train.index)
                
                # Detect anomalies in training data
                print("Detecting anomalies in training data...")
                train_anomaly_results = detect_anomalies_with_nan_handling(
                    corrected_train['vst_raw'],
                    train_preds_series,
                    contamination=contamination,
                    magnitude_threshold=magnitude_threshold,
                    direction_bias=direction_bias
                )
                
                # Apply corrections to training data where anomalies are detected and predictions are not NaN
                train_anomaly_mask = (train_anomaly_results['anomaly'] == 1) & (~np.isnan(train_preds_series))
                train_anomaly_indices = corrected_train.index[train_anomaly_mask]
                train_anomaly_count = len(train_anomaly_indices)
                print(f"Detected {train_anomaly_count} potential anomalies in training data "
                     f"({(train_anomaly_count / len(train_anomaly_results)) * 100:.2f}% of training data)")
                
                if len(train_anomaly_indices) > 0:
                    corrected_train.loc[train_anomaly_indices, 'vst_raw_corrected'] = train_preds_series.loc[train_anomaly_indices]
                    
                    # Smooth corrected segments in training data
                    if smoothing_window and smoothing_window > 1:
                        corrected_train['vst_raw_corrected'] = self._smooth_corrected_segments(
                            corrected_train['vst_raw_corrected'],
                            train_anomaly_mask,
                            smoothing_window
                        )
            
            # Store this iteration's corrected values
            correction_history.append(corrected_test['vst_raw_corrected'].copy())
            
            # 7. Check for convergence
            if iteration > 0:
                # Calculate change ratio between current and previous corrected values
                changes = (corrected_test['vst_raw_corrected'] - previous_corrected).abs()
                valid_changes = changes[~np.isnan(changes)]
                max_change = valid_changes.max() if len(valid_changes) > 0 else 0
                mean_change = valid_changes.mean() if len(valid_changes) > 0 else 0
                print(f"Changes: Max = {max_change:.4f}, Mean = {mean_change:.4f}")
                
                if mean_change < convergence_threshold:
                    print(f"Convergence reached at iteration {iteration+1}. Mean change: {mean_change:.6f}")
                    break
        
        # Restore original output feature
        self.config['output_features'] = [orig_output_feature]
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'actual': test_data['vst_raw'],
            'corrected': corrected_test['vst_raw_corrected'],
            'anomaly': anomaly_results['anomaly']
        })
        
        # Generate visualizations
        if output_path:
            try:
                visualize_all_results(
                    original_data=test_data,
                    error_data=None,  # No synthetic errors in this case
                    correction_history=correction_history,
                    anomaly_masks=anomaly_masks,
                    results=result_df,
                    output_path=output_dir
                )
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")
        
        return result_df, correction_history, anomaly_masks
        
    def _smooth_corrected_segments(self, data_series, anomaly_mask, window_size):
        """
        Apply smoothing to anomalous segments and their boundaries.
        
        Args:
            data_series: Series containing the data to be smoothed
            anomaly_mask: Boolean Series indicating anomalous points
            window_size: The size of the rolling window for smoothing
            
        Returns:
            Series with smoothed anomalous segments
        """
        smoothed_series = data_series.copy()
        
        # Quick check if there are any anomalies to smooth
        if not anomaly_mask.any():
            return smoothed_series
        
        # Find indices where anomalies start and end
        anomaly_diff = anomaly_mask.astype(int).diff()
        start_indices = anomaly_diff[anomaly_diff == 1].index
        end_indices = anomaly_diff[anomaly_diff == -1].index
        
        # Handle edge cases where anomalies start/end at the series boundaries
        if anomaly_mask.iloc[0]:
            start_indices = start_indices.insert(0, data_series.index[0])
        if anomaly_mask.iloc[-1]:
            end_indices = end_indices.append(pd.Index([data_series.index[-1]]))
        
        # Check for valid segments
        if len(start_indices) != len(end_indices):
            print(f"Warning: Mismatch in anomaly segment start/end ({len(start_indices)} vs {len(end_indices)}). Skipping smoothing.")
            return smoothed_series
        
        # Process each segment
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = end_indices[i]
            
            # Extend the segment to include boundary points for better transitions
            boundary_extension = window_size
            
            segment_start_idx = data_series.index.get_loc(start)
            segment_end_idx = data_series.index.get_loc(end)
            
            # Create extended segment indices
            extended_start_idx = max(0, segment_start_idx - boundary_extension)
            extended_end_idx = min(len(data_series) - 1, segment_end_idx + boundary_extension)
            
            # Extract the extended segment
            extended_segment = data_series.iloc[extended_start_idx:extended_end_idx + 1].copy()
            
            # Skip if segment is too short for meaningful smoothing
            if len(extended_segment) < window_size:
                continue
            
            # Apply centered rolling mean
            smoothed_values = extended_segment.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            
            # Update original series with smoothed values for the anomaly segment only
            anomaly_segment_indices = data_series.loc[start:end].index
            smoothed_series.loc[anomaly_segment_indices] = smoothed_values.loc[anomaly_segment_indices]
            
            # Create smooth transitions at the boundaries
            # Blend at the start boundary if we're not at the beginning of the series
            if segment_start_idx > 0 and extended_start_idx < segment_start_idx:
                blend_indices = data_series.index[extended_start_idx:segment_start_idx]
                blend_length = len(blend_indices)
                
                if blend_length > 0:
                    for i, idx in enumerate(blend_indices):
                        # Linear weight from 0 to 1
                        weight = i / blend_length
                        # Weighted average between original and smoothed values
                        blended_val = (1 - weight) * data_series.loc[idx] + weight * smoothed_values.loc[idx]
                        smoothed_series.loc[idx] = blended_val
            
            # Blend at the end boundary if we're not at the end of the series
            if segment_end_idx < len(data_series) - 1 and segment_end_idx < extended_end_idx:
                blend_indices = data_series.index[segment_end_idx+1:extended_end_idx+1]
                blend_length = len(blend_indices)
                
                if blend_length > 0:
                    for i, idx in enumerate(blend_indices):
                        # Linear weight from 1 to 0
                        weight = 1 - (i / blend_length)
                        # Weighted average between original and smoothed values
                        blended_val = weight * smoothed_values.loc[idx] + (1 - weight) * data_series.loc[idx]
                        smoothed_series.loc[idx] = blended_val
        
        return smoothed_series
            
def run_iterative_correction_pipeline(config, preprocessor, train_data, val_data, test_data, 
                                   output_path=None, max_iterations=5, 
                                   contamination=0.05,
                                   smoothing_window=5,
                                   magnitude_threshold=0.5,
                                   direction_bias=-1,
                                   anomaly_types=None):
    """
    Run the complete iterative anomaly correction pipeline.
    
    Args:
        config: Model configuration
        preprocessor: DataPreprocessor instance
        train_data: Training data DataFrame
        val_data: Validation data DataFrame
        test_data: Test data DataFrame to correct
        output_path: Path to save results
        max_iterations: Maximum number of iterations
        contamination: Expected proportion of anomalies
        smoothing_window: Window size for smoothing corrected segments
        magnitude_threshold: Minimum relative deviation to consider as anomaly
        direction_bias: Focus on specific direction of anomalies (-1 for drops, 1 for spikes, 0 for both)
        anomaly_types: List of anomaly types to detect or None for default behavior
        
    Returns:
        DataFrame with original and corrected values
    """
    # Initialize the corrector
    corrector = IterativeAnomalyCorrector(config, preprocessor)
    
    # Run iterative correction
    results, correction_history, anomaly_masks = corrector.iterative_correction(
        train_data, 
        val_data, 
        test_data,
        max_iterations=max_iterations,
        contamination=contamination,
        smoothing_window=smoothing_window,
        magnitude_threshold=magnitude_threshold,
        direction_bias=direction_bias,
        anomaly_types=anomaly_types,
        output_path=output_path
    )
    
    # Calculate difference statistics
    differences = results['actual'] - results['corrected']
    anomaly_diffs = differences[results['anomaly'] == 1]
    
    # Handle NaN values in statistics
    differences_clean = differences.dropna()
    anomaly_diffs_clean = anomaly_diffs.dropna()
    
    diff_stats = {
        'mean_diff': differences_clean.mean() if len(differences_clean) > 0 else 0,
        'median_diff': differences_clean.median() if len(differences_clean) > 0 else 0,
        'max_diff': differences_clean.abs().max() if len(differences_clean) > 0 else 0,
        'anomaly_count': results['anomaly'].sum(),
        'anomaly_percentage': (results['anomaly'].sum() / len(results)) * 100,
        'mean_anomaly_diff': anomaly_diffs_clean.mean() if len(anomaly_diffs_clean) > 0 else 0,
        'max_anomaly_diff': anomaly_diffs_clean.abs().max() if len(anomaly_diffs_clean) > 0 else 0
    }
    
    # Print statistics
    print("\n--- CORRECTION STATISTICS ---")
    for stat, value in diff_stats.items():
        print(f"  {stat}: {value}")
    
    # Calculate RMSE between iterations to show convergence
    if len(correction_history) > 1:
        print("\n--- CONVERGENCE METRICS ---")
        for i in range(1, len(correction_history)):
            prev = correction_history[i-1]
            curr = correction_history[i]
            
            # Calculate RMSE only on non-NaN values
            valid_mask = ~np.isnan(prev) & ~np.isnan(curr)
            if valid_mask.sum() > 0:
                rmse = np.sqrt(((prev[valid_mask] - curr[valid_mask]) ** 2).mean())
                print(f"  RMSE between iterations {i-1} and {i}: {rmse:.6f}")
            else:
                print(f"  RMSE between iterations {i-1} and {i}: N/A (no valid data)")
    
    return results, diff_stats, correction_history, anomaly_masks

def identify_scaling_errors(actual, predicted, anomaly_mask):
    """
    Identify segments that are likely scaling errors rather than point anomalies.
    
    Args:
        actual: Series with actual values
        predicted: Series with predicted values
        anomaly_mask: Boolean mask of detected anomalies
        
    Returns:
        Dictionary with identified scaling segments and correction factors
    """
    # Find contiguous segments of anomalies
    anomaly_indices = actual.index[anomaly_mask]
    
    if len(anomaly_indices) == 0:
        return {}
    
    # Group into segments
    segments = []
    current_segment = [anomaly_indices[0]]
    for i in range(1, len(anomaly_indices)):
        # Check if consecutive (within 1 hour)
        if (anomaly_indices[i] - anomaly_indices[i-1]).total_seconds() <= 3600:
            current_segment.append(anomaly_indices[i])
        else:
            if len(current_segment) >= 10:  # Only include segments of meaningful length
                segments.append(current_segment)
            current_segment = [anomaly_indices[i]]
    
    # Add the last segment
    if current_segment and len(current_segment) >= 10:
        segments.append(current_segment)
    
    # Analyze each segment to check if it's a scaling error
    scaling_segments = {}
    
    for segment in segments:
        # Calculate relative error for this segment
        segment_actual = actual.loc[segment].values
        segment_predicted = predicted.loc[segment].values
        
        # Skip if we have NaN values
        if np.isnan(segment_actual).any() or np.isnan(segment_predicted).any():
            continue
        
        # Calculate relative error (actual/predicted)
        rel_errors = segment_actual / segment_predicted
        
        # A scaling error should have consistent relative error
        rel_error_mean = np.mean(rel_errors)
        rel_error_std = np.std(rel_errors)
        
        # If standard deviation is low compared to mean, it's likely a scaling error
        consistency_ratio = rel_error_std / abs(rel_error_mean) if rel_error_mean != 0 else float('inf')
        
        # Clear scaling error if:
        # 1. Relative error is consistent (low std deviation relative to mean)
        # 2. Mean relative error shows significant scaling (not close to 1.0)
        if consistency_ratio < 0.3 and abs(rel_error_mean - 1.0) > 0.15:
            # Calculate scaling factor (inverse of relative error)
            scaling_factor = 1.0 / rel_error_mean
            
            # Add to scaling segments
            scaling_segments[tuple(segment)] = {
                'factor': scaling_factor,
                'rel_error_mean': rel_error_mean,
                'rel_error_std': rel_error_std,
                'consistency_ratio': consistency_ratio,
                'length': len(segment)
            }
            
            print(f"  Identified scaling error: {len(segment)} points, factor: {scaling_factor:.2f}")
    
    return scaling_segments 