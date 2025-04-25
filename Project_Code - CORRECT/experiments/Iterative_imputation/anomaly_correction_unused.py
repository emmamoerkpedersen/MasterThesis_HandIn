"""
Anomaly correction utilities for water level data.

This module implements a three-pass approach to detect and correct anomalies in water level data:
1. First Pass: Generate accurate LSTM predictions (may include anomalies)
2. Second Pass: Use Isolation Forest to identify anomalous periods
3. Third Pass: Generate corrected predictions by replacing anomalous periods
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import IsolationForest

from experiments.Iterative_imputation.anomaly_detection import detect_anomalies_from_residuals
from _3_lstm_model.feature_engineering import FeatureEngineer
from experiments.Iterative_imputation.visualization_utils import (
    visualize_correction_results,
    visualize_correction_segments,
    visualize_anomalies
)

class AnomalyCorrector:
    def __init__(self, model, preprocessor, config):
        """
        Initialize the anomaly corrector.
        
        Args:
            model: Trained LSTM model
            preprocessor: DataPreprocessor instance
            config: Configuration dictionary
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
    def detect_and_correct_anomalies(self, data, contamination=0.05, method='isolation_forest',
                                    window_size=24, smoothing_window=5, output_path=None):
        """
        Execute the three-pass approach for anomaly detection and correction.
        
        Args:
            data: DataFrame containing time series data with original water level values
            contamination: Expected proportion of anomalies in the data
            method: Anomaly detection method (default: 'isolation_forest')
            window_size: Window size for correction (in hours)
            smoothing_window: Window size for residual smoothing
            output_path: Path to save visualizations
            
        Returns:
            DataFrame with original values, predictions, anomaly flags, and corrected values
        """
        print("\n--- THREE-PASS ANOMALY DETECTION AND CORRECTION ---")
        
        # FIRST PASS: Generate predictions using the LSTM model
        print("\nPASS 1: Generating initial LSTM predictions...")
        predictions, _, _ = self.model.predict(data)
        
        # Flatten predictions to match data length if needed
        predictions_flat = np.array(predictions).flatten()
        if len(predictions_flat) > len(data):
            predictions_flat = predictions_flat[:len(data)]
        elif len(predictions_flat) < len(data):
            # Pad with NaN if predictions are shorter
            padding = np.full(len(data) - len(predictions_flat), np.nan)
            predictions_flat = np.concatenate([predictions_flat, padding])
        
        # Create predictions Series with same index as data
        predictions_series = pd.Series(predictions_flat, index=data.index)
        actual_series = data['vst_raw']
        
        # SECOND PASS: Detect anomalies using residuals
        print(f"\nPASS 2: Detecting anomalies using {method}...")
        anomaly_results = detect_anomalies_from_residuals(
            actual_series,
            predictions_series,
            method=method,
            contamination=contamination,
            apply_smoothing=True,
            smoothing_window=smoothing_window
        )
        
        print(f"Detected {anomaly_results['anomaly'].sum()} potential anomalies "
              f"({(anomaly_results['anomaly'].sum() / len(anomaly_results) * 100):.2f}% of data)")
        
        # THIRD PASS: Generate corrected predictions by modifying lagged inputs during anomalous periods
        print("\nPASS 3: Generating corrected predictions...")
        corrected_data = self._generate_corrected_predictions(
            data, 
            anomaly_results['anomaly'],
            predictions_series,
            window_size
        )
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'actual': actual_series,
            'predicted': predictions_series,
            'anomaly': anomaly_results['anomaly'],
            'corrected': corrected_data['vst_raw_corrected']
        })
        
        # Generate visualizations
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate correction results visualization
            visualize_correction_results(result_df, output_dir)
            
            # Generate detailed segment visualizations
            segments = self._identify_anomaly_segments(result_df['anomaly'])
            visualize_correction_segments(result_df, segments, output_dir)
        
        return result_df
    
    def _generate_corrected_predictions(self, data, anomaly_flags, predictions, window_size=24):
        """
        Generate corrected predictions by replacing anomalous inputs with predicted values
        for subsequent predictions.
        
        Args:
            data: Original data DataFrame
            anomaly_flags: Series of 0/1 indicating anomalies
            predictions: Series of initial LSTM predictions
            window_size: Window size for correction
            
        Returns:
            DataFrame with corrected values
        """
        # Create a copy of the data to avoid modifying the original
        corrected_data = data.copy()
        
        # Add a column for corrected values (initially same as original)
        corrected_data['vst_raw_corrected'] = corrected_data['vst_raw'].copy()
        
        # First, identify continuous segments of anomalies
        anomaly_segments = self._identify_anomaly_segments(anomaly_flags)
        print(f"Identified {len(anomaly_segments)} anomalous segments to correct")
        
        # If we have lagged features in the model, we'll need to modify them 
        # to avoid propagating the anomalies
        lag_feature_present = False
        lag_features = []
        for col in self.preprocessor.feature_cols:
            if 'water_level_lag' in col:
                lag_feature_present = True
                lag_features.append(col)
        
        # If lag features are being used
        if lag_feature_present and lag_features:
            print(f"Found {len(lag_features)} lag features: {lag_features}")
            
            # For each anomaly segment, correct values and update lag features
            for segment_start, segment_end in anomaly_segments:
                segment_indices = corrected_data.index[segment_start:segment_end+1]
                print(f"Processing anomaly segment from {segment_indices[0]} to {segment_indices[-1]} "
                      f"({len(segment_indices)} points)")
                
                # Replace anomalous values with predicted values
                corrected_data.loc[segment_indices, 'vst_raw_corrected'] = predictions.loc[segment_indices]
                
                # If lag features exist, we need to update them with corrected values for future predictions
                if lag_features:
                    # Extend the correction window by the longest lag to properly update lag features
                    max_lag = max([int(f.split('_')[-1].replace('h', '')) for f in lag_features])
                    extension_end = min(segment_end + max_lag + 1, len(corrected_data))
                    
                    # For points after the anomaly segment, recalculate lag features using corrected values
                    if extension_end > segment_end + 1:
                        extension_indices = corrected_data.index[segment_end+1:extension_end]
                        
                        # Update lag features using corrected values
                        for lag_feature in lag_features:
                            lag_hours = int(lag_feature.split('_')[-1].replace('h', ''))
                            
                            # For each point in the extension, recalculate the lag feature
                            for i, idx in enumerate(extension_indices):
                                lag_idx = corrected_data.index.get_indexer([idx - pd.Timedelta(hours=lag_hours)])
                                
                                if lag_idx[0] >= 0:  # If the lagged index exists
                                    corrected_data.loc[idx, lag_feature] = corrected_data.iloc[lag_idx[0]]['vst_raw_corrected']
        else:
            # Just replace anomalous values with predicted values
            for segment_start, segment_end in anomaly_segments:
                segment_indices = corrected_data.index[segment_start:segment_end+1]
                corrected_data.loc[segment_indices, 'vst_raw_corrected'] = predictions.loc[segment_indices]
                
        return corrected_data
    
    def _identify_anomaly_segments(self, anomaly_flags):
        """
        Identify continuous segments of anomalies.
        
        Args:
            anomaly_flags: Series of 0/1 indicating anomalies
            
        Returns:
            List of tuples (start_idx, end_idx) for each anomaly segment
        """
        segments = []
        in_segment = False
        start_idx = None
        
        for i, flag in enumerate(anomaly_flags):
            if flag == 1 and not in_segment:
                # Start of a new segment
                in_segment = True
                start_idx = i
            elif flag == 0 and in_segment:
                # End of a segment
                segments.append((start_idx, i-1))
                in_segment = False
                
        # Handle case where the last segment extends to the end
        if in_segment:
            segments.append((start_idx, len(anomaly_flags)-1))
            
        return segments

def run_anomaly_correction_pipeline(model, preprocessor, config, test_data, output_path=None,
                                   contamination=0.05, method='isolation_forest'):
    """
    Run the complete anomaly correction pipeline.
    
    Args:
        model: Trained LSTM model
        preprocessor: DataPreprocessor instance
        config: Configuration dictionary
        test_data: DataFrame with test data
        output_path: Path to save results
        contamination: Expected proportion of anomalies
        method: Anomaly detection method
        
    Returns:
        DataFrame with original values, predictions, anomaly flags, and corrected values
    """
    # Initialize anomaly corrector
    corrector = AnomalyCorrector(model, preprocessor, config)
    
    # Detect and correct anomalies
    results = corrector.detect_and_correct_anomalies(
        test_data,
        contamination=contamination,
        method=method,
        output_path=output_path
    )
    
    # Calculate metrics for original and corrected data
    from utils.pipeline_utils import calculate_performance_metrics
    
    # Calculate original metrics
    original_predictions = results['predicted'].values
    original_mask = ~np.isnan(original_predictions)
    original_metrics = calculate_performance_metrics(
        results['actual'].values,
        original_predictions,
        original_mask
    )
    
    # Calculate corrected metrics
    corrected_predictions = results['corrected'].values
    corrected_mask = ~np.isnan(corrected_predictions)
    corrected_metrics = calculate_performance_metrics(
        results['actual'].values,
        corrected_predictions,
        corrected_mask
    )
    
    # Print comparison
    print("\n--- METRICS COMPARISON ---")
    print("Original Predictions:")
    for metric, value in original_metrics.items():
        print(f"  {metric}: {value}")
    
    print("\nCorrected Predictions:")
    for metric, value in corrected_metrics.items():
        print(f"  {metric}: {value}")
    
    # Calculate how many values were corrected
    num_corrected = (results['actual'] != results['corrected']).sum()
    print(f"\nTotal corrected values: {num_corrected} ({num_corrected/len(results)*100:.2f}%)")
    
    return results, original_metrics, corrected_metrics 