"""
Evaluation metrics for anomaly detection performance.
Provides comprehensive metrics for synthetic error detection across multiple thresholds.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .z_score import calculate_z_scores_mad

@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""
    threshold: float
    
    # Event-level metrics (per error period)
    events_detected: int
    total_events: int
    event_detection_rate: float
    
    # Point-level metrics (per time point)
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Derived metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    # Coverage metrics (for variable-length errors)
    avg_coverage: float  # Average % of each error period detected
    avg_precision_in_period: float  # Average % of detections within error periods
    
    # Error type breakdown
    error_type_metrics: Dict[str, Dict] = None

class AnomalyEvaluator:
    """Evaluates anomaly detection performance against ground truth error periods."""
    
    def __init__(self, thresholds: List[float] = [2.0, 2.5, 3.0]):
        """
        Initialize evaluator with multiple thresholds to test.
        
        Args:
            thresholds: List of z-score thresholds to evaluate
        """
        self.thresholds = thresholds
        
    def evaluate_detection_performance(self, 
                                     val_data: pd.DataFrame,
                                     predictions: np.ndarray,
                                     error_periods: List,
                                     window_size: int = 96) -> Dict[float, DetectionMetrics]:
        """
        Evaluate detection performance across multiple thresholds.
        
        Args:
            val_data: Validation data with potential synthetic errors
            predictions: Model predictions
            error_periods: List of ErrorPeriod objects from SyntheticErrorGenerator
            window_size: Window size for z-score calculation
            
        Returns:
            Dictionary mapping threshold -> DetectionMetrics
        """
        results = {}
        
        print(f"\nEvaluating detection performance across {len(self.thresholds)} thresholds...")
        print(f"Found {len(error_periods)} injected error periods to evaluate against")
        
        for threshold in self.thresholds:
            print(f"\nEvaluating threshold {threshold}...")
            
            # Calculate z-scores and anomalies for this threshold
            z_scores, detected_anomalies = calculate_z_scores_mad(
                val_data['vst_raw'].values,
                predictions,
                window_size=window_size,
                threshold=threshold
            )
            
            # Calculate metrics for this threshold
            metrics = self._calculate_metrics(
                val_data=val_data,
                detected_anomalies=detected_anomalies,
                error_periods=error_periods,
                threshold=threshold
            )
            
            results[threshold] = metrics
            
            # Print summary for this threshold
            print(f"  Event Detection Rate: {metrics.event_detection_rate:.2%} ({metrics.events_detected}/{metrics.total_events})")
            print(f"  Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1_score:.3f}")
            print(f"  Average Coverage: {metrics.avg_coverage:.2%}")
        
        return results
    
    def _calculate_metrics(self, 
                          val_data: pd.DataFrame,
                          detected_anomalies: np.ndarray,
                          error_periods: List,
                          threshold: float) -> DetectionMetrics:
        """Calculate comprehensive metrics for a single threshold."""
        
        # Create ground truth mask for all error periods
        ground_truth_mask = self._create_ground_truth_mask(val_data, error_periods)
        
        # Event-level metrics
        events_detected, total_events = self._calculate_event_metrics(
            val_data, detected_anomalies, error_periods
        )
        event_detection_rate = events_detected / total_events if total_events > 0 else 0.0
        
        # Point-level confusion matrix
        true_positives = np.sum(ground_truth_mask & detected_anomalies)
        false_positives = np.sum(~ground_truth_mask & detected_anomalies)
        true_negatives = np.sum(~ground_truth_mask & ~detected_anomalies)
        false_negatives = np.sum(ground_truth_mask & ~detected_anomalies)
        
        # Derived metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (true_positives + true_negatives) / len(detected_anomalies)
        
        # Coverage metrics
        avg_coverage, avg_precision_in_period = self._calculate_coverage_metrics(
            val_data, detected_anomalies, error_periods
        )
        
        # Error type breakdown
        error_type_metrics = self._calculate_error_type_metrics(
            val_data, detected_anomalies, error_periods
        )
        
        return DetectionMetrics(
            threshold=threshold,
            events_detected=events_detected,
            total_events=total_events,
            event_detection_rate=event_detection_rate,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            avg_coverage=avg_coverage,
            avg_precision_in_period=avg_precision_in_period,
            error_type_metrics=error_type_metrics
        )
    
    def _create_ground_truth_mask(self, val_data: pd.DataFrame, error_periods: List) -> np.ndarray:
        """Create boolean mask indicating true error periods."""
        ground_truth = np.zeros(len(val_data), dtype=bool)
        
        for period in error_periods:
            # Find indices for this error period
            period_mask = (val_data.index >= period.start_time) & (val_data.index <= period.end_time)
            ground_truth |= period_mask
            
        return ground_truth
    
    def _calculate_event_metrics(self, val_data: pd.DataFrame, detected_anomalies: np.ndarray, error_periods: List) -> Tuple[int, int]:
        """Calculate event-level detection metrics."""
        events_detected = 0
        total_events = len(error_periods)
        
        for period in error_periods:
            # Find indices for this error period
            period_mask = (val_data.index >= period.start_time) & (val_data.index <= period.end_time)
            period_indices = np.where(period_mask)[0]
            
            if len(period_indices) == 0:
                continue
                
            # Check if any point in this period was detected as anomalous
            if np.any(detected_anomalies[period_indices]):
                events_detected += 1
        
        return events_detected, total_events
    
    def _calculate_coverage_metrics(self, val_data: pd.DataFrame, detected_anomalies: np.ndarray, error_periods: List) -> Tuple[float, float]:
        """Calculate coverage and precision metrics for variable-length errors."""
        if not error_periods:
            return 0.0, 0.0
            
        coverage_scores = []
        precision_scores = []
        
        for period in error_periods:
            # Find indices for this error period
            period_mask = (val_data.index >= period.start_time) & (val_data.index <= period.end_time)
            period_indices = np.where(period_mask)[0]
            
            if len(period_indices) == 0:
                continue
            
            # Coverage: What % of the error period was detected?
            detected_in_period = np.sum(detected_anomalies[period_indices])
            coverage = detected_in_period / len(period_indices)
            coverage_scores.append(coverage)
            
            # Precision within period: What % of detections in this period are correct?
            # (For true error periods, all detections within the period are correct)
            if detected_in_period > 0:
                precision_scores.append(1.0)  # All detections within true error period are correct
            
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        avg_precision_in_period = np.mean(precision_scores) if precision_scores else 0.0
        
        return avg_coverage, avg_precision_in_period
    
    def _calculate_error_type_metrics(self, val_data: pd.DataFrame, detected_anomalies: np.ndarray, error_periods: List) -> Dict[str, Dict]:
        """Calculate metrics broken down by error type."""
        error_type_metrics = {}
        
        # Group error periods by type
        error_types = {}
        for period in error_periods:
            error_type = period.error_type
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(period)
        
        # Calculate metrics for each error type
        for error_type, periods in error_types.items():
            events_detected = 0
            total_events = len(periods)
            coverage_scores = []
            
            for period in periods:
                # Find indices for this error period
                period_mask = (val_data.index >= period.start_time) & (val_data.index <= period.end_time)
                period_indices = np.where(period_mask)[0]
                
                if len(period_indices) == 0:
                    continue
                
                # Event detection
                if np.any(detected_anomalies[period_indices]):
                    events_detected += 1
                
                # Coverage
                detected_in_period = np.sum(detected_anomalies[period_indices])
                coverage = detected_in_period / len(period_indices)
                coverage_scores.append(coverage)
            
            event_detection_rate = events_detected / total_events if total_events > 0 else 0.0
            avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
            
            error_type_metrics[error_type] = {
                'events_detected': events_detected,
                'total_events': total_events,
                'event_detection_rate': event_detection_rate,
                'avg_coverage': avg_coverage
            }
        
        return error_type_metrics 