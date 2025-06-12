"""
Comprehensive evaluation and visualization for anomaly detection performance.
Provides integrated analysis across multiple thresholds with detailed visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns

from .evaluation_metrics import AnomalyEvaluator, DetectionMetrics
from .z_score import calculate_z_scores_mad


def extract_ground_truth_from_stations_results(stations_results, station_id, dataset_type='val'):
    """
    Extract ground truth flags from stations_results dictionary.
    
    Args:
        stations_results: Dictionary returned from inject_errors_into_dataset
        station_id: Station identifier
        dataset_type: Type of dataset ('train' or 'val')
        
    Returns:
        np.array: Boolean array indicating ground truth anomaly locations
    """
    # Try different possible keys for ground truth data
    possible_keys = [
        f"{station_id}_anomaly_{dataset_type}_vst_raw",  # New anomaly detection key
        f"{station_id}_{dataset_type}_vst_raw",          # Main target column
        f"{station_id}_{dataset_type}",                  # Fallback
    ]
    
    # Also try feature columns that might have been corrupted
    for key_name in list(stations_results.keys()):
        if f"{station_id}_{dataset_type}" in key_name or f"{station_id}_anomaly_{dataset_type}" in key_name:
            possible_keys.append(key_name)
    
    print(f"Looking for ground truth in stations_results with keys: {list(stations_results.keys())}")
    
    for key in possible_keys:
        if key in stations_results and 'ground_truth' in stations_results[key]:
            ground_truth_df = stations_results[key]['ground_truth']
            if 'error' in ground_truth_df.columns:
                ground_truth_flags = ground_truth_df['error'].values.astype(bool)
                print(f"✅ Extracted {np.sum(ground_truth_flags)} ground truth flags from key: {key}")
                return ground_truth_flags
    
    print(f"⚠️ No ground truth found for {station_id}_{dataset_type}")
    return None


def calculate_confusion_matrix_metrics(y_true, y_pred):
    """
    Calculate confusion matrix and related metrics for anomaly detection.
    
    Args:
        y_true: Ground truth binary labels (1 for anomaly, 0 for normal)
        y_pred: Predicted binary labels (1 for anomaly, 0 for normal)
        
    Returns:
        dict: Dictionary containing confusion matrix and metrics
    """
    # Convert to numpy arrays and handle NaN values
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    
    # Remove NaN entries if any exist
    valid_mask = ~(np.isnan(y_true.astype(float)) | np.isnan(y_pred.astype(float)))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    # Calculate confusion matrix components
    true_positives = np.sum((y_true == True) & (y_pred == True))
    false_positives = np.sum((y_true == False) & (y_pred == True))
    true_negatives = np.sum((y_true == False) & (y_pred == False))
    false_negatives = np.sum((y_true == True) & (y_pred == False))
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / len(y_true) if len(y_true) > 0 else 0.0
    
    # Create confusion matrix
    confusion_matrix = np.array([[true_negatives, false_positives],
                                [false_negatives, true_positives]])
    
    metrics = {
        'confusion_matrix': confusion_matrix,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'total_predictions': len(y_true),
        'total_anomalies_true': np.sum(y_true),
        'total_anomalies_pred': np.sum(y_pred)
    }
    
    return metrics


def print_anomaly_detection_results(metrics, threshold):
    """
    Print comprehensive anomaly detection results.
    
    Args:
        metrics: Dictionary containing confusion matrix and metrics
        threshold: Z-score threshold used for detection
    """
    print(f"\n{'='*60}")
    print(f"ANOMALY DETECTION RESULTS (Threshold: {threshold})")
    print(f"{'='*60}")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal     {metrics['true_negatives']:6d}   {metrics['false_positives']:6d}")
    print(f"Actual Anomaly    {metrics['false_negatives']:6d}   {metrics['true_positives']:6d}")
    
    print(f"\nDetection Metrics:")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    
    print(f"\nDetection Summary:")
    print(f"  Total data points:     {metrics['total_predictions']:,}")
    print(f"  True anomalies:        {metrics['total_anomalies_true']:,} ({metrics['total_anomalies_true']/metrics['total_predictions']*100:.2f}%)")
    print(f"  Predicted anomalies:   {metrics['total_anomalies_pred']:,} ({metrics['total_anomalies_pred']/metrics['total_predictions']*100:.2f}%)")
    print(f"  Correctly detected:    {metrics['true_positives']:,}")
    print(f"  False alarms:          {metrics['false_positives']:,}")
    print(f"  Missed anomalies:      {metrics['false_negatives']:,}")


def run_single_threshold_anomaly_detection(
    val_data: pd.DataFrame,
    predictions: np.ndarray,
    stations_results: Dict,
    station_id: str,
    config: Dict,
    output_dir: Path,
    original_val_data: pd.DataFrame = None,
    error_multiplier: float = None,
    filename_prefix: str = ""
) -> Dict:
    """
    Run anomaly detection using a single threshold from configuration.
    
    Args:
        val_data: Validation data with potential synthetic errors
        predictions: Model predictions
        stations_results: Results from inject_errors_into_dataset containing ground truth
        station_id: Station ID for labeling
        config: Anomaly detection configuration (needs 'window_size', 'threshold')
        output_dir: Directory to save results and plots
        original_val_data: Original validation data before error injection
        error_multiplier: Error multiplier used (for labeling)
        filename_prefix: Prefix for the filename of the generated plot
        
    Returns:
        Dictionary with anomaly detection results
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION using Z-Score MAD")
    print("="*60)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract configuration
    threshold = config['threshold']
    window_size = config['window_size']
    
    print(f"Configuration:")
    print(f"  Method: Z-Score MAD")
    print(f"  Window size: {window_size} points")
    print(f"  Threshold: {threshold}")
    
    # Extract ground truth flags from stations_results
    ground_truth_flags = extract_ground_truth_from_stations_results(
        stations_results, station_id, dataset_type='val'
    )
    
    if ground_truth_flags is None:
        print("❌ Cannot proceed without ground truth flags")
        return {'error': 'No ground truth available'}
    
    # Calculate z-scores and detect anomalies using MAD-based approach
    print(f"\nCalculating z-scores and detecting anomalies...")
    z_scores, detected_anomalies = calculate_z_scores_mad(
        val_data['vst_raw'].values,
        predictions,
        window_size=window_size,
        threshold=threshold
    )
    
    # Calculate confidence levels
    from .anomaly_visualization import calculate_anomaly_confidence
    confidence = calculate_anomaly_confidence(z_scores, threshold)
    
    # Count detections by confidence level
    high_conf = np.sum((detected_anomalies) & (confidence == 'High'))
    med_conf = np.sum((detected_anomalies) & (confidence == 'Medium'))
    low_conf = np.sum((detected_anomalies) & (confidence == 'Low'))
    total_detected = np.sum(detected_anomalies)
    
    print(f"\nDetection Summary:")
    print(f"  Total anomalies detected: {total_detected}")
    print(f"  High confidence: {high_conf}")
    print(f"  Medium confidence: {med_conf}")
    print(f"  Low confidence: {low_conf}")
    
    # Calculate confusion matrix and metrics
    confusion_metrics = calculate_confusion_matrix_metrics(ground_truth_flags, detected_anomalies)
    print_anomaly_detection_results(confusion_metrics, threshold)
    
    # Store results
    anomaly_results = {
        'z_scores': z_scores,
        'detected_anomalies': detected_anomalies,
        'confidence': confidence,
        'ground_truth': ground_truth_flags,
        'confusion_metrics': confusion_metrics,
        'config': config
    }
    
    print(f"\n✅ Anomaly detection completed successfully!")
    
    # Generate plot
    from shared.anomaly_detection.anomaly_visualization import plot_water_level_anomalies
    png_path, _ = plot_water_level_anomalies(
        test_data=val_data,
        predictions=predictions,
        z_scores=z_scores,
        anomalies=detected_anomalies,
        threshold=threshold,
        title=f"Anomaly Detection - Station {station_id} (Threshold: {threshold})",
        output_dir=output_dir,
        save_png=True,
        save_html=False,
        show_plot=False,
        filename_prefix=filename_prefix,
        confidence=confidence,
        original_data=original_val_data,
        ground_truth_flags=ground_truth_flags
    )
    
    return anomaly_results


def run_comprehensive_evaluation(
    val_data: pd.DataFrame,
    predictions: np.ndarray,
    error_generator,
    station_id: str,
    config: Dict,
    output_dir: Path,
    error_multiplier: float = None,
    original_val_data: pd.DataFrame = None
) -> Dict:
    """
    Run comprehensive anomaly detection evaluation with visualizations.
    
    Args:
        val_data: Validation data with potential synthetic errors
        predictions: Model predictions
        error_generator: SyntheticErrorGenerator with error_periods
        station_id: Station ID for labeling
        config: Model configuration (needs 'window_size', 'threshold')
        output_dir: Directory to save results and plots
        error_multiplier: Error multiplier used (for labeling)
        original_val_data: Original validation data before error injection
        
    Returns:
        Dictionary with evaluation results and file paths
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ANOMALY DETECTION EVALUATION")
    print("="*80)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator with comprehensive threshold range
    thresholds_to_test = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    evaluator = AnomalyEvaluator(thresholds=thresholds_to_test)
    
    # Run evaluation across all thresholds
    detection_results = evaluator.evaluate_detection_performance(
        val_data=val_data,
        predictions=predictions,
        error_periods=error_generator.error_periods,
        window_size=config['window_size']
    )
    
    # Save detailed results to CSV
    results_df = _create_results_dataframe(detection_results)
    results_csv_path = output_dir / f"detection_evaluation_results_{station_id}.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    # Print summary
    _print_evaluation_summary(results_df, config['threshold'])
    
    # Print error type breakdown
    _print_error_type_breakdown(detection_results, config['threshold'])
    
    # Generate comprehensive visualization
    viz_path = _create_comprehensive_visualization(
        results_df, station_id, config['threshold'], error_multiplier, output_dir
    )
    
    # Generate detailed threshold comparison
    comparison_path = _create_detailed_threshold_comparison(
        detection_results, station_id, error_multiplier, output_dir
    )
    
    # Save error type analysis
    error_type_path = _save_error_type_analysis(
        detection_results, config['threshold'], station_id, output_dir
    )
    
    # Generate plots for different thresholds
    threshold_plots = _create_threshold_series_plots(
        val_data=val_data,
        predictions=predictions,
        station_id=station_id,
        config=config,
        output_dir=output_dir,
        original_data=original_val_data,
        error_multiplier=error_multiplier
    )
    
    print(f"\nEvaluation results saved to:")
    print(f"  📊 Main results: {results_csv_path}")
    print(f"  📈 Visualization: {viz_path}")
    print(f"  📉 Detailed comparison: {comparison_path}")
    print(f"  📋 Error type analysis: {error_type_path}")
    print(f"  🎯 Threshold series plots: {len(threshold_plots)} plots")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return {
        'results_df': results_df,
        'detection_results': detection_results,
        'files': {
            'csv': results_csv_path,
            'main_plot': viz_path,
            'detailed_plot': comparison_path,
            'error_analysis': error_type_path,
            'threshold_plots': threshold_plots
        }
    }


def _create_results_dataframe(detection_results: Dict[float, DetectionMetrics]) -> pd.DataFrame:
    """Create comprehensive results DataFrame from detection results."""
    results_summary = []
    
    for threshold, metrics in detection_results.items():
        results_summary.append({
            'threshold': threshold,
            'event_detection_rate': metrics.event_detection_rate,
            'events_detected': metrics.events_detected,
            'total_events': metrics.total_events,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'accuracy': metrics.accuracy,
            'avg_coverage': metrics.avg_coverage,
            'avg_precision_in_period': metrics.avg_precision_in_period,
            'true_positives': metrics.true_positives,
            'false_positives': metrics.false_positives,
            'true_negatives': metrics.true_negatives,
            'false_negatives': metrics.false_negatives
        })
    
    return pd.DataFrame(results_summary)


def _print_evaluation_summary(results_df: pd.DataFrame, current_threshold: float):
    """Print formatted summary table of evaluation results."""
    print(f"\nSUMMARY OF DETECTION PERFORMANCE:")
    print(f"{'Threshold':<10} {'Event Rate':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Coverage':<10}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        marker = " 👈" if row['threshold'] == current_threshold else ""
        print(f"{row['threshold']:<10.1f} {row['event_detection_rate']:<12.2%} {row['precision']:<10.3f} "
              f"{row['recall']:<10.3f} {row['f1_score']:<10.3f} {row['avg_coverage']:<10.2%}{marker}")
    
    # Find and highlight best performing threshold by F1-score
    best_f1_idx = results_df['f1_score'].idxmax()
    best_threshold = results_df.loc[best_f1_idx, 'threshold']
    best_f1 = results_df.loc[best_f1_idx, 'f1_score']
    
    print(f"\n🏆 BEST THRESHOLD BY F1-SCORE: {best_threshold} (F1: {best_f1:.3f})")


def _print_error_type_breakdown(detection_results: Dict, current_threshold: float):
    """Print error type breakdown for current threshold."""
    current_threshold_metrics = detection_results[current_threshold]
    
    if current_threshold_metrics.error_type_metrics:
        print(f"\nERROR TYPE BREAKDOWN (Threshold: {current_threshold}):")
        print(f"{'Error Type':<15} {'Detected':<10} {'Total':<8} {'Rate':<10} {'Coverage':<10}")
        print("-" * 55)
        
        for error_type, type_metrics in current_threshold_metrics.error_type_metrics.items():
            print(f"{error_type:<15} {type_metrics['events_detected']:<10} {type_metrics['total_events']:<8} "
                  f"{type_metrics['event_detection_rate']:<10.2%} {type_metrics['avg_coverage']:<10.2%}")


def _create_comprehensive_visualization(
    results_df: pd.DataFrame, 
    station_id: str, 
    current_threshold: float, 
    error_multiplier: float,
    output_dir: Path
) -> Path:
    """Create comprehensive 4-panel visualization."""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Find best threshold for highlighting
    best_f1_idx = results_df['f1_score'].idxmax()
    best_threshold = results_df.loc[best_f1_idx, 'threshold']
    
    # Plot 1: Event Detection Rate vs Threshold
    ax1.plot(results_df['threshold'], results_df['event_detection_rate'], 'b-o', linewidth=3, markersize=8)
    ax1.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                label=f"Current ({current_threshold})")
    ax1.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.8, linewidth=2, 
                label=f"Best F1 ({best_threshold})")
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Event Detection Rate', fontsize=12)
    ax1.set_title('Event Detection Rate vs Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Precision vs Recall
    ax2.plot(results_df['recall'], results_df['precision'], 'g-o', linewidth=3, markersize=8)
    current_idx = results_df[results_df['threshold'] == current_threshold].index[0]
    best_idx = results_df[results_df['threshold'] == best_threshold].index[0]
    ax2.plot(results_df.loc[current_idx, 'recall'], results_df.loc[current_idx, 'precision'], 
             'ro', markersize=12, label=f"Current ({current_threshold})")
    ax2.plot(results_df.loc[best_idx, 'recall'], results_df.loc[best_idx, 'precision'], 
             'go', markersize=12, label=f"Best F1 ({best_threshold})")
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    
    # Plot 3: F1-Score vs Threshold
    ax3.plot(results_df['threshold'], results_df['f1_score'], 'purple', linewidth=3, marker='o', markersize=8)
    ax3.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                label=f"Current ({current_threshold})")
    ax3.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.8, linewidth=2, 
                label=f"Best F1 ({best_threshold})")
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 1.05)
    
    # Plot 4: Coverage vs Threshold
    ax4.plot(results_df['threshold'], results_df['avg_coverage'], 'orange', linewidth=3, marker='o', markersize=8)
    ax4.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                label=f"Current ({current_threshold})")
    ax4.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.8, linewidth=2, 
                label=f"Best F1 ({best_threshold})")
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Average Coverage', fontsize=12)
    ax4.set_title('Error Period Coverage vs Threshold', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 1.05)
    
    # Overall title
    title_text = f'Anomaly Detection Performance Analysis - Station {station_id}'
    if error_multiplier:
        title_text += f'\nError Multiplier: {error_multiplier}x'
    plt.suptitle(title_text, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"threshold_analysis_{station_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return plot_path


def _create_detailed_threshold_comparison(
    detection_results: Dict, 
    station_id: str, 
    error_multiplier: float,
    output_dir: Path
) -> Path:
    """Create detailed threshold comparison with confusion matrix heatmap."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Select key thresholds for detailed analysis
    key_thresholds = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    
    for i, threshold in enumerate(key_thresholds):
        if threshold in detection_results:
            metrics = detection_results[threshold]
            
            # Create confusion matrix
            cm = np.array([
                [metrics.true_positives, metrics.false_positives],
                [metrics.false_negatives, metrics.true_negatives]
            ])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                       yticklabels=['Actual Normal', 'Actual Anomaly'])
            
            axes[i].set_title(f'Threshold: {threshold}\nF1: {metrics.f1_score:.3f}, '
                             f'Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}')
    
    title_text = f'Detailed Threshold Comparison - Station {station_id}'
    if error_multiplier:
        title_text += f'\nError Multiplier: {error_multiplier}x'
    plt.suptitle(title_text, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"detailed_threshold_comparison_{station_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return plot_path


def _save_error_type_analysis(
    detection_results: Dict, 
    current_threshold: float, 
    station_id: str,
    output_dir: Path
) -> Path:
    """Save detailed error type analysis to CSV."""
    
    error_type_data = []
    
    for threshold, metrics in detection_results.items():
        if metrics.error_type_metrics:
            for error_type, type_metrics in metrics.error_type_metrics.items():
                error_type_data.append({
                    'threshold': threshold,
                    'error_type': error_type,
                    'events_detected': type_metrics['events_detected'],
                    'total_events': type_metrics['total_events'],
                    'event_detection_rate': type_metrics['event_detection_rate'],
                    'avg_coverage': type_metrics['avg_coverage']
                })
    
    error_type_df = pd.DataFrame(error_type_data)
    error_type_path = output_dir / f"error_type_analysis_{station_id}.csv"
    error_type_df.to_csv(error_type_path, index=False)
    
    return error_type_path


def _create_threshold_series_plots(
    val_data: pd.DataFrame,
    predictions: np.ndarray,
    station_id: str,
    config: Dict,
    output_dir: Path,
    original_data: pd.DataFrame = None,
    error_multiplier: float = None
) -> List[Path]:
    """
    Create water level anomaly plots for a series of different z-score thresholds.
    
    Args:
        val_data: Validation data with potential synthetic errors
        predictions: Model predictions
        station_id: Station ID for labeling
        config: Model configuration
        output_dir: Output directory
        original_data: Original validation data before error injection
        error_multiplier: Error multiplier used (for labeling)
        
    Returns:
        List of paths to generated plots
    """
    # Define thresholds to test
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 15.0]
    plot_paths = []
    
    # Create a subdirectory for threshold series plots
    threshold_dir = output_dir / "threshold_series"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating anomaly plots for different thresholds...")
    
    for threshold in thresholds:
        print(f"  Processing threshold: {threshold}")
        
        # Calculate z-scores and anomalies for this threshold
        z_scores, anomalies = calculate_z_scores_mad(
            val_data['vst_raw'].values,
            predictions,
            window_size=config['window_size'],
            threshold=threshold
        )
        
        # Calculate confidence levels
        from shared.anomaly_detection.anomaly_visualization import calculate_anomaly_confidence
        confidence = calculate_anomaly_confidence(z_scores, threshold)
        
        # Count anomalies by confidence
        high_conf = np.sum((anomalies) & (confidence == 'High'))
        med_conf = np.sum((anomalies) & (confidence == 'Medium'))
        low_conf = np.sum((anomalies) & (confidence == 'Low'))
        
        # Create title with threshold and counts
        title = f"Anomaly Detection - Station {station_id} (Threshold: {threshold})"
        if error_multiplier:
            title += f"\nError Multiplier: {error_multiplier}x"
        title += f"\nDetected: {high_conf} High, {med_conf} Medium, {low_conf} Low confidence"
        
        # Generate plot
        from shared.anomaly_detection.anomaly_visualization import plot_water_level_anomalies
        png_path, _ = plot_water_level_anomalies(
            test_data=val_data,
            predictions=predictions,
            z_scores=z_scores,
            anomalies=anomalies,
            threshold=threshold,
            title=title,
            output_dir=threshold_dir,
            save_png=True,
            save_html=False,
            show_plot=False,
            filename_prefix=f"threshold_{threshold:.1f}_",
            confidence=confidence,
            original_data=original_data
        )
        
        plot_paths.append(png_path)
        
    print(f"\nThreshold series plots saved to: {threshold_dir}")
    return plot_paths 