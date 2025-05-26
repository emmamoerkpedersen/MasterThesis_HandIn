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
        from _4_anomaly_detection.anomaly_visualization import calculate_anomaly_confidence
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
        from _4_anomaly_detection.anomaly_visualization import plot_water_level_anomalies
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