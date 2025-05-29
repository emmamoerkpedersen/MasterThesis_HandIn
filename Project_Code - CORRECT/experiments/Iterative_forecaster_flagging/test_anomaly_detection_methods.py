import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import local modules
from experiments.iterative_forecaster.alternating_config import ALTERNATING_CONFIG
from experiments.iterative_forecaster.simple_anomaly_detector import SimpleAnomalyDetector
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _4_anomaly_detection.z_score import calculate_z_scores_mad
from _4_anomaly_detection.anomaly_visualization import plot_water_level_anomalies
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from utils.error_utils import configure_error_params, inject_errors_into_dataset
from config import SYNTHETIC_ERROR_PARAMS

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test different anomaly detection methods')
    parser.add_argument('--station_id', type=str, default='21006846', help='Station ID to process')
    parser.add_argument('--error_multiplier', type=float, default=1.0, help='Error multiplier for synthetic errors')
    parser.add_argument('--output_dir', type=str, default='anomaly_detection_tests', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def analyze_baseline_data(data, output_dir):
    """
    Analyze baseline characteristics of normal water level data.
    
    Args:
        data: Clean water level data (pandas Series)
        output_dir: Directory to save analysis plots
    
    Returns:
        baseline_stats: Dictionary with baseline statistics
    """
    print("\n" + "="*50)
    print("📊 BASELINE DATA ANALYSIS")
    print("="*50)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove NaN values for analysis
    clean_data = data.dropna()
    
    # Basic statistics
    baseline_stats = {
        'mean': clean_data.mean(),
        'std': clean_data.std(),
        'min': clean_data.min(),
        'max': clean_data.max(),
        'percentiles': {
            'p5': clean_data.quantile(0.05),
            'p25': clean_data.quantile(0.25),
            'p50': clean_data.quantile(0.50),
            'p75': clean_data.quantile(0.75),
            'p95': clean_data.quantile(0.95),
        },
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data),
    }
    
    # Calculate change rates (15-minute differences)
    change_rates = clean_data.diff().dropna()
    baseline_stats['change_rates'] = {
        'mean': change_rates.mean(),
        'std': change_rates.std(),
        'p95': change_rates.quantile(0.95),
        'p99': change_rates.quantile(0.99),
    }
    
    print(f"Data points: {len(clean_data)}")
    print(f"Time range: {clean_data.index[0]} to {clean_data.index[-1]}")
    print(f"Mean: {baseline_stats['mean']:.2f} mm")
    print(f"Std Dev: {baseline_stats['std']:.2f} mm")
    print(f"Range: {baseline_stats['min']:.2f} to {baseline_stats['max']:.2f} mm")
    print(f"Change rate std: {baseline_stats['change_rates']['std']:.3f} mm/15min")
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution histogram
    axes[0, 0].hist(clean_data, bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title('Water Level Distribution')
    axes[0, 0].set_xlabel('Water Level (mm)')
    axes[0, 0].set_ylabel('Density')
    
    # Box plot
    axes[0, 1].boxplot(clean_data)
    axes[0, 1].set_title('Water Level Box Plot')
    axes[0, 1].set_ylabel('Water Level (mm)')
    
    # Change rates distribution
    axes[1, 0].hist(change_rates, bins=50, alpha=0.7)
    axes[1, 0].set_title('Change Rates Distribution')
    axes[1, 0].set_xlabel('Change Rate (mm/15min)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Time series sample (last 30 days)
    sample_data = clean_data[-30*24*4:]  # 30 days of 15-min data
    axes[1, 1].plot(sample_data.index, sample_data.values)
    axes[1, 1].set_title('Recent 30 Days Sample')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Water Level (mm)')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return baseline_stats

def analyze_synthetic_errors(error_generator, data_with_errors, original_data, output_dir):
    """
    Analyze characteristics of synthetic errors.
    
    Args:
        error_generator: SyntheticErrorGenerator with error_periods
        data_with_errors: Data with synthetic errors injected
        original_data: Original clean data
        output_dir: Directory to save analysis
    
    Returns:
        error_stats: Dictionary with error characteristics
    """
    print("\n" + "="*50)
    print("🚨 SYNTHETIC ERROR ANALYSIS")
    print("="*50)
    
    if not hasattr(error_generator, 'error_periods') or not error_generator.error_periods:
        print("No error periods found!")
        return {}
    
    output_dir = Path(output_dir)
    
    error_stats = {
        'total_errors': len(error_generator.error_periods),
        'error_types': {},
        'durations': [],
        'magnitudes': [],
        'affected_points': 0
    }
    
    # Analyze each error period
    for period in error_generator.error_periods:
        error_type = period.error_type
        
        if error_type not in error_stats['error_types']:
            error_stats['error_types'][error_type] = {
                'count': 0,
                'durations': [],
                'magnitudes': [],
                'examples': []
            }
        
        # Calculate duration in hours
        duration_hours = (period.end_time - period.start_time).total_seconds() / 3600
        error_stats['error_types'][error_type]['durations'].append(duration_hours)
        error_stats['durations'].append(duration_hours)
        
        # Calculate magnitude (difference from original)
        magnitude = np.abs(period.modified_values - period.original_values).max()
        error_stats['error_types'][error_type]['magnitudes'].append(magnitude)
        error_stats['magnitudes'].append(magnitude)
        
        error_stats['error_types'][error_type]['count'] += 1
        error_stats['affected_points'] += len(period.modified_values)
        
        # Store example for visualization
        if len(error_stats['error_types'][error_type]['examples']) < 3:
            error_stats['error_types'][error_type]['examples'].append(period)
    
    # Print summary
    print(f"Total synthetic errors: {error_stats['total_errors']}")
    print(f"Total affected points: {error_stats['affected_points']}")
    print(f"Percentage of data affected: {error_stats['affected_points']/len(data_with_errors)*100:.2f}%")
    
    for error_type, stats in error_stats['error_types'].items():
        print(f"\n{error_type}:")
        print(f"  Count: {stats['count']}")
        print(f"  Duration range: {min(stats['durations']):.1f} - {max(stats['durations']):.1f} hours")
        print(f"  Magnitude range: {min(stats['magnitudes']):.1f} - {max(stats['magnitudes']):.1f} mm")
    
    return error_stats

def test_detection_method(method_name, detector_func, data_with_errors, original_data, 
                         true_anomaly_flags, output_dir):
    """
    Test a specific anomaly detection method.
    
    Args:
        method_name: Name of the detection method
        detector_func: Function that returns (anomaly_flags, scores)
        data_with_errors: Data with synthetic errors
        original_data: Original clean data  
        true_anomaly_flags: Ground truth anomaly flags
        output_dir: Directory to save results
    
    Returns:
        results: Dictionary with performance metrics
    """
    print(f"\n🔍 Testing {method_name}...")
    
    try:
        # Run detection
        detected_flags, scores = detector_func(data_with_errors, original_data)
        
        # Calculate performance metrics
        true_positives = np.sum(detected_flags & true_anomaly_flags)
        false_positives = np.sum(detected_flags & ~true_anomaly_flags)
        false_negatives = np.sum(~detected_flags & true_anomaly_flags)
        true_negatives = np.sum(~detected_flags & ~true_anomaly_flags)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'method': method_name,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detected_count': np.sum(detected_flags),
            'true_count': np.sum(true_anomaly_flags)
        }
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
        print(f"  Detected: {np.sum(detected_flags)} / True: {np.sum(true_anomaly_flags)}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return {
            'method': method_name,
            'error': str(e),
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }

def create_detection_methods():
    """Create different anomaly detection methods to test."""
    
    methods = {}
    
    # Z-Score MAD method with smaller windows and lower thresholds
    # Test combinations of window sizes and thresholds
    window_sizes = [10, 25, 50, 100, 200, 500, 1500]  # From very small to current
    thresholds = [0.1,0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # From very sensitive to current
    
    # Test key combinations (focused on promising ranges + new lower thresholds)
    test_combinations = [
        # Very small windows with very low thresholds (super sensitive)
        (10, 0.1), (10, 0.25), (10, 0.5), (10, 0.75), (10, 1.0),
        (25, 0.1), (25, 0.25), (25, 0.5), (25, 0.75), (25, 1.0),
        (50, 0.1), (50, 0.25), (50, 0.5), (50, 0.75), (50, 1.0),
        
        # Medium windows with low-medium thresholds
        (100, 0.5), (100, 0.75), (100, 1.0), (100, 1.25),
        (200, 0.75), (200, 1.0), (200, 1.25),
        
        # Keep one larger window for comparison
        (500, 1.0), (1500, 2.0)
    ]
    
    for window_size, threshold in test_combinations:
        def zscore_detector(data_with_errors, original_data, win_size=window_size, thresh=threshold):
            z_scores, anomalies = calculate_z_scores_mad(
                original_data.values, 
                data_with_errors.values,
                window_size=win_size,
                threshold=thresh
            )
            return anomalies, z_scores
        
        methods[f'ZScore_W{window_size}_T{threshold}'] = zscore_detector
    
    # Keep the other methods for comparison
    # Isolation Forest
    def isolation_forest_detector(data_with_errors, original_data):
        # Prepare features (value + change rate)
        values = data_with_errors.values.reshape(-1, 1)
        changes = np.diff(data_with_errors, prepend=data_with_errors.iloc[0]).reshape(-1, 1)
        features = np.hstack([values, changes])
        
        # Remove NaN values
        valid_mask = ~np.isnan(features).any(axis=1)
        
        if np.sum(valid_mask) < 100:
            return np.zeros(len(data_with_errors), dtype=bool), np.zeros(len(data_with_errors))
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = np.ones(len(data_with_errors))
        predictions[valid_mask] = iso_forest.fit_predict(features[valid_mask])
        
        anomalies = predictions == -1
        scores = -predictions  # Convert to anomaly scores
        
        return anomalies, scores
    
    methods['Isolation_Forest'] = isolation_forest_detector
    
    # Simple threshold on change rates
    def change_rate_detector(data_with_errors, original_data):
        changes = np.abs(data_with_errors.diff())
        threshold = changes.quantile(0.99)  # 99th percentile
        anomalies = changes > threshold
        return anomalies.values, changes.values
    
    methods['Change_Rate_99p'] = change_rate_detector
    
    return methods

def run_anomaly_detection_tests(args):
    """Main function to run anomaly detection tests."""
    
    print("🧪 ANOMALY DETECTION TESTING FRAMEWORK")
    print("="*60)
    
    # Setup
    output_dir = Path("results") / "anomaly_detection_tests" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    config = ALTERNATING_CONFIG.copy()
    preprocessor = DataPreprocessor(config)
    
    # Load original data
    data_dir = project_dir / "data_utils" / "Sample data"
    data = pd.read_pickle(data_dir / "preprocessed_data.pkl")
    station_data = data.get(args.station_id)
    
    if not station_data:
        raise ValueError(f"Station ID {args.station_id} not found")
    
    # Get water level data (combine train + val for comprehensive testing)
    water_level_data = station_data['vst_raw']['vst_raw']
    
    # Use 2018-2023 for comprehensive testing
    test_data = water_level_data[(water_level_data.index.year >= 2018) & 
                                (water_level_data.index.year <= 2023)]
    
    print(f"Using data from {test_data.index[0]} to {test_data.index[-1]}")
    print(f"Total data points: {len(test_data)}")
    
    # Analyze baseline characteristics
    baseline_stats = analyze_baseline_data(test_data, output_dir)
    
    # Inject synthetic errors
    print(f"\n🚨 Injecting synthetic errors (multiplier: {args.error_multiplier})...")
    error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, args.error_multiplier)
    error_generator = SyntheticErrorGenerator(error_config)
    
    # Convert to DataFrame for error injection
    test_df = pd.DataFrame({'vst_raw': test_data})
    test_df_with_errors, error_report = inject_errors_into_dataset(
        test_df, error_generator, f"{args.station_id}_test", ['vst_raw']
    )
    
    # Analyze synthetic errors
    error_stats = analyze_synthetic_errors(
        error_generator, test_df_with_errors['vst_raw'], test_data, output_dir
    )
    
    # Create ground truth anomaly flags
    detector = SimpleAnomalyDetector()
    true_anomaly_flags = detector.create_perfect_flags(
        error_generator, len(test_df_with_errors), test_df_with_errors.index
    )
    
    print(f"\nGround truth: {np.sum(true_anomaly_flags)} anomalous points")
    
    # Test different detection methods
    methods = create_detection_methods()
    results = []
    
    print("\n" + "="*50)
    print("🔍 TESTING DETECTION METHODS")
    print("="*50)
    
    for method_name, detector_func in methods.items():
        result = test_detection_method(
            method_name, detector_func, 
            test_df_with_errors['vst_raw'], test_data,
            true_anomaly_flags, output_dir
        )
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'detection_results.csv', index=False)
    
    # Sort by F1-score for visualization
    results_sorted = results_df.sort_values('f1_score', ascending=False)
    
    # Visualize top 3 methods
    visualize_top_methods(
        results_sorted, test_df_with_errors, test_data, true_anomaly_flags,
        methods, output_dir, top_n=3
    )
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Filter out error results
    valid_results = results_sorted[~results_sorted['precision'].isna()]
    
    plt.subplot(2, 2, 1)
    plt.bar(range(len(valid_results)), valid_results['precision'])
    plt.title('Precision by Method')
    plt.xticks(range(len(valid_results)), valid_results['method'], rotation=45, ha='right')
    plt.ylabel('Precision')
    
    plt.subplot(2, 2, 2)
    plt.bar(range(len(valid_results)), valid_results['recall'])
    plt.title('Recall by Method')
    plt.xticks(range(len(valid_results)), valid_results['method'], rotation=45, ha='right')
    plt.ylabel('Recall')
    
    plt.subplot(2, 2, 3)
    plt.bar(range(len(valid_results)), valid_results['f1_score'])
    plt.title('F1-Score by Method')
    plt.xticks(range(len(valid_results)), valid_results['method'], rotation=45, ha='right')
    plt.ylabel('F1-Score')
    
    plt.subplot(2, 2, 4)
    plt.scatter(valid_results['precision'], valid_results['recall'], s=100)
    for i, method in enumerate(valid_results['method']):
        plt.annotate(method, (valid_results.iloc[i]['precision'], valid_results.iloc[i]['recall']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Plot')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)
    
    best_f1 = valid_results.iloc[0]  # First row has best F1-score
    print(f"Best F1-Score: {best_f1['method']} ({best_f1['f1_score']:.3f})")
    
    best_precision = valid_results.loc[valid_results['precision'].idxmax()]
    print(f"Best Precision: {best_precision['method']} ({best_precision['precision']:.3f})")
    
    best_recall = valid_results.loc[valid_results['recall'].idxmax()]
    print(f"Best Recall: {best_recall['method']} ({best_recall['recall']:.3f})")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results, baseline_stats, error_stats

def debug_detection_visualization(method_name, detected_flags, test_data, test_df_with_errors, 
                                  true_anomaly_flags, output_dir):
    """
    Create a detailed debug visualization to see what's really being detected.
    """
    print(f"\n🔍 Debug visualization for {method_name}...")
    
    # Print detailed statistics
    total_detected = np.sum(detected_flags)
    total_true = np.sum(true_anomaly_flags)
    overlap = np.sum(detected_flags & true_anomaly_flags)
    false_positives = np.sum(detected_flags & ~true_anomaly_flags)
    false_negatives = np.sum(~detected_flags & true_anomaly_flags)
    
    precision = overlap / total_detected if total_detected > 0 else 0
    recall = overlap / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Total data points: {len(detected_flags)}")
    print(f"  Detected anomalies: {total_detected}")
    print(f"  True anomalies: {total_true}")
    print(f"  Correct detections (overlap): {overlap}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")
    print(f"  Calculated Precision: {precision:.3f}")
    print(f"  Calculated Recall: {recall:.3f}")
    print(f"  Calculated F1: {f1:.3f}")
    
    # Find periods with most detections for detailed view
    detection_series = pd.Series(detected_flags, index=test_data.index)
    monthly_detections = detection_series.resample('M').sum()
    top_months = monthly_detections.nlargest(3)
    
    print(f"  Top 3 months with most detections:")
    for month, count in top_months.items():
        print(f"    {month.strftime('%Y-%m')}: {count} detections")
    
    # Create detailed view of the most active month
    if len(top_months) > 0:
        target_month = top_months.index[0]
        start_date = target_month - pd.DateOffset(days=15)
        end_date = target_month + pd.DateOffset(days=45)  # Show ~2 months around peak
        
        # Filter data for this period
        mask = (test_data.index >= start_date) & (test_data.index <= end_date)
        
        if mask.any():
            period_original = test_data[mask]
            period_corrupted = test_df_with_errors['vst_raw'][mask]
            period_detected = detected_flags[mask]
            period_true = true_anomaly_flags[mask]
            
            # Create detailed plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
            
            # Top plot: Time series with ALL detections visible
            ax1.plot(period_original.index, period_original.values, 'lightblue', 
                    label='Original Data', alpha=0.7, linewidth=1)
            ax1.plot(period_corrupted.index, period_corrupted.values, 'blue', 
                    label='Data with Errors', linewidth=1)
            
            # Mark detections (every single one)
            detected_indices = period_original.index[period_detected]
            if len(detected_indices) > 0:
                ax1.scatter(detected_indices, period_corrupted.loc[detected_indices], 
                           color='red', s=10, alpha=0.8, label=f'Detected ({len(detected_indices)})')
            
            # Mark true anomalies
            true_indices = period_original.index[period_true]
            if len(true_indices) > 0:
                ax1.scatter(true_indices, period_corrupted.loc[true_indices], 
                           color='orange', s=15, alpha=0.9, label=f'True Anomalies ({len(true_indices)})', 
                           marker='x')
            
            ax1.set_title(f'{method_name} - Detailed View: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
            ax1.set_ylabel('Water Level (mm)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Detection flags as binary signals
            ax2.fill_between(period_original.index, 0, period_detected.astype(int), 
                           alpha=0.7, color='red', label='Detected Anomalies')
            ax2.fill_between(period_original.index, 0, period_true.astype(int), 
                           alpha=0.5, color='orange', label='True Anomalies')
            
            ax2.set_title('Detection Flags (Binary)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Anomaly Flag')
            ax2.set_ylim(-0.1, 1.1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_dir = Path(output_dir)
            debug_dir = output_dir / "debug_visualizations"
            debug_dir.mkdir(exist_ok=True)
            
            filename = f'debug_{method_name.replace(".", "_")}_detailed.png'
            plt.savefig(debug_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Debug plot saved: {filename}")
            print(f"  Period shown: {len(detected_indices)} detections, {len(true_indices)} true anomalies")

def visualize_top_methods(top_results, test_df_with_errors, test_data, true_anomaly_flags, 
                         methods, output_dir, top_n=3):
    """
    Visualize the detection results of the top N performing methods across full data range.
    """
    print(f"\n📊 Visualizing top {top_n} detection methods across full data range...")
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "method_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Take top N methods by F1-score
    top_methods = top_results.head(top_n)
    
    # Create individual plots for each top method
    for idx, (_, method_row) in enumerate(top_methods.iterrows()):
        method_name = method_row['method']
        
        if method_name not in methods:
            print(f"  Skipping {method_name} - not found in methods")
            continue
            
        print(f"  Creating plot for {method_name}...")
        
        try:
            # Run detection for this method
            detected_flags, scores = methods[method_name](test_df_with_errors['vst_raw'], test_data)
            
            # ADD DEBUG VISUALIZATION
            debug_detection_visualization(method_name, detected_flags, test_data, 
                                       test_df_with_errors, true_anomaly_flags, output_dir)
            
            # Continue with original visualization...
            # Create figure for this method
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
            
            # Top plot: Full time series with detections
            # Sample every 10th point for performance (still shows full range)
            sample_step = 10
            sample_indices = slice(None, None, sample_step)
            
            sample_original = test_data.iloc[sample_indices]
            sample_corrupted = test_df_with_errors['vst_raw'].iloc[sample_indices]
            sample_detected = detected_flags[sample_indices]
            sample_true = true_anomaly_flags[sample_indices]
            
            # Plot data lines
            ax1.plot(sample_original.index, sample_original.values, 'lightblue', 
                    label='Original Data', alpha=0.7, linewidth=0.8)
            ax1.plot(sample_corrupted.index, sample_corrupted.values, 'blue', 
                    label='Data with Synthetic Errors', linewidth=1.0)
            
            # Mark detections as red dots
            detected_indices = sample_original.index[sample_detected]
            if len(detected_indices) > 0:
                ax1.scatter(detected_indices, sample_corrupted.loc[detected_indices], 
                           color='red', s=2, alpha=0.8, label='Detected Anomalies')
            
            # Mark true anomalies as orange x marks
            true_indices = sample_original.index[sample_true]
            if len(true_indices) > 0:
                ax1.scatter(true_indices, sample_corrupted.loc[true_indices], 
                           color='orange', s=8, alpha=0.9, label='True Anomalies', marker='x')
            
            # Format top plot
            ax1.set_title(f'{method_name}\nF1={method_row["f1_score"]:.3f}, Precision={method_row["precision"]:.3f}, Recall={method_row["recall"]:.3f}', 
                         fontsize=14)
            ax1.set_ylabel('Water Level (mm)', fontsize=12)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Detection density over time (monthly aggregation)
            detection_monthly = pd.Series(detected_flags, index=test_data.index).resample('M').sum()
            true_monthly = pd.Series(true_anomaly_flags, index=test_data.index).resample('M').sum()
            
            ax2.bar(detection_monthly.index, detection_monthly.values, alpha=0.7, 
                   color='red', label='Detected Anomalies per Month', width=20)
            ax2.bar(true_monthly.index, true_monthly.values, alpha=0.6, 
                   color='orange', label='True Anomalies per Month', width=20)
            
            ax2.set_title('Monthly Detection Counts', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Detections per Month', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format dates on both plots
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save individual plot
            plot_filename = f'method_{idx+1}_{method_name.replace(".", "_")}.png'
            plt.savefig(viz_dir / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print detection statistics
            total_detected = np.sum(detected_flags)
            total_true = np.sum(true_anomaly_flags)
            overlap = np.sum(detected_flags & true_anomaly_flags)
            
            print(f"    Total detections: {total_detected}")
            print(f"    True anomalies: {total_true}")
            print(f"    Overlap: {overlap}")
            print(f"    Plot saved: {plot_filename}")
            
        except Exception as e:
            print(f"    ❌ Error visualizing {method_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create summary comparison plot
    print("  Creating summary comparison...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Show performance metrics for top methods
        x_pos = np.arange(len(top_methods))
        width = 0.25
        
        ax.bar(x_pos - width, top_methods['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x_pos, top_methods['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x_pos + width, top_methods['f1_score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title(f'Top {top_n} Method Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_methods['method'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'top_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Error creating summary plot: {str(e)}")
    
    print(f"  ✅ Visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    args = parse_arguments()
    results, baseline_stats, error_stats = run_anomaly_detection_tests(args) 