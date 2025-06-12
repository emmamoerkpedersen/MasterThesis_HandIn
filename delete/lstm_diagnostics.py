"""
Enhanced LSTM anomaly detection diagnostics module.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')

def plot_anomaly_detection_performance(
    station_key: str,
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    predictions: np.ndarray,
    prediction_errors: np.ndarray,
    anomaly_flags: np.ndarray,
    timestamps,
    threshold: float,
    ground_truth: Dict = None,
    output_dir: Path = None,
    z_scores: np.ndarray = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 300
):
    """
    Visualize anomaly detection performance using forecasting.
    
    Args:
        station_key: Station identifier
        original_data: Original data without synthetic errors
        modified_data: Data with synthetic errors
        predictions: Forecasted values from LSTM model
        prediction_errors: Array of prediction errors
        anomaly_flags: Binary array indicating anomalies
        timestamps: Timestamps for the predictions
        threshold: Threshold used for anomaly detection
        ground_truth: Dictionary with ground truth anomaly information (optional)
        output_dir: Directory to save visualizations
        z_scores: Z-scores of prediction errors (optional)
        figsize: Figure size
        dpi: DPI for static images
    """
    # Create output directory
    if output_dir:
        plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if original_data is None or modified_data is None:
        print(f"Error: Missing data for {station_key}")
        return
        
    # Check if we have predictions
    if predictions is None or len(predictions) == 0:
        print(f"Warning: No predictions available for {station_key}. Plotting data only.")
        # Create a simplified plot with just the data
        plt.figure(figsize=(12, 6))
        plt.plot(original_data.index, original_data['Value'], 
                color='blue', label='Original Data', linewidth=1.5)
        plt.plot(modified_data.index, modified_data['Value'], 
                color='red', label='Modified Data', linewidth=1.5)
        plt.title(f'Water Level Data - {station_key} (No Predictions)', fontsize=16)
        plt.ylabel('Water Level (mm)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(plot_dir / f"{station_key}_data_only.png", dpi=dpi)
        
        plt.show()
        return
    
    # Process 'Value' column naming
    for df in [original_data, modified_data]:
        if isinstance(df, pd.DataFrame) and 'Value' not in df.columns:
            if len(df.columns) > 0:
                df.rename(columns={df.columns[0]: 'Value'}, inplace=True)
    
    # Align timestamps
    if isinstance(timestamps, pd.DatetimeIndex):
        aligned_timestamps = timestamps
    else:
        aligned_timestamps = pd.DatetimeIndex(timestamps)
    
    # Get anomaly points
    anomaly_indices = np.where(anomaly_flags)[0]
    anomaly_timestamps = aligned_timestamps[anomaly_indices]
    
    # Find Y values for anomalies in modified data
    anomaly_values = []
    for ts in anomaly_timestamps:
        if ts in modified_data.index:
            anomaly_values.append(modified_data.loc[ts, 'Value'])
        else:
            # Find closest point
            closest_idx = modified_data.index.get_indexer([ts], method='nearest')[0]
            anomaly_values.append(modified_data['Value'].iloc[closest_idx])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
    # Data plot (top)
    ax1 = axes[0]
    
    # Plot original and modified data
    ax1.plot(original_data.index, original_data['Value'], 
             color='blue', label='Original Data', linewidth=1.5, alpha=0.7)
    
    ax1.plot(modified_data.index, modified_data['Value'], 
             color='red', label='Modified Data', linewidth=1.5, alpha=0.7)
    
    # Plot predicted values
    if predictions is not None:
        ax1.plot(aligned_timestamps, predictions, 
                color='green', label='Forecasted Values', linestyle='--', linewidth=1.5)
    
    # Mark anomalies
    if len(anomaly_indices) > 0:
        ax1.scatter(anomaly_timestamps, anomaly_values, 
                   color='yellow', marker='X', s=100, label='Detected Anomalies', 
                   edgecolors='black', zorder=5)
    
    # Add ground truth markers if available
    if ground_truth is not None and isinstance(ground_truth, dict) and 'periods' in ground_truth:
        # Mark ground truth periods with shaded regions
        for period in ground_truth['periods']:
            start = period['start']
            end = period['end']
            ax1.axvspan(start, end, color='red', alpha=0.1, label='_Ground Truth Period')
    
    # Format top plot
    ax1.set_title(f'Water Level Data - {station_key}', fontsize=16)
    ax1.set_ylabel('Water Level (mm)', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Only show one "Ground Truth Period" in legend
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax1.legend(unique_handles, unique_labels, fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis with date ticker
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Error visualization (bottom)
    ax2 = axes[1]
    
    if z_scores is not None:
        # Plot z-scores instead of raw errors
        ax2.plot(aligned_timestamps, z_scores, 
                color='purple', label='Z-Score', linewidth=1.5)
        
        # Add z-score threshold line
        z_threshold = z_scores[anomaly_flags == 1].min() if np.any(anomaly_flags) else 0
        ax2.axhline(y=z_threshold, color='red', linestyle='--', 
                   label=f'Z-Score Threshold ({z_threshold:.2f})')
        
        ax2.set_ylabel('Z-Score', fontsize=14)
        ax2.set_title('Z-Score Analysis (Standard Deviations from Mean)', fontsize=14)
    else:
        # Plot prediction errors
        ax2.plot(aligned_timestamps, prediction_errors, 
                color='purple', label='Forecast Error', linewidth=1.5)
        
        # Add threshold line
        ax2.axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Threshold ({threshold:.4f})')
        
        ax2.set_ylabel('Error', fontsize=14)
        ax2.set_title('Forecast Error & Threshold', fontsize=14)
    
    # Mark anomalies on error plot too
    if len(anomaly_indices) > 0:
        if z_scores is not None:
            anomaly_z_scores = z_scores[anomaly_indices]
            ax2.scatter(anomaly_timestamps, anomaly_z_scores, 
                       color='yellow', marker='X', s=80, 
                       edgecolors='black', zorder=5)
        else:
            anomaly_errors = prediction_errors[anomaly_indices]
            ax2.scatter(anomaly_timestamps, anomaly_errors, 
                       color='yellow', marker='X', s=80, 
                       edgecolors='black', zorder=5)
    
    # Format bottom plot
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis with date ticker
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.set_xlabel('Time', fontsize=14)
    
    # Add summary statistics text box
    anomaly_count = np.sum(anomaly_flags)
    total_points = len(anomaly_flags)
    anomaly_pct = (anomaly_count / total_points) * 100
    
    stats_text = (f"Total Points: {total_points}\n"
                  f"Detected Anomalies: {anomaly_count} ({anomaly_pct:.2f}%)")
    
    # Add text box in bottom right
    plt.figtext(0.92, 0.02, stats_text, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                fontsize=10, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save static plot if output directory provided
    if output_dir:
        static_path = plot_dir / f"{station_key}_anomaly_detection.png"
        fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {static_path}")
    
    plt.show()
    plt.close(fig)

def plot_training_history(
    history: Dict,
    model_name: str = None, 
    output_dir: Path = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
):
    """
    Plot LSTM training history.
    
    Args:
        history: Dictionary containing training history (loss, val_loss)
        model_name: Name of the model for labeling
        output_dir: Output directory for saving plots
        figsize: Figure size for matplotlib
        dpi: DPI for saving static images
    """
    # Create output directory if provided
    if output_dir:
        plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract history data
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    # Create epochs list
    epochs = list(range(1, len(train_loss) + 1))
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training and validation loss
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    
    if val_loss and len(val_loss) > 0:
        # Ensure val_loss has compatible dimension
        val_epochs = list(range(1, len(val_loss) + 1))
        ax.plot(val_epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    
    # Add best model marker if available
    if 'best_val_loss' in history:
        best_val_loss = history['best_val_loss']
        best_epoch = val_loss.index(best_val_loss) + 1 if best_val_loss in val_loss else None
        
        if best_epoch:
            ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
            ax.text(best_epoch + 0.1, min(train_loss), 'Best Model', 
                    rotation=90, verticalalignment='bottom', alpha=0.7)
    
    # Formatting
    title = f'Training History - {model_name}' if model_name else 'LSTM Training History'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save static plot if output directory provided
    if output_dir:
        model_id = model_name.replace(' ', '_').lower() if model_name else 'lstm'
        static_path = plot_dir / f"{model_id}_training_history.png"
        fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
        print(f"Training history plot saved to {static_path}")
    
    plt.show()
    plt.close(fig)

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Confusion Matrix',
    output_dir: Path = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300
):
    """
    Plot confusion matrix for anomaly detection results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Plot title
        output_dir: Output directory for saving plots
        figsize: Figure size for matplotlib
        dpi: DPI for saving static images
    """
    # Create output directory if provided
    if output_dir:
        plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title, fontsize=16)
    
    # Add class labels
    tick_marks = [0.5, 1.5]
    plt.xticks(tick_marks, ['Normal', 'Anomaly'])
    plt.yticks(tick_marks, ['Normal', 'Anomaly'])
    
    # Calculate and display metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics_text = (f"Precision: {precision:.3f}\n"
                   f"Recall: {recall:.3f}\n"
                   f"F1 Score: {f1:.3f}\n"
                   f"Accuracy: {accuracy:.3f}")
    
    plt.figtext(0.15, 0.01, metrics_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save static plot if output directory provided
    if output_dir:
        static_path = plot_dir / "confusion_matrix.png"
        plt.savefig(static_path, dpi=dpi, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {static_path}")
    
    plt.show()
    plt.close()
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def run_all_diagnostics(
    training_results: Dict,
    combined_train_data: Dict,
    training_anomalies: List,
    split_datasets: Dict,
    stations_results: Dict,
    output_dir: Path
):
    """
    Run all LSTM model diagnostics in one function.
    """
    print("\nGenerating LSTM diagnostics...")
    
    # Create output directory
    diagnostics_dir = output_dir / "diagnostics" / "lstm"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    diagnostic_results = {}
    
    # Plot training history for each model
    for key, result in training_results.items():
        if key.startswith('model_') and 'history' in result:
            try:
                station_id = key.replace('model_', '')
                print(f"Plotting training history for {station_id}...")
                plot_training_history(
                    history=result['history'],
                    model_name=f"Station {station_id}",
                    output_dir=output_dir,
                    figsize=(10, 6),
                    dpi=300
                )
            except Exception as e:
                print(f"Error plotting training history for {key}: {e}")
    
    # Add debug info to see what's in the results
    print("\nDiagnostics debug information:")
    for station_key, result in training_results.items():
        if not station_key.startswith('model_') and station_key != 'history':
            print(f"Station: {station_key}")
            print(f"  Keys available: {list(result.keys())}")
            
            # Check if predictions are present
            if 'predictions' in result:
                print(f"  Has predictions: {len(result['predictions'])} points")
            elif 'reconstructed_values' in result:
                print(f"  Has reconstructed_values: {len(result['reconstructed_values'])} points")
                
            # Check if data dimensions match
            if 'timestamps' in result and ('predictions' in result or 'reconstructed_values' in result):
                timestamps_len = len(result['timestamps'])
                pred_key = 'predictions' if 'predictions' in result else 'reconstructed_values'
                pred_len = len(result[pred_key])
                print(f"  Timestamps length: {timestamps_len}, Predictions length: {pred_len}")
    
    # Plot anomaly detection performance for each station-year
    for station_key, result in training_results.items():
        # Skip metadata entries and model entries
        if station_key.startswith('model_') or station_key == 'history':
            continue
            
        try:
            print(f"Plotting anomaly detection results for {station_key}...")
            
            # Get required data
            original_data = result.get('original_data')
            modified_data = result.get('modified_data')
            
            # FIX: Use 'reconstructed_values' when 'predictions' not found
            predictions = result.get('predictions')
            if predictions is None:
                predictions = result.get('reconstructed_values')  # Use this as fallback
                
            errors = result.get('prediction_errors')
            if errors is None:
                errors = result.get('reconstruction_errors')  # Use this as fallback
                
            anomaly_flags = result.get('anomaly_flags')
            timestamps = result.get('timestamps')
            threshold = result.get('threshold')
            z_scores = result.get('z_scores')
            
            ground_truth = stations_results[station_key]['ground_truth'] if station_key in stations_results else None
            
            # Generate visualization
            plot_anomaly_detection_performance(
                station_key=station_key,
                original_data=original_data,
                modified_data=modified_data,
                predictions=predictions,
                prediction_errors=errors,
                anomaly_flags=anomaly_flags,
                timestamps=timestamps,
                threshold=threshold,
                ground_truth=ground_truth,
                output_dir=output_dir,
                z_scores=z_scores,  # Pass z-scores to the plot function
                figsize=(16, 10),
                dpi=300
            )
            
        except Exception as e:
            print(f"Error plotting anomaly detection for {station_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze aggregate performance if ground truth available
    try:
        print("\nAnalyzing aggregate performance across all stations...")
        all_y_true = []
        all_y_pred = []
        
        for station_key, result in training_results.items():
            if station_key.startswith('model_') or station_key == 'history':
                continue
                
            # Skip if missing required data
            if ('anomaly_flags' not in result or 'timestamps' not in result or 
                station_key not in stations_results):
                continue
                
            # Get ground truth
            gt = stations_results[station_key]['ground_truth']
            
            # FIX: Check ground truth type properly
            has_periods = False
            if isinstance(gt, dict) and 'periods' in gt:
                has_periods = True
            elif isinstance(gt, pd.DataFrame) and 'error' in gt.columns:
                # Use error column to determine periods
                gt_periods = []
                error_mask = gt['error'] > 0
                # This is simplified - actual code would extract proper periods
                if error_mask.any():
                    has_periods = True
                    # For simplicity, just use the DataFrame's index directly
                    gt_array = error_mask.astype(int).values
                    anomaly_flags = result['anomaly_flags']
                    
                    # If lengths don't match, we need to align
                    if len(gt_array) != len(anomaly_flags):
                        # Skip this station if we can't align properly
                        print(f"  Skipping {station_key}: length mismatch between ground truth and predictions")
                        continue
                    
                    all_y_true.extend(gt_array)
                    all_y_pred.extend(anomaly_flags)
                    continue  # Skip the regular ground truth processing
            
            if not has_periods:
                print(f"  Skipping {station_key}: no valid ground truth periods")
                continue
                
            # Get predictions and timestamps
            anomaly_flags = result['anomaly_flags']
            timestamps = result['timestamps']
            
            # Create ground truth array
            gt_array = np.zeros(len(timestamps))
            for period in gt['periods']:
                # Mark points within anomaly periods as 1
                start, end = period['start'], period['end']
                mask = (pd.DatetimeIndex(timestamps) >= start) & (pd.DatetimeIndex(timestamps) <= end)
                gt_array[mask] = 1
            
            # Add to combined arrays
            all_y_true.extend(gt_array)
            all_y_pred.extend(anomaly_flags)
        
        # Plot combined confusion matrix
        if all_y_true and all_y_pred:
            metrics = plot_confusion_matrix(
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                title="Combined Anomaly Detection Performance",
                output_dir=output_dir,
                figsize=(8, 6),
                dpi=300
            )
            diagnostic_results['aggregate_metrics'] = metrics
            
    except Exception as e:
        print(f"Error analyzing aggregate performance: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDiagnostics generation complete.")
    return diagnostic_results

def analyze_threshold_sensitivity(
    station_key: str,
    prediction_errors: np.ndarray,
    ground_truth: np.ndarray,
    percentiles=range(80, 100, 2),
    output_dir: Path = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Analyze how different thresholds affect anomaly detection performance.
    
    Args:
        station_key: Station identifier
        prediction_errors: Array of prediction errors
        ground_truth: Binary ground truth array
        percentiles: Range of percentiles to test
        output_dir: Output directory for plots
        figsize: Figure size
    """
    metrics = []
    for percentile in percentiles:
        threshold = np.percentile(prediction_errors, percentile)
        predictions = (prediction_errors > threshold).astype(int)
        
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        
        metrics.append({
            'percentile': percentile,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    df = pd.DataFrame(metrics)
    
    # Plot metrics
    plt.figure(figsize=figsize)
    plt.plot(df['percentile'], df['precision'], 'b-', label='Precision')
    plt.plot(df['percentile'], df['recall'], 'r-', label='Recall')
    plt.plot(df['percentile'], df['f1'], 'g-', label='F1 Score')
    
    plt.axvline(x=95, color='gray', linestyle='--', label='Current (95%)')
    
    # Find optimal F1 threshold
    best_idx = df['f1'].idxmax()
    best_percentile = df.loc[best_idx, 'percentile']
    plt.axvline(x=best_percentile, color='green', linestyle='--', 
                label=f'Optimal F1 ({best_percentile}%)')
    
    plt.xlabel('Percentile Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Threshold Sensitivity Analysis - {station_key}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / f"{station_key}_threshold_analysis.png", dpi=300)
    
    plt.show()
    return df

def plot_contextual_errors(
    station_key: str,
    original_data: pd.DataFrame,
    predictions: np.ndarray,
    prediction_errors: np.ndarray,
    anomaly_flags: np.ndarray,
    timestamps,
    window_size: int = 24,  # 24 points = 6 hours at 15-min intervals
    output_dir: Path = None
):
    """
    Visualize prediction errors in context with local statistics.
    
    Args:
        station_key: Station identifier
        original_data: Original data
        predictions: Predicted values
        prediction_errors: Array of prediction errors
        anomaly_flags: Binary anomaly flags
        timestamps: Timestamps for predictions
        window_size: Size of rolling window for local statistics
        output_dir: Output directory for plots
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': np.nan,  # Will fill from original_data
        'predicted': predictions,
        'error': prediction_errors,
        'is_anomaly': anomaly_flags
    }).set_index('timestamp')
    
    # Fill actual values from original data
    for idx in df.index:
        if idx in original_data.index:
            df.loc[idx, 'actual'] = original_data.loc[idx, 'Value']
    
    # Calculate rolling statistics
    df['error_mean'] = df['error'].rolling(window=window_size, center=True).mean()
    df['error_std'] = df['error'].rolling(window=window_size, center=True).std()
    df['z_score'] = (df['error'] - df['error_mean']) / df['error_std']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                  gridspec_kw={'height_ratios': [2, 1]})
    
    # Top plot: Values and predictions
    ax1.plot(df.index, df['actual'], 'b-', label='Actual', alpha=0.7)
    ax1.plot(df.index, df['predicted'], 'g--', label='Predicted', alpha=0.7)
    
    # Mark anomalies
    anomaly_points = df[df['is_anomaly'] == 1]
    ax1.scatter(anomaly_points.index, anomaly_points['actual'], 
              color='red', marker='X', s=100, label='Anomalies')
    
    ax1.set_title(f'Actual vs Predicted Values - {station_key}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Contextual errors
    ax2.plot(df.index, df['error'], 'k-', label='Absolute Error')
    ax2.plot(df.index, df['error_mean'], 'b--', label='Rolling Mean Error')
    ax2.fill_between(df.index, 
                     df['error_mean'] - df['error_std'], 
                     df['error_mean'] + df['error_std'],
                     color='blue', alpha=0.2, label='±1 Std Dev')
    
    # Mark points with high z-scores
    high_z = df[df['z_score'] > 2]
    ax2.scatter(high_z.index, high_z['error'], 
               color='orange', marker='o', s=50, 
               label='High Z-Score (>2)')
    
    ax2.set_title('Contextual Error Analysis')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f"{station_key}_contextual_errors.png", dpi=300)
    
    plt.show()

def plot_station_comparison(
    training_results: Dict,
    stations_results: Dict,
    metric: str = 'f1_score',
    output_dir: Path = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Compare anomaly detection performance across stations.
    
    Args:
        training_results: Dictionary of training and evaluation results
        stations_results: Dictionary with synthetic error information
        metric: Metric to compare ('precision', 'recall', 'f1_score', 'accuracy')
        output_dir: Output directory for plots
        figsize: Figure size
    """
    station_metrics = []
    
    for station_key, result in training_results.items():
        if station_key.startswith('model_') or station_key == 'history':
            continue
            
        if 'metrics' in result and metric in result['metrics']:
            # Extract station ID from station_key (format: station_year)
            station_id = station_key.split('_')[0]
            year = station_key.split('_')[1]
            
            # Get metric value
            value = result['metrics'][metric]
            
            # Get error count
            error_count = 0
            if station_key in stations_results and 'ground_truth' in stations_results[station_key]:
                gt = stations_results[station_key]['ground_truth']
                if 'periods' in gt:
                    error_count = len(gt['periods'])
            
            station_metrics.append({
                'station_key': station_key,
                'station_id': station_id,
                'year': year,
                'metric_value': value,
                'error_count': error_count
            })
    
    if not station_metrics:
        print("No performance metrics available for comparison")
        return
        
    df = pd.DataFrame(station_metrics)
    
    # Group by station_id and calculate average metric
    station_avg = df.groupby('station_id')['metric_value'].mean().reset_index()
    station_avg = station_avg.sort_values('metric_value', ascending=False)
    
    # Plot
    plt.figure(figsize=figsize)
    bars = plt.bar(station_avg['station_id'], station_avg['metric_value'], 
                  color='skyblue', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.axhline(y=station_avg['metric_value'].mean(), color='red', linestyle='--',
               label=f'Average ({station_avg["metric_value"].mean():.3f})')
    
    plt.xlabel('Station ID')
    plt.ylabel(f'{metric.replace("_", " ").title()}')
    plt.title(f'Station Comparison by {metric.replace("_", " ").title()}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f"station_comparison_{metric}.png", dpi=300)
    
    plt.show()
    return df
