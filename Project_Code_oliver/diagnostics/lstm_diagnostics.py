"""
Enhanced LSTM anomaly detection diagnostics module.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc, precision_score, recall_score, f1_score

# Set matplotlib style for professional-looking static plots
plt.style.use('seaborn-v0_8-whitegrid')


def plot_training_history(
    history: Dict,
    output_dir: Path,
    model_name: str = "lstm_autoencoder",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300
):
    """
    Plot and save LSTM training history.
    
    Args:
        history: Dictionary containing training history (loss, val_loss)
        output_dir: Output directory for saving plots
        model_name: Name of the model for file naming
        figsize: Figure size for matplotlib
        dpi: DPI for saving static images
    """
    # Create output directory
    plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract history data
    train_loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    
    # Convert range to list for Plotly compatibility
    epochs = list(range(1, len(train_loss) + 1))
    
    # Create static matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training and validation loss
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    if val_loss:
        ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    
    # Add early stopping marker if applicable
    if 'early_stopping_epoch' in history:
        stop_epoch = history['early_stopping_epoch']
        ax.axvline(x=stop_epoch, color='green', linestyle='--', alpha=0.7)
        ax.text(stop_epoch + 0.1, min(train_loss), 'Early Stopping', 
                rotation=90, verticalalignment='bottom', alpha=0.7)
    
    # Formatting
    ax.set_title('LSTM Autoencoder Training History', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save static plot
    static_path = plot_dir / f"{model_name}_training_history.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive Plotly figure
    fig_plotly = go.Figure()
    
    # Add training loss
    fig_plotly.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add validation loss if available
    if val_loss:
        fig_plotly.add_trace(go.Scatter(
            x=epochs,
            y=val_loss,
            mode='lines',
            name='Validation Loss',
            line=dict(color='firebrick', width=2)
        ))
    
    # Add early stopping marker if applicable
    if 'early_stopping_epoch' in history:
        stop_epoch = history['early_stopping_epoch']
        fig_plotly.add_vline(x=stop_epoch, line_dash="dash", line_color="green", opacity=0.7)
        fig_plotly.add_annotation(
            x=stop_epoch,
            y=min(train_loss),
            text="Early Stopping",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=40
        )
    
    # Update layout
    fig_plotly.update_layout(
        title='LSTM Autoencoder Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Save interactive plot
    interactive_path = plot_dir / f"{model_name}_training_history.html"
    fig_plotly.write_html(interactive_path)
    
    print(f"Training history plots saved to {plot_dir}")
    
    return {
        'static_path': static_path,
        'interactive_path': interactive_path
    }


def plot_reconstruction_results(
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    reconstruction_errors: np.ndarray,
    anomaly_flags: np.ndarray,
    timestamps,
    threshold: float,
    station_name: str,
    output_dir: Path,
    reconstructed_values: np.ndarray = None,
    prediction_intervals: Dict = None,
    z_scores: np.ndarray = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300,
    create_subdirs: bool = True  # Add this parameter
):
    """
    Create professional visualization of LSTM reconstruction results.
    """
    # Create output directory
    if create_subdirs:
        plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
    else:
        plot_dir = output_dir
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if original_data is None or modified_data is None:
        print(f"Error: Missing data for {station_name}")
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
    
    # Handle length mismatches
    min_len = min(len(aligned_timestamps), len(reconstruction_errors), len(anomaly_flags))
    if len(aligned_timestamps) != len(reconstruction_errors):
        print(f"Warning: Length mismatch - timestamps: {len(aligned_timestamps)}, errors: {len(reconstruction_errors)}")
        aligned_timestamps = aligned_timestamps[:min_len]
        reconstruction_errors = reconstruction_errors[:min_len]
        anomaly_flags = anomaly_flags[:min_len]
        if reconstructed_values is not None:
            reconstructed_values = reconstructed_values[:min_len]
    
    # Debug information
    print(f"\nPlotting diagnostics for {station_name}:")
    print(f"Reconstruction errors shape: {reconstruction_errors.shape}")
    print(f"Error range: [{np.min(reconstruction_errors):.6f}, {np.max(reconstruction_errors):.6f}]")
    print(f"Threshold value: {threshold:.6f}")
    
    # Normalize reconstruction errors if they're too large
    error_max = np.max(np.abs(reconstruction_errors))
    if error_max > 100:  # If errors are very large
        print("Normalizing large reconstruction errors...")
        reconstruction_errors = reconstruction_errors / error_max
        threshold = threshold / error_max
        print(f"Normalized error range: [{np.min(reconstruction_errors):.6f}, {np.max(reconstruction_errors):.6f}]")
        print(f"Normalized threshold: {threshold:.6f}")
    
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
    
    # Create static matplotlib plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Data plot (top)
    ax1 = axes[0]
    
    # Plot original and modified data
    ax1.plot(original_data.index, original_data['Value'], 
             color='blue', label='Original Data', linewidth=1.5, alpha=0.7)
    
    ax1.plot(modified_data.index, modified_data['Value'], 
             color='red', label='Modified Data', linewidth=1.5, alpha=0.7)
    
    # Plot reconstructed values if available
    if reconstructed_values is not None:
        ax1.plot(aligned_timestamps, reconstructed_values, 
                 color='green', label='Reconstructed', linestyle='--', linewidth=1.5)
    
    # Mark anomalies
    if len(anomaly_indices) > 0:
        ax1.scatter(anomaly_timestamps, anomaly_values, 
                   color='yellow', marker='X', s=100, label='Detected Anomalies', 
                   edgecolors='black', zorder=5)
    
    # Format top plot
    ax1.set_title(f'Water Level Data - {station_name}', fontsize=16)
    ax1.set_ylabel('Water Level (mm)', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis with date ticker
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Reconstruction error plot (bottom)
    ax2 = axes[1]
    
    # Plot reconstruction errors with proper scaling
    ax2.plot(aligned_timestamps, reconstruction_errors, 
             color='purple', label='Reconstruction Error', linewidth=1.5)
    
    # Add threshold line
    ax2.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    
    # Add Z-scores if available
    if z_scores is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(aligned_timestamps, z_scores, 
                     color='orange', label='Z-Score', linewidth=1.5, linestyle=':')
        ax2_twin.set_ylabel('Z-Score', fontsize=14, color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        ax2_twin.legend(loc='upper right')
    
    # Format bottom plot
    ax2.set_title('Reconstruction Error & Threshold', fontsize=14)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis with date ticker
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save static plot
    static_path = plot_dir / f"{station_name}_results.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive Plotly figure
    fig_plotly = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f'Water Level Data - {station_name}',
            'Reconstruction Error & Threshold'
        ],
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Add original data
    fig_plotly.add_trace(
        go.Scatter(
            x=original_data.index,
            y=original_data['Value'],
            name='Original Data',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add modified data
    fig_plotly.add_trace(
        go.Scatter(
            x=modified_data.index,
            y=modified_data['Value'],
            name='Modified Data',
            line=dict(color='red', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add reconstructed values if available
    if reconstructed_values is not None:
        fig_plotly.add_trace(
            go.Scatter(
                x=aligned_timestamps,
                y=reconstructed_values,
                name='Reconstructed',
                line=dict(color='green', width=1.5, dash='dash')
            ),
            row=1, col=1
        )
    
    # Add anomaly markers
    if len(anomaly_indices) > 0:
        fig_plotly.add_trace(
            go.Scatter(
                x=anomaly_timestamps,
                y=anomaly_values,
                mode='markers',
                name='Detected Anomalies',
                marker=dict(
                    color='yellow',
                    size=10,
                    symbol='x',
                    line=dict(color='black', width=1)
                )
            ),
            row=1, col=1
        )
    
    # Add reconstruction error
    fig_plotly.add_trace(
        go.Scatter(
            x=aligned_timestamps,
            y=reconstruction_errors,
            name='Reconstruction Error',
            line=dict(color='purple', width=1.5)
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig_plotly.add_trace(
        go.Scatter(
            x=[aligned_timestamps[0], aligned_timestamps[-1]],
            y=[threshold, threshold],
            name='Threshold',
            line=dict(color='red', width=1.5, dash='dash')
        ),
        row=2, col=1
    )
    
    # Add Z-scores if available
    if z_scores is not None:
        fig_plotly.add_trace(
            go.Scatter(
                x=aligned_timestamps,
                y=z_scores,
                name='Z-Score',
                line=dict(color='orange', width=1.5, dash='dot'),
                yaxis="y3"
            ),
            row=2, col=1
        )
        
        # Add secondary y-axis for Z-scores
        fig_plotly.update_layout(
            yaxis3=dict(
                title="Z-Score",
                titlefont=dict(color="orange"),
                tickfont=dict(color="orange"),
                anchor="x",
                overlaying="y2",
                side="right"
            )
        )
    
    # Update layout
    fig_plotly.update_layout(
        height=800,
        title_text=f"LSTM Anomaly Detection Results - {station_name}",
        showlegend=True,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig_plotly.update_yaxes(title_text="Water Level (mm)", row=1)
    fig_plotly.update_yaxes(title_text="Error", row=2)
    fig_plotly.update_xaxes(title_text="Time", row=2)
    
    # Save interactive plot
    interactive_path = plot_dir / f"{station_name}_results.html"
    fig_plotly.write_html(interactive_path)
    
    print(f"Reconstruction plots saved to {plot_dir}")
    
    return {
        'static_path': static_path,
        'interactive_path': interactive_path
    }


def plot_error_distribution(
    results: Dict,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
):
    """Plot distribution of reconstruction errors across all stations."""
    all_normal_errors = []
    all_anomaly_errors = []

    for station_key, result in results.items():
        if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
            continue
            
        # Skip if result is not a dictionary or missing required keys
        if not isinstance(result, dict):
            continue
            
        if 'reconstruction_errors' in result and 'threshold' in result:
            # Extract errors
            errors = result['reconstruction_errors']
            threshold = result['threshold']
            
            # Check for sufficient variance to avoid warnings
            if np.var(errors) < 1e-10:
                print(f"  Warning: Very low variance in errors for {station_key}, adding small noise")
                # Add tiny noise to ensure variance (won't affect visualization)
                errors = errors + np.random.normal(0, 1e-10, size=len(errors))
            
            # Split errors into normal and anomalous
            normal_mask = errors <= threshold
            anomaly_mask = ~normal_mask
            
            normal_errors = errors[normal_mask]
            anomaly_errors = errors[anomaly_mask]
            
            all_normal_errors.extend(normal_errors)
            all_anomaly_errors.extend(anomaly_errors)
    
    # Create static matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Error distribution (left)
    ax1 = axes[0]
    
    # Plot kernel density for normal errors
    sns.kdeplot(all_normal_errors, ax=ax1, label='Normal Errors', color='blue')
    
    # Plot kernel density for anomaly errors
    sns.kdeplot(all_anomaly_errors, ax=ax1, label='Anomaly Errors', color='red')
    
    # Format left plot
    ax1.set_title('Reconstruction Error Distribution', fontsize=16)
    ax1.set_xlabel('Error', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Log-scale distribution (right)
    ax2 = axes[1]
    
    # Plot kernel density with log x-scale
    sns.kdeplot(all_normal_errors, ax=ax2, label='Normal Errors', color='blue')
    sns.kdeplot(all_anomaly_errors, ax=ax2, label='Anomaly Errors', color='red')
    
    # Format right plot
    ax2.set_title('Reconstruction Error Distribution (Log Scale)', fontsize=16)
    ax2.set_xlabel('Error (Log Scale)', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_xscale('log')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save static plot
    static_path = output_dir / "diagnostics" / "lstm" / "plots" / "error_distribution.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive Plotly figure
    fig_plotly = go.Figure()
    
    # Add KDE for normal errors
    fig_plotly.add_trace(go.Scatter(
        x=all_normal_errors,
        y=np.zeros_like(all_normal_errors),
        mode='lines',
        name='Normal Errors',
        line=dict(color='blue', width=2)
    ))
    
    # Add KDE for anomaly errors
    fig_plotly.add_trace(go.Scatter(
        x=all_anomaly_errors,
        y=np.zeros_like(all_anomaly_errors),
        mode='lines',
        name='Anomaly Errors',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    fig_plotly.update_layout(
        title='Reconstruction Error Distribution',
        xaxis_title='Error',
        yaxis_title='Density',
        template='plotly_white',
        hovermode='closest'
    )
    
    # Add log-scale button
    fig_plotly.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                showactive=True,
                buttons=[
                    dict(
                        label="Linear Scale",
                        method="relayout",
                        args=[{"xaxis.type": "linear"}]
                    ),
                    dict(
                        label="Log Scale",
                        method="relayout",
                        args=[{"xaxis.type": "log"}]
                    )
                ]
            )
        ]
    )
    
    # Save interactive plot
    interactive_path = output_dir / "diagnostics" / "lstm" / "plots" / "error_distribution.html"
    fig_plotly.write_html(interactive_path)
    
    print(f"Error distribution plots saved to {output_dir / 'diagnostics' / 'lstm' / 'plots'}")
    
    return {
        'static_path': static_path,
        'interactive_path': interactive_path
    }


def plot_confusion_matrix(results: Dict, output_dir: Path, figsize=(10, 8), dpi=300):
    """Plot confusion matrix for anomaly detection results."""
    
    print("\nChecking ground truth data availability:")
    
    # Collect all predictions and ground truth
    y_true_all = []
    y_pred_all = []
    
    for station_key, result in results.items():
        if station_key in ['initial_training', 'fine_tuning', 'history', 'config', 'training_anomalies']:
            print(f"  {station_key}: Skipping metadata key")
            continue
            
        print(f"  {station_key}:", end=" ")
        
        if not isinstance(result, dict) or 'ground_truth' not in result:
            print("No ground truth available")
            continue
            
        ground_truth = result.get('ground_truth')
        print(f"Has ground_truth of type {type(ground_truth)}")
        
        if isinstance(ground_truth, pd.DataFrame):
            print(f"    DataFrame columns: {list(ground_truth.columns)}")
            print(f"    DataFrame shape: {ground_truth.shape}")
            
            # Get predictions
            predictions = result.get('anomaly_flags')
            if predictions is None:
                print("    No predictions available")
                continue
                
            # Ensure predictions and ground truth are aligned
            print(f"    Ground truth length: {len(ground_truth)}, Prediction length: {len(predictions)}")
            
            # Get timestamps for alignment
            timestamps = result.get('timestamps')
            if timestamps is None or len(timestamps) != len(predictions):
                print("    Invalid timestamps for alignment")
                continue
                
            # Convert timestamps to datetime if needed
            if isinstance(timestamps[0], str):
                timestamps = pd.to_datetime(timestamps)
            
            # Ensure arrays are 1-dimensional
            if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
                print(f"    Flattening {predictions.ndim}-dimensional predictions")
                predictions = predictions.flatten()
                
            if isinstance(timestamps, np.ndarray) and timestamps.ndim > 1:
                print(f"    Flattening {timestamps.ndim}-dimensional timestamps")
                timestamps = timestamps.flatten()
                
            # Create a DataFrame with predictions
            pred_df = pd.DataFrame({
                'timestamp': timestamps,
                'prediction': predictions
            }).set_index('timestamp')
            
            # Ensure ground truth has datetime index
            if not isinstance(ground_truth.index, pd.DatetimeIndex):
                ground_truth.index = pd.to_datetime(ground_truth.index)
            
            # Align ground truth with predictions
            aligned_data = ground_truth.join(pred_df, how='inner')
            
            if len(aligned_data) > 0:
                print(f"    Successfully aligned {len(aligned_data)} ground truth points with predictions")
                
                # Convert error column to binary (any error type = 1)
                y_true = (aligned_data['error'] > 0).astype(int).values
                y_pred = aligned_data['prediction'].values
                
                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)
            else:
                print("    No matching timestamps between ground truth and predictions")
    
    if not y_true_all or not y_pred_all:
        print("No valid ground truth and prediction pairs found")
        return
        
    # Convert to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    print(f"Combined data for confusion matrix: {len(y_true_all)} points")
    print(f"  Positive ground truth: {int(np.sum(y_true_all))} ({np.mean(y_true_all)*100:.2f}%)")
    print(f"  Positive predictions: {int(np.sum(y_pred_all))} ({np.mean(y_pred_all)*100:.2f}%)")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    # Create plot
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Anomaly Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    output_dir = Path(output_dir) / "diagnostics" / "lstm" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "confusion_matrix.png", bbox_inches='tight')
    plt.close()
    
    # Calculate and print metrics
    precision = precision_score(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all)
    
    print("\nPerformance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_roc_curve(
    results: Dict,
    output_dir: Path,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
):
    """Plot ROC curve for anomaly detection results."""
    all_true_values = []
    all_scores = []
    
    for station_key, result in results.items():
        if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
            continue
            
        if not isinstance(result, dict):
            continue
            
        if 'ground_truth' in result and 'reconstruction_errors' in result and 'timestamps' in result:
            gt = result['ground_truth']
            
            if isinstance(gt, pd.DataFrame) and 'error' in gt.columns:
                # Get prediction timestamps and scores (errors)
                pred_timestamps = result['timestamps']
                pred_scores = result['reconstruction_errors']
                
                # Create aligned ground truth
                aligned_gt = []
                for ts in pred_timestamps:
                    if ts in gt.index:
                        aligned_gt.append(gt.loc[ts, 'error'])
                    else:
                        nearest_idx = gt.index.get_indexer([ts], method='nearest')[0]
                        aligned_gt.append(gt.iloc[nearest_idx]['error'])
                
                # Convert to binary values
                aligned_gt = [1 if x else 0 for x in aligned_gt]
                
                # Store aligned values
                all_true_values.append(aligned_gt)
                all_scores.append(pred_scores)
    
    if not all_true_values:
        print("No ground truth data available for ROC curve")
        return
    
    # Combine all data
    y_true = np.concatenate(all_true_values)
    y_scores = np.concatenate(all_scores)
    
    # Check if we have both classes for ROC
    if len(np.unique(y_true)) < 2:
        print("Warning: ROC curve requires samples from both classes - dataset is too imbalanced")
        return
    
    # Create static matplotlib plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
    
    # All stations combined
    all_true = np.concatenate(all_true_values)
    all_scores = np.concatenate(all_scores)
    
    # Calculate ROC curve and AUC for combined data
    fpr, tpr, _ = roc_curve(all_true, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot combined ROC curve
    ax.plot(fpr, tpr, 'b-', label=f'All Stations (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Plot individual station ROC curves
    for true, scores in zip(all_true_values, all_scores):
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(true, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, alpha=0.6, label=f'Station (AUC = {roc_auc:.3f})')
    
    # Format plot
    ax.set_title('ROC Curve for Anomaly Detection', fontsize=16)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save static plot
    static_path = output_dir / "diagnostics" / "lstm" / "plots" / "roc_curve.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive Plotly figure
    fig_plotly = go.Figure()
    
    # Add diagonal reference line
    fig_plotly.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='black', dash='dash', width=1),
        name='Random',
        opacity=0.6
    ))
    
    # Add combined ROC curve
    fpr, tpr, _ = roc_curve(all_true, all_scores)
    roc_auc = auc(fpr, tpr)
    
    fig_plotly.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'All Stations (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add individual station ROC curves
    for true, scores in zip(all_true_values, all_scores):
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(true, scores)
        roc_auc = auc(fpr, tpr)
        
        fig_plotly.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'Station (AUC = {roc_auc:.3f})',
            opacity=0.6
        ))
    
    # Update layout
    fig_plotly.update_layout(
        title='ROC Curve for Anomaly Detection',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        template='plotly_white',
        legend=dict(x=0.7, y=0.1),
        hovermode='closest'
    )
    
    # Save interactive plot
    interactive_path = output_dir / "diagnostics" / "lstm" / "plots" / "roc_curve.html"
    fig_plotly.write_html(interactive_path)
    
    print(f"ROC curve plots saved to {output_dir / 'diagnostics' / 'lstm' / 'plots'}")
    
    return {
        'static_path': static_path,
        'interactive_path': interactive_path,
        'auc': roc_auc
    }


def plot_precision_recall_curve(
    results: Dict,
    output_dir: Path,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
):
    """
    Plot precision-recall curve for anomaly detection results.
    
    Args:
        results: Dictionary of anomaly detection results
        output_dir: Output directory for saving plots
        figsize: Figure size for matplotlib
        dpi: DPI for saving static images
    """
    # Create output directory
    plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect reconstruction errors and ground truth
    station_data = []
    
    for station_key, result in results.items():
        if station_key in ['initial_training', 'fine_tuning', 'history']:
            continue
            
        if ('reconstruction_errors' in result and 
            'ground_truth' in result and 
            result['ground_truth'] is not None):
            
            # Extract errors
            errors = result['reconstruction_errors']
            
            # Extract ground truth
            ground_truth = result['ground_truth']
            
            # Convert ground truth to binary format if needed
            if isinstance(ground_truth, pd.DataFrame) and 'is_anomaly' in ground_truth.columns:
                true = ground_truth['is_anomaly'].values
            elif isinstance(ground_truth, dict) and 'is_anomaly' in ground_truth:
                true = ground_truth['is_anomaly']
            else:
                continue
                
            # Ensure lengths match by trimming
            min_len = min(len(errors), len(true))
            errors = errors[:min_len]
            true = true[:min_len]
            
            # Store data
            station_data.append({
                'station': station_key,
                'errors': errors,
                'true': true
            })
    
    if not station_data:
        print("No ground truth data available for precision-recall curve")
        return
    
    # Create static matplotlib plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # All stations combined
    all_true = np.concatenate([d['true'] for d in station_data])
    all_errors = np.concatenate([d['errors'] for d in station_data])
    
    # Calculate precision-recall curve for combined data
    precision, recall, _ = precision_recall_curve(all_true, all_errors)
    avg_precision = average_precision_score(all_true, all_errors)
    
    # Plot combined PR curve
    ax.plot(recall, precision, 'b-', label=f'All Stations (AP = {avg_precision:.3f})', linewidth=2)
    
    # Plot individual station PR curves
    for data in station_data:
        station = data['station']
        true = data['true']
        errors = data['errors']
        
        # Calculate precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(true, errors)
        avg_precision = average_precision_score(true, errors)
        
        # Plot PR curve
        ax.plot(recall, precision, alpha=0.6, label=f'{station} (AP = {avg_precision:.3f})')
    
    # Format plot
    ax.set_title('Precision-Recall Curve for Anomaly Detection', fontsize=16)
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save static plot
    static_path = plot_dir / "precision_recall_curve.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive Plotly figure
    fig_plotly = go.Figure()
    
    # Add combined PR curve
    precision, recall, _ = precision_recall_curve(all_true, all_errors)
    avg_precision = average_precision_score(all_true, all_errors)
    
    fig_plotly.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'All Stations (AP = {avg_precision:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add individual station PR curves
    for data in station_data:
        station = data['station']
        true = data['true']
        errors = data['errors']
        
        # Calculate precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(true, errors)
        avg_precision = average_precision_score(true, errors)
        
        fig_plotly.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{station} (AP = {avg_precision:.3f})',
            opacity=0.6
        ))
    
    # Update layout
    fig_plotly.update_layout(
        title='Precision-Recall Curve for Anomaly Detection',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        template='plotly_white',
        legend=dict(x=0.7, y=0.1),
        hovermode='closest'
    )
    
    # Save interactive plot
    interactive_path = plot_dir / "precision_recall_curve.html"
    fig_plotly.write_html(interactive_path)
    
    print(f"Precision-recall curve plots saved to {plot_dir}")
    
    return {
        'static_path': static_path,
        'interactive_path': interactive_path,
        'avg_precision': avg_precision
    }


def plot_temporal_performance(
    results: Dict,
    output_dir: Path,
    window_size: int = 24*4,  # Default: 24 hours (at 15-min intervals)
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300
):
    """
    Plot temporal performance of the anomaly detection system.
    
    Args:
        results: Dictionary of anomaly detection results
        output_dir: Output directory for saving plots
        window_size: Size of the moving window for aggregating metrics
        figsize: Figure size for matplotlib
        dpi: DPI for saving static images
    """
    # Create output directory
    plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data by station
    temporal_metrics = []
    
    for station_key, result in results.items():
        if station_key in ['initial_training', 'fine_tuning', 'history']:
            continue
            
        if ('reconstruction_errors' in result and 
            'timestamps' in result and 
            'ground_truth' in result and 
            result['ground_truth'] is not None):
            
            # Extract data
            timestamps = result['timestamps']
            errors = result['reconstruction_errors']
            threshold = result['threshold']
            predictions = errors > threshold
            
            # Extract ground truth
            ground_truth = result['ground_truth']
            
            # Convert ground truth to binary format if needed
            if isinstance(ground_truth, pd.DataFrame) and 'is_anomaly' in ground_truth.columns:
                true = ground_truth['is_anomaly'].values
            elif isinstance(ground_truth, dict) and 'is_anomaly' in ground_truth:
                true = ground_truth['is_anomaly']
            else:
                continue
            
            # Ensure lengths match by trimming
            min_len = min(len(timestamps), len(predictions), len(true))
            timestamps = timestamps[:min_len]
            predictions = predictions[:min_len]
            true = true[:min_len]
            
            # Create DataFrame with metrics
            df = pd.DataFrame({
                'timestamp': timestamps,
                'prediction': predictions,
                'true': true,
                'error': errors[:min_len],
                'station': station_key
            })
            
            temporal_metrics.append(df)
    
    if not temporal_metrics:
        print("No temporal data available for performance analysis")
        return
    
    # Combine all station data
    combined_df = pd.concat(temporal_metrics)
    combined_df = combined_df.sort_values('timestamp')
    
    # Add hour, day of week, month features
    combined_df['hour'] = combined_df['timestamp'].dt.hour
    combined_df['day_of_week'] = combined_df['timestamp'].dt.dayofweek
    combined_df['month'] = combined_df['timestamp'].dt.month
    
    # Calculate metrics by time segments
    hourly_metrics = []
    daily_metrics = []
    monthly_metrics = []
    
    # Hour of day analysis
    for hour in range(24):
        hour_data = combined_df[combined_df['hour'] == hour]
        if len(hour_data) > 0:
            tp = sum((hour_data['prediction'] == 1) & (hour_data['true'] == 1))
            fp = sum((hour_data['prediction'] == 1) & (hour_data['true'] == 0))
            fn = sum((hour_data['prediction'] == 0) & (hour_data['true'] == 1))
            tn = sum((hour_data['prediction'] == 0) & (hour_data['true'] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            hourly_metrics.append({
                'hour': hour,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_error': hour_data['error'].mean(),
                'count': len(hour_data)
            })
    
    # Day of week analysis
    for day in range(7):
        day_data = combined_df[combined_df['day_of_week'] == day]
        if len(day_data) > 0:
            tp = sum((day_data['prediction'] == 1) & (day_data['true'] == 1))
            fp = sum((day_data['prediction'] == 1) & (day_data['true'] == 0))
            fn = sum((day_data['prediction'] == 0) & (day_data['true'] == 1))
            tn = sum((day_data['prediction'] == 0) & (day_data['true'] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            daily_metrics.append({
                'day': day,
                'day_name': day_names[day],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_error': day_data['error'].mean(),
                'count': len(day_data)
            })
    
    # Month analysis
    for month in range(1, 13):
        month_data = combined_df[combined_df['month'] == month]
        if len(month_data) > 0:
            tp = sum((month_data['prediction'] == 1) & (month_data['true'] == 1))
            fp = sum((month_data['prediction'] == 1) & (month_data['true'] == 0))
            fn = sum((month_data['prediction'] == 0) & (month_data['true'] == 1))
            tn = sum((month_data['prediction'] == 0) & (month_data['true'] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            monthly_metrics.append({
                'month': month,
                'month_name': month_names[month-1],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_error': month_data['error'].mean(),
                'count': len(month_data)
            })
    
    # Create static matplotlib plot
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Hourly metrics
    hourly_df = pd.DataFrame(hourly_metrics)
    
    if not hourly_df.empty:
        ax1 = axes[0]
        x = hourly_df['hour']
        
        # Plot metrics
        ax1.plot(x, hourly_df['precision'], 'b-', label='Precision', linewidth=1.5)
        ax1.plot(x, hourly_df['recall'], 'r-', label='Recall', linewidth=1.5)
        ax1.plot(x, hourly_df['f1'], 'g-', label='F1', linewidth=1.5)
        
        # Add error on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, hourly_df['avg_error'], 'k--', label='Avg Error', alpha=0.5)
        ax1_twin.set_ylabel('Average Error', fontsize=12, color='gray')
        
        # Format plot
        ax1.set_title('Performance by Hour of Day', fontsize=16)
        ax1.set_xlabel('Hour', fontsize=14)
        ax1.set_ylabel('Metric Value', fontsize=14)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xlim([-0.5, 23.5])
        ax1.set_ylim([0, 1])
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # Daily metrics
    daily_df = pd.DataFrame(daily_metrics)
    
    if not daily_df.empty:
        ax2 = axes[1]
        x = range(len(daily_df))
        
        # Plot metrics
        ax2.plot(x, daily_df['precision'], 'b-', label='Precision', linewidth=1.5)
        ax2.plot(x, daily_df['recall'], 'r-', label='Recall', linewidth=1.5)
        ax2.plot(x, daily_df['f1'], 'g-', label='F1', linewidth=1.5)
        
        # Add error on secondary axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, daily_df['avg_error'], 'k--', label='Avg Error', alpha=0.5)
        ax2_twin.set_ylabel('Average Error', fontsize=12, color='gray')
        
        # Format plot
        ax2.set_title('Performance by Day of Week', fontsize=16)
        ax2.set_xlabel('Day', fontsize=14)
        ax2.set_ylabel('Metric Value', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(daily_df['day_name'])
        ax2.set_ylim([0, 1])
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Monthly metrics
    monthly_df = pd.DataFrame(monthly_metrics)
    
    if not monthly_df.empty:
        ax3 = axes[2]
        x = range(len(monthly_df))
        
        # Plot metrics
        ax3.plot(x, monthly_df['precision'], 'b-', label='Precision', linewidth=1.5)
        ax3.plot(x, monthly_df['recall'], 'r-', label='Recall', linewidth=1.5)
        ax3.plot(x, monthly_df['f1'], 'g-', label='F1', linewidth=1.5)
        
        # Add error on secondary axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x, monthly_df['avg_error'], 'k--', label='Avg Error', alpha=0.5)
        ax3_twin.set_ylabel('Average Error', fontsize=12, color='gray')
        
        # Format plot
        ax3.set_title('Performance by Month', fontsize=16)
        ax3.set_xlabel('Month', fontsize=14)
        ax3.set_ylabel('Metric Value', fontsize=14)
        ax3.tick_params(axis='both', labelsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(monthly_df['month_name'])
        ax3.set_ylim([0, 1])
        ax3.legend(loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save static plot
    static_path = plot_dir / "temporal_performance.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive Plotly figure
    fig_plotly = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Performance by Hour of Day',
            'Performance by Day of Week',
            'Performance by Month'
        ],
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Add hourly metrics
    if not hourly_df.empty:
        x = hourly_df['hour']
        
        # Add primary metrics
        fig_plotly.add_trace(
            go.Scatter(x=x, y=hourly_df['precision'], mode='lines+markers', name='Precision', line=dict(color='blue')),
            row=1, col=1, secondary_y=False
        )
        fig_plotly.add_trace(
            go.Scatter(x=x, y=hourly_df['recall'], mode='lines+markers', name='Recall', line=dict(color='red')),
            row=1, col=1, secondary_y=False
        )
        fig_plotly.add_trace(
            go.Scatter(x=x, y=hourly_df['f1'], mode='lines+markers', name='F1', line=dict(color='green')),
            row=1, col=1, secondary_y=False
        )
        
        # Add error on secondary axis
        fig_plotly.add_trace(
            go.Scatter(x=x, y=hourly_df['avg_error'], mode='lines', name='Avg Error', 
                      line=dict(color='black', dash='dash'), opacity=0.5),
            row=1, col=1, secondary_y=True
        )
    
    # Add daily metrics
    if not daily_df.empty:
        x = daily_df['day_name']
        
        # Add primary metrics
        fig_plotly.add_trace(
            go.Scatter(x=x, y=daily_df['precision'], mode='lines+markers', name='Precision', 
                      line=dict(color='blue'), showlegend=False),
            row=2, col=1, secondary_y=False
        )
        fig_plotly.add_trace(
            go.Scatter(x=x, y=daily_df['recall'], mode='lines+markers', name='Recall', 
                      line=dict(color='red'), showlegend=False),
            row=2, col=1, secondary_y=False
        )
        fig_plotly.add_trace(
            go.Scatter(x=x, y=daily_df['f1'], mode='lines+markers', name='F1', 
                      line=dict(color='green'), showlegend=False),
            row=2, col=1, secondary_y=False
        )
        
        # Add error on secondary axis
        fig_plotly.add_trace(
            go.Scatter(x=x, y=daily_df['avg_error'], mode='lines', name='Avg Error', 
                      line=dict(color='black', dash='dash'), opacity=0.5, showlegend=False),
            row=2, col=1, secondary_y=True
        )
    
    # Add monthly metrics
    if not monthly_df.empty:
        x = monthly_df['month_name']
        
        # Add primary metrics
        fig_plotly.add_trace(
            go.Scatter(x=x, y=monthly_df['precision'], mode='lines+markers', name='Precision', 
                      line=dict(color='blue'), showlegend=False),
            row=3, col=1, secondary_y=False
        )
        fig_plotly.add_trace(
            go.Scatter(x=x, y=monthly_df['recall'], mode='lines+markers', name='Recall', 
                      line=dict(color='red'), showlegend=False),
            row=3, col=1, secondary_y=False
        )
        fig_plotly.add_trace(
            go.Scatter(x=x, y=monthly_df['f1'], mode='lines+markers', name='F1', 
                      line=dict(color='green'), showlegend=False),
            row=3, col=1, secondary_y=False
        )
        
        # Add error on secondary axis
        fig_plotly.add_trace(
            go.Scatter(x=x, y=monthly_df['avg_error'], mode='lines', name='Avg Error', 
                      line=dict(color='black', dash='dash'), opacity=0.5, showlegend=False),
            row=3, col=1, secondary_y=True
        )
    
    # Update layout
    fig_plotly.update_layout(
        height=1000,
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Update y-axis ranges
    fig_plotly.update_yaxes(range=[0, 1], secondary_y=False)
    
    # Update axis labels
    fig_plotly.update_xaxes(title_text="Hour", row=1, col=1)
    fig_plotly.update_xaxes(title_text="Day", row=2, col=1)
    fig_plotly.update_xaxes(title_text="Month", row=3, col=1)
    
    fig_plotly.update_yaxes(title_text="Metric Value", secondary_y=False, row=1, col=1)
    fig_plotly.update_yaxes(title_text="Average Error", secondary_y=True, row=1, col=1)
    
    fig_plotly.update_yaxes(title_text="Metric Value", secondary_y=False, row=2, col=1)
    fig_plotly.update_yaxes(title_text="Average Error", secondary_y=True, row=2, col=1)
    
    fig_plotly.update_yaxes(title_text="Metric Value", secondary_y=False, row=3, col=1)
    fig_plotly.update_yaxes(title_text="Average Error", secondary_y=True, row=3, col=1)
    
    # Save interactive plot
    interactive_path = plot_dir / "temporal_performance.html"
    fig_plotly.write_html(interactive_path)
    
    print(f"Temporal performance plots saved to {plot_dir}")
    
    return {
        'static_path': static_path,
        'interactive_path': interactive_path,
        'hourly_metrics': hourly_metrics,
        'daily_metrics': daily_metrics,
        'monthly_metrics': monthly_metrics
    }


def generate_lstm_report(
    detection_results: Dict,
    training_results: Dict,
    output_dir: Path
):
    """
    Generate a comprehensive HTML report of LSTM model performance.
    
    Args:
        detection_results: Dictionary of detection results by station
        training_results: Dictionary containing training history and metadata
        output_dir: Output directory for saving the report
    """
    report_dir = output_dir / "diagnostics" / "lstm"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Format timestamp for report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML report
    with open(report_dir / "lstm_report.html", "w") as f:
        # Write header and styling
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LSTM Anomaly Detection Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    color: #333;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 5px;
                }}
                h1, h2, h3 {{
                    margin-top: 30px;
                    color: #2c3e50;
                }}
                h1 {{
                    font-size: 26px;
                    margin-top: 0;
                    color: white;
                }}
                h2 {{
                    font-size: 22px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                h3 {{
                    font-size: 18px;
                    margin-top: 20px;
                }}
                .card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                    margin: 10px 0;
                }}
                .metric-name {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .plot-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .plot-img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .timestamp {{
                    font-size: 14px;
                    color: #95a5a6;
                    margin-top: 5px;
                }}
                footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding: 20px;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                .link-button {{
                    display: inline-block;
                    margin: 10px 0;
                    padding: 10px 15px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.3s;
                }}
                .link-button:hover {{
                    background-color: #2980b9;
                }}
            </style>
        </head>
        <body>
            <header>
                <div class="container">
                    <h1>LSTM Anomaly Detection Report</h1>
                    <div class="timestamp">Generated on {timestamp}</div>
                </div>
            </header>
            
            <div class="container">
        """)
        
        # 1. Model Summary
        f.write("""
                <section id="model-summary">
                    <h2>Model Summary</h2>
                    <div class="card">
        """)
        
        # Extract training configuration
        if 'config' in training_results:
            config = training_results['config']
            f.write("<h3>Model Configuration</h3>")
            f.write("<table>")
            f.write("<tr><th>Parameter</th><th>Value</th></tr>")
            
            for key, value in config.items():
                if isinstance(value, dict):
                    continue
                f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
            
            f.write("</table>")
        
        # Training results
        if 'history' in training_results:
            history = training_results['history']
            
            f.write("<h3>Training Results</h3>")
            
            # Training metrics
            f.write('<div class="metrics-grid">')
            
            if 'loss' in history:
                initial_loss = history['loss'][0] if len(history['loss']) > 0 else None
                final_loss = history['loss'][-1] if len(history['loss']) > 0 else None
                
                if initial_loss is not None:
                    f.write(f'''
                        <div class="metric-card">
                            <div class="metric-name">Initial Training Loss</div>
                            <div class="metric-value">{initial_loss:.6f}</div>
                        </div>
                    ''')
                
                if final_loss is not None:
                    f.write(f'''
                        <div class="metric-card">
                            <div class="metric-name">Final Training Loss</div>
                            <div class="metric-value">{final_loss:.6f}</div>
                        </div>
                    ''')
            
            if 'val_loss' in history:
                initial_val_loss = history['val_loss'][0] if len(history['val_loss']) > 0 else None
                final_val_loss = history['val_loss'][-1] if len(history['val_loss']) > 0 else None
                
                if initial_val_loss is not None:
                    f.write(f'''
                        <div class="metric-card">
                            <div class="metric-name">Initial Validation Loss</div>
                            <div class="metric-value">{initial_val_loss:.6f}</div>
                        </div>
                    ''')
                
                if final_val_loss is not None:
                    f.write(f'''
                        <div class="metric-card">
                            <div class="metric-name">Final Validation Loss</div>
                            <div class="metric-value">{final_val_loss:.6f}</div>
                        </div>
                    ''')
            
            # Training epochs
            epochs_completed = len(history.get('loss', []))
            early_stop_epoch = history.get('early_stopping_epoch', None)
            
            f.write(f'''
                <div class="metric-card">
                    <div class="metric-name">Training Epochs</div>
                    <div class="metric-value">{epochs_completed}</div>
                </div>
            ''')
            
            if early_stop_epoch is not None:
                f.write(f'''
                    <div class="metric-card">
                        <div class="metric-name">Early Stopping Epoch</div>
                        <div class="metric-value">{early_stop_epoch}</div>
                    </div>
                ''')
            
            # End metrics grid
            f.write('</div>')
            
            # Add training plot
            plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
            history_plot = plot_dir / "lstm_autoencoder_training_history.png"
            history_plot_relative = history_plot.relative_to(output_dir) if history_plot.exists() else None
            
            if history_plot_relative:
                f.write(f'''
                    <div class="plot-container">
                        <h3>Training History</h3>
                        <img src="../{history_plot_relative}" class="plot-img" alt="Training History">
                        <a href="../{str(history_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                    </div>
                ''')
        
        f.write("</div></section>")
        
        # 2. Detection Results Summary
        f.write("""
                <section id="detection-summary">
                    <h2>Detection Results Summary</h2>
                    <div class="card">
        """)
        
        # Count stations and anomalies
        station_count = 0
        total_anomalies = 0
        total_data_points = 0
        anomaly_stations = []
        
        for station_key, result in detection_results.items():
            if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
                continue
                
            station_count += 1
            
            if 'anomaly_flags' in result:
                station_anomalies = sum(result['anomaly_flags'])
                station_points = len(result['anomaly_flags'])
                
                total_anomalies += station_anomalies
                total_data_points += station_points
                
                # Store station info
                anomaly_stations.append({
                    'name': station_key,
                    'anomalies': station_anomalies,
                    'data_points': station_points,
                    'anomaly_pct': (station_anomalies / station_points * 100) if station_points > 0 else 0,
                    'threshold': result.get('threshold', 0)
                })
        
        # Detection summary metrics
        f.write('<div class="metrics-grid">')
        
        f.write(f'''
            <div class="metric-card">
                <div class="metric-name">Stations Analyzed</div>
                <div class="metric-value">{station_count}</div>
            </div>
        ''')
        
        f.write(f'''
            <div class="metric-card">
                <div class="metric-name">Total Data Points</div>
                <div class="metric-value">{int(total_data_points):,}</div>
            </div>
        ''')
        
        f.write(f'''
            <div class="metric-card">
                <div class="metric-name">Total Anomalies</div>
                <div class="metric-value">{int(total_anomalies):,}</div>
            </div>
        ''')
        
        anomaly_pct = (total_anomalies / total_data_points * 100) if total_data_points > 0 else 0
        f.write(f'''
            <div class="metric-card">
                <div class="metric-name">Overall Anomaly Rate</div>
                <div class="metric-value">{anomaly_pct:.2f}%</div>
            </div>
        ''')
        
        f.write('</div>')  # End metrics grid
        
        # Station Results Table
        if anomaly_stations:
            f.write('<h3>Results by Station</h3>')
            f.write('<table>')
            f.write('<tr><th>Station</th><th>Data Points</th><th>Anomalies</th><th>Anomaly %</th><th>Threshold</th></tr>')
            
            for station in sorted(anomaly_stations, key=lambda x: x['name']):
                f.write(f'''
                    <tr>
                        <td>{station['name']}</td>
                        <td>{int(station['data_points']):,}</td>
                        <td>{int(station['anomalies']):,}</td>
                        <td>{station['anomaly_pct']:.2f}%</td>
                        <td>{station['threshold']:.6f}</td>
                    </tr>
                ''')
            
            f.write('</table>')
        
        # Error distribution plot
        error_dist_plot = plot_dir / "error_distribution.png"
        error_dist_plot_relative = error_dist_plot.relative_to(output_dir) if error_dist_plot.exists() else None
        
        if error_dist_plot_relative:
            f.write(f'''
                <div class="plot-container">
                    <h3>Error Distribution</h3>
                    <img src="../{error_dist_plot_relative}" class="plot-img" alt="Error Distribution">
                    <a href="../{str(error_dist_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                </div>
            ''')
        
        f.write("</div></section>")
        
        # 3. Performance Evaluation
        f.write("""
                <section id="performance">
                    <h2>Performance Evaluation</h2>
                    <div class="card">
        """)
        
        # Check if we have ground truth for evaluation
        has_ground_truth = False
        for station_key, result in detection_results.items():
            if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
                continue
                
            if 'ground_truth' in result and result['ground_truth'] is not None:
                has_ground_truth = True
                break
        
        if has_ground_truth:
            # Summary of evaluation metrics
            f.write('<h3>Evaluation Metrics</h3>')
            
            # Add confusion matrix plot if available
            confusion_plot = plot_dir / "confusion_matrix.png"
            confusion_plot_relative = confusion_plot.relative_to(output_dir) if confusion_plot.exists() else None
            
            if confusion_plot_relative:
                f.write(f'''
                    <div class="plot-container">
                        <h3>Confusion Matrix</h3>
                        <img src="../{confusion_plot_relative}" class="plot-img" alt="Confusion Matrix">
                        <a href="../{str(confusion_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                    </div>
                ''')
            
            # Add ROC curve if available
            roc_plot = plot_dir / "roc_curve.png"
            roc_plot_relative = roc_plot.relative_to(output_dir) if roc_plot.exists() else None
            
            if roc_plot_relative:
                f.write(f'''
                    <div class="plot-container">
                        <h3>ROC Curve</h3>
                        <img src="../{roc_plot_relative}" class="plot-img" alt="ROC Curve">
                        <a href="../{str(roc_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                    </div>
                ''')
            
            # Add Precision-Recall curve if available
            pr_plot = plot_dir / "precision_recall_curve.png"
            pr_plot_relative = pr_plot.relative_to(output_dir) if pr_plot.exists() else None
            
            if pr_plot_relative:
                f.write(f'''
                    <div class="plot-container">
                        <h3>Precision-Recall Curve</h3>
                        <img src="../{pr_plot_relative}" class="plot-img" alt="Precision-Recall Curve">
                        <a href="../{str(pr_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                    </div>
                ''')
            
            # Add temporal performance if available
            temporal_plot = plot_dir / "temporal_performance.png"
            temporal_plot_relative = temporal_plot.relative_to(output_dir) if temporal_plot.exists() else None
            
            if temporal_plot_relative:
                f.write(f'''
                    <div class="plot-container">
                        <h3>Temporal Performance</h3>
                        <img src="../{temporal_plot_relative}" class="plot-img" alt="Temporal Performance">
                        <a href="../{str(temporal_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                    </div>
                ''')
        else:
            f.write('''
                <p><em>No ground truth data available for performance evaluation.</em></p>
            ''')
        
        f.write("</div></section>")
        
        # 4. Station Details
        f.write("""
                <section id="station-details">
                    <h2>Station Details</h2>
        """)
        
        for station_key, result in detection_results.items():
            if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
                continue
                
            f.write(f'''
                <div class="card">
                    <h3>{station_key}</h3>
            ''')
            
            # Station metrics
            if 'anomaly_flags' in result:
                station_anomalies = sum(result['anomaly_flags'])
                station_points = len(result['anomaly_flags'])
                anomaly_pct = (station_anomalies / station_points * 100) if station_points > 0 else 0
                
                f.write('<div class="metrics-grid">')
                
                f.write(f'''
                    <div class="metric-card">
                        <div class="metric-name">Data Points</div>
                        <div class="metric-value">{station_points:,}</div>
                    </div>
                ''')
                
                f.write(f'''
                    <div class="metric-card">
                        <div class="metric-name">Detected Anomalies</div>
                        <div class="metric-value">{station_anomalies:,}</div>
                    </div>
                ''')
                
                f.write(f'''
                    <div class="metric-card">
                        <div class="metric-name">Anomaly Rate</div>
                        <div class="metric-value">{anomaly_pct:.2f}%</div>
                    </div>
                ''')
                
                if 'threshold' in result:
                    f.write(f'''
                        <div class="metric-card">
                            <div class="metric-name">Detection Threshold</div>
                            <div class="metric-value">{result['threshold']:.6f}</div>
                        </div>
                    ''')
                
                # End metrics grid
                f.write('</div>')
            
            # Add station plot if available
            station_plot = plot_dir / f"{station_key}_results.png"
            station_plot_relative = station_plot.relative_to(output_dir) if station_plot.exists() else None
            
            if station_plot_relative:
                f.write(f'''
                    <div class="plot-container">
                        <img src="../{station_plot_relative}" class="plot-img" alt="{station_key} Results">
                        <a href="../{str(station_plot_relative).replace('.png', '.html')}" class="link-button">Interactive Plot</a>
                    </div>
                ''')
            
            f.write('</div>')  # End station card
        
        f.write("</section>")
        
        # 5. Conclusion
        f.write("""
                <section id="conclusion">
                    <h2>Conclusion</h2>
                    <div class="card">
                        <p>This report presents the results of anomaly detection using an LSTM autoencoder model on water level data.</p>
                        <p>The model was trained to reconstruct normal patterns in the data and identify anomalies based on reconstruction errors.</p>
                        <p>For more detailed analysis, please refer to the interactive plots and individual station results.</p>
                    </div>
                </section>
        """)
        
        # 6. Model Architecture
        f.write("""
        <section id="model-architecture">
            <h2>Model Architecture</h2>
            <div class="card">
                <h3>Model Registry Information</h3>
""")

        # Check if we have station-specific models
        has_station_models = False
        for result_key, result in detection_results.items():
            if isinstance(result, dict) and result.get('model_type') == 'station':
                has_station_models = True
                break

        if has_station_models:
            f.write('''
                <div class="info-panel">
                    <h4>Station-Specific Models Used</h4>
                    <p>The system used specialized models for individual stations where sufficient data was available.</p>
                    <table>
                        <tr><th>Station</th><th>Model Type</th><th>Performance</th></tr>
            ''')
            
            for result_key, result in detection_results.items():
                if isinstance(result, dict) and 'model_type' in result:
                    station_id = result_key.split('_')[0] if '_' in result_key else result_key
                    model_type = result.get('model_type', 'unknown')
                    f.write(f'''
                        <tr>
                            <td>{station_id}</td>
                            <td>{model_type.capitalize()} Model</td>
                            <td>{result.get('error_mean', 'N/A'):.6f}</td>
                        </tr>
                    ''')
            
            f.write('</table></div>')
        else:
            f.write('''
                <div class="info-panel">
                    <h4>Global Model Used</h4>
                    <p>A single global model was trained on all available station data.</p>
                </div>
            ''')

        f.write("</div></section>")
        
        # End container and add footer
        f.write("""
            </div>
            
            <footer>
                <div class="container">
                    <p>Generated by LSTM Anomaly Detection System</p>
                    <p>© 2023 All Rights Reserved</p>
                </div>
            </footer>
        </body>
        </html>
        """)
    
    print(f"LSTM report generated at: {report_dir / 'lstm_report.html'}")
    
    return {
        'report_path': report_dir / "lstm_report.html",
        'timestamp': timestamp
    }
    
    
def plot_synthetic_anomaly_detection(
    results: Dict,
    ground_truth: Dict,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
):
    """
    Plot detection performance on synthetic anomalies.
    
    Args:
        results: Dictionary of anomaly detection results
        ground_truth: Dictionary of ground truth anomaly periods
        output_dir: Output directory for saving plots
        figsize: Figure size for matplotlib
        dpi: DPI for saving static images
    """
    # Create output directory
    plot_dir = output_dir / "diagnostics" / "lstm" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect performance data
    all_stations = []
    detection_rates = []
    false_alarm_rates = []
    
    # Process each station
    for station_key, result in results.items():
        if station_key in ['initial_training', 'fine_tuning', 'history', 'config']:
            continue
            
        if not isinstance(result, dict) or 'timestamps' not in result:
            continue
            
        # Get ground truth for this station
        if station_key not in ground_truth:
            continue
            
        # Get anomaly flags and timestamps
        timestamps = result['timestamps']
        anomaly_flags = result['anomaly_flags']
        
        # Get ground truth periods
        gt = ground_truth[station_key]
        
        # Create ground truth mask
        gt_mask = np.zeros(len(timestamps), dtype=bool)
        
        if 'periods' in gt:
            for period in gt['periods']:
                period_start = period['start']
                period_end = period['end']
                
                # Mark points in this period as anomalous
                for i, ts in enumerate(timestamps):
                    if period_start <= ts <= period_end:
                        gt_mask[i] = True
        
        # Calculate true positives and false positives
        true_positives = np.sum(np.logical_and(anomaly_flags, gt_mask))
        false_positives = np.sum(np.logical_and(anomaly_flags, ~gt_mask))
        false_negatives = np.sum(np.logical_and(~np.array(anomaly_flags), gt_mask))
        true_negatives = np.sum(np.logical_and(~np.array(anomaly_flags), ~gt_mask))
        
        # Calculate detection rate and false alarm rate
        total_anomalies = np.sum(gt_mask)
        total_normals = len(gt_mask) - total_anomalies
        
        if total_anomalies > 0:
            detection_rate = true_positives / total_anomalies
        else:
            detection_rate = np.nan
            
        if total_normals > 0:
            false_alarm_rate = false_positives / total_normals
        else:
            false_alarm_rate = np.nan
            
        # Store results
        all_stations.append(station_key)
        detection_rates.append(detection_rate)
        false_alarm_rates.append(false_alarm_rate)
        
        # Print summary
        print(f"\nSynthetic anomaly detection for {station_key}:")
        print(f"  Total data points: {len(timestamps)}")
        print(f"  Synthetic anomalies: {total_anomalies} ({total_anomalies/len(timestamps)*100:.2f}%)")
        print(f"  Detection rate: {detection_rate*100:.2f}%")
        print(f"  False alarm rate: {false_alarm_rate*100:.2f}%")
    
    if not all_stations:
        print("No stations with both results and ground truth data")
        return
    
    # Create bar chart comparing detection across stations
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(len(all_stations))
    width = 0.35
    
    # Plot bars
    rects1 = ax.bar(x - width/2, detection_rates, width, label='Detection Rate')
    rects2 = ax.bar(x + width/2, false_alarm_rates, width, label='False Alarm Rate')
    
    # Add labels and formatting
    ax.set_xlabel('Station', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Synthetic Anomaly Detection Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_stations, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height*100:.1f}%',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    # Save static image
    static_path = plot_dir / "synthetic_anomaly_detection.png"
    fig.savefig(static_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Create interactive plotly version
    fig_plotly = go.Figure()
    
    # Add detection rate bars
    fig_plotly.add_trace(go.Bar(
        x=all_stations,
        y=detection_rates,
        name='Detection Rate',
        text=[f'{rate*100:.1f}%' if not np.isnan(rate) else 'N/A' for rate in detection_rates],
        marker_color='royalblue'
    ))
    
    # Add false alarm rate bars
    fig_plotly.add_trace(go.Bar(
        x=all_stations,
        y=false_alarm_rates,
        name='False Alarm Rate',
        text=[f'{rate*100:.1f}%' if not np.isnan(rate) else 'N/A' for rate in false_alarm_rates],
        marker_color='firebrick'
    ))
    
    # Update layout
    fig_plotly.update_layout(
        title='Synthetic Anomaly Detection Performance',
        xaxis_title='Station',
        yaxis_title='Rate',
        yaxis=dict(range=[0, 1.0]),
        barmode='group',
        template='plotly_white'
    )
    
    # Save interactive version
    interactive_path = plot_dir / "synthetic_anomaly_detection.html"
    fig_plotly.write_html(interactive_path)
    
    # Return paths to created visualizations
    return {
        'static_path': static_path,
        'interactive_path': interactive_path,
        'metrics': {
            'stations': all_stations,
            'detection_rates': detection_rates,
            'false_alarm_rates': false_alarm_rates
        }
    }
    
    
def plot_training_anomalies(
    anomalies_per_iteration: List[List[Dict]], 
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
):
    """
    Plot anomalies found during iterative training.
    
    Args:
        anomalies_per_iteration: List of anomalies found in each iteration
        output_dir: Directory to save plots
        figsize: Figure size
        dpi: DPI for saved plots
    """
    if not anomalies_per_iteration:
        print("No training anomalies to visualize")
        return
        
    # Create output directory
    plot_dir = Path(output_dir) / "diagnostics" / "lstm" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Group anomalies by station
    stations_anomalies = {}
    for iteration, anomalies in enumerate(anomalies_per_iteration):
        for anomaly in anomalies:
            station_key = anomaly['station_key']
            if station_key not in stations_anomalies:
                stations_anomalies[station_key] = []
            anomaly['iteration'] = iteration
            stations_anomalies[station_key].append(anomaly)
    
    # Plot for each station
    for station_key, anomalies in stations_anomalies.items():
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
            fig.suptitle(f'Training Anomalies - {station_key}', fontsize=16)
            
            # Sort anomalies by timestamp
            anomalies = sorted(anomalies, key=lambda x: x['timestamp'])
            
            # Extract data for plotting
            timestamps = [a['timestamp'] for a in anomalies]
            values = [a['original_value'] for a in anomalies]
            reconstructed = [a['reconstructed_value'] for a in anomalies]
            iterations = [a['iteration'] for a in anomalies]
            detection_methods = [a.get('detection_methods', []) for a in anomalies]
            
            # Plot original and reconstructed values
            ax1.plot(timestamps, values, 'b-', label='Original', alpha=0.6)
            ax1.plot(timestamps, reconstructed, 'g--', label='Reconstructed', alpha=0.6)
            
            # Color points by iteration
            for i in range(max(iterations) + 1):
                mask = [it == i for it in iterations]
                if any(mask):
                    iter_timestamps = [ts for ts, m in zip(timestamps, mask) if m]
                    iter_values = [v for v, m in zip(values, mask) if m]
                    ax1.scatter(iter_timestamps, iter_values, 
                              label=f'Iteration {i}', s=100, alpha=0.6)
            
            # Format first subplot
            ax1.set_ylabel('Value', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot detection methods
            unique_methods = sorted(set(method for methods in detection_methods for method in methods))
            method_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_methods)))
            
            for method, color in zip(unique_methods, method_colors):
                mask = [method in methods for methods in detection_methods]
                if any(mask):
                    method_timestamps = [ts for ts, m in zip(timestamps, mask) if m]
                    ax2.scatter(method_timestamps, [1]*len(method_timestamps), 
                              label=method, alpha=0.6, color=color)
            
            # Format second subplot
            ax2.set_ylabel('Detection\nMethods', fontsize=12)
            ax2.set_yticks([])
            ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = plot_dir / f"{station_key}_training_anomalies.png"
            plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting training anomalies for {station_key}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Training anomaly visualizations saved to", plot_dir)
    
def plot_confidence_detector_iterations(
    original_data: Dict, 
    anomalies_per_iteration: List[List[Dict]],
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300
):
    """
    Visualize the anomalies detected and removed during confidence detector iterations.
    
    Args:
        original_data: Dictionary of original data by station
        anomalies_per_iteration: List of anomalies detected in each iteration
        output_dir: Directory to save visualizations
        figsize: Figure size for static plots
        dpi: DPI for static images
    """
    # Create output directory
    plot_dir = output_dir / "diagnostics" / "lstm" / "plots" / "confidence_iterations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique stations
    all_stations = set()
    for iteration_anomalies in anomalies_per_iteration:
        for anomaly in iteration_anomalies:
            all_stations.add(anomaly['station_key'])
    
    # Process each station
    for station_key in all_stations:
        if station_key not in original_data:
            print(f"Warning: Original data for {station_key} not found")
            continue
            
        # Extract original station data
        station_data = original_data[station_key].get('vst_raw')
        if station_data is None or not isinstance(station_data, pd.DataFrame):
            print(f"Warning: Invalid data format for {station_key}")
            continue
            
        # Process each iteration for this station
        for i, iteration_anomalies in enumerate(anomalies_per_iteration):
            # Filter anomalies for this station
            station_anomalies = [a for a in iteration_anomalies if a['station_key'] == station_key]
            if not station_anomalies:
                continue
                
            # Create visualization
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot original data
            ax.plot(station_data.index, station_data['Value'], 
                    color='blue', label='Original Data', linewidth=1.5)
            
            # Extract anomaly timestamps and values
            anomaly_timestamps = [a['timestamp'] for a in station_anomalies]
            anomaly_values = []
            
            for ts in anomaly_timestamps:
                if ts in station_data.index:
                    anomaly_values.append(station_data.loc[ts, 'Value'])
                else:
                    # Find closest point
                    closest_idx = station_data.index.get_indexer([ts], method='nearest')[0]
                    anomaly_values.append(station_data['Value'].iloc[closest_idx])
            
            # Mark anomalies
            ax.scatter(anomaly_timestamps, anomaly_values, 
                       color='red', marker='X', s=100, 
                       label=f'Iteration {i+1} Anomalies ({len(station_anomalies)})', 
                       edgecolors='black', zorder=5)
            
            # Add z-score thresholds if available
            if 'threshold' in station_anomalies[0]:
                threshold = station_anomalies[0]['threshold']
                z_score = station_anomalies[0]['z_score'] / (station_anomalies[0]['error'] - threshold)
                
                # Add annotation about confidence interval
                ax.annotate(
                    f"Confidence interval: z={z_score:.2f}",
                    xy=(0.02, 0.95),
                    xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Format plot
            ax.set_title(f'{station_key} - Iteration {i+1} Anomaly Detection', fontsize=16)
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Water Level (mm)', fontsize=14)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis with date ticker
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            # Save plot
            fig.tight_layout()
            plot_path = plot_dir / f"{station_key}_iteration_{i+1}.png"
            fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Confidence detector iteration visualizations saved to {plot_dir}")
    
    return {"plot_dir": plot_dir}