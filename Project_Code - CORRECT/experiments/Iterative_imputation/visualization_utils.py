"""
Visualization utilities for the iterative anomaly correction process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

def plot_iteration_convergence(original_data, error_data, correction_history, output_path):
    """Plot the convergence of corrections across iterations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot original data
    ax.plot(original_data.index, original_data['vst_raw'], 'k-', 
            label='Original Data', linewidth=2, alpha=0.7)
    if error_data is not None:
        ax.plot(error_data.index, error_data['vst_raw'], 'r--', 
                label='Data with Errors', linewidth=1.5, alpha=0.5)
    
    # Use colormap to show progression of iterations
    colors = cm.viridis(np.linspace(0, 1, len(correction_history)))
    
    # Skip the first entry which is the error-injected/original data
    for i, corrections in enumerate(correction_history[1:], 1):
        label = f'Iteration {i}' if i == 1 or i == len(correction_history)-1 else None
        ax.plot(original_data.index, corrections, '-', color=colors[i], 
               alpha=0.3 + 0.7*i/len(correction_history), 
               linewidth=1 + i/len(correction_history),
               label=label)
    
    ax.set_title('Convergence of Corrections Across Iterations')
    ax.set_ylabel('Water Level')
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "iteration_convergence.png")
    plt.close(fig)

def plot_correction_regions(original_data, error_data, correction_history, output_path):
    """Plot detailed views of significant correction regions."""
    if len(correction_history) <= 1:
        return
        
    final_corrections = correction_history[-1]
    initial_data = error_data['vst_raw'] if error_data is not None else original_data['vst_raw']
    
    # Calculate absolute differences between initial and final
    diff_series = abs(final_corrections - initial_data)
    
    # Find regions with large corrections (above 75th percentile of differences)
    threshold = np.nanpercentile(diff_series, 75)
    significant_regions = diff_series > threshold
    
    # Get contiguous segments with significant corrections
    from scipy import ndimage
    labeled_regions, num_regions = ndimage.label(significant_regions)
    
    # Show up to 3 interesting segments
    regions_to_show = min(3, num_regions)
    colors = cm.viridis(np.linspace(0, 1, len(correction_history)))
    
    if regions_to_show > 0:
        fig, axes = plt.subplots(regions_to_show, 1, figsize=(14, 5*regions_to_show), sharex=False)
        if regions_to_show == 1:
            axes = [axes]
        
        for i in range(1, regions_to_show+1):
            region_indices = np.where(labeled_regions == i)[0]
            if len(region_indices) < 5:  # Skip very small regions
                continue
            
            # Get data indices for the region with some context
            context_size = 10
            start_idx = max(0, region_indices[0] - context_size)
            end_idx = min(len(original_data)-1, region_indices[-1] + context_size)
            segment_indices = original_data.index[start_idx:end_idx+1]
            
            # Plot original and error data
            axes[i-1].plot(original_data.loc[segment_indices].index, 
                         original_data.loc[segment_indices, 'vst_raw'], 
                         'k-', label='Original', linewidth=2, alpha=0.7)
            
            if error_data is not None:
                axes[i-1].plot(error_data.loc[segment_indices].index, 
                             error_data.loc[segment_indices, 'vst_raw'], 
                             'r--', label='With Errors', linewidth=1.5, alpha=0.5)
            
            # Plot iterations
            for j, corrections in enumerate(correction_history[1:], 1):
                label = f'Iteration {j}' if j == 1 or j == len(correction_history)-1 else None
                axes[i-1].plot(segment_indices, corrections.loc[segment_indices], 
                             '-', color=colors[j], 
                             alpha=0.3 + 0.7*j/len(correction_history),
                             linewidth=1 + j/len(correction_history),
                             label=label)
            
            axes[i-1].set_title(f'Region {i}: Correction Progress Over Iterations')
            axes[i-1].set_ylabel('Water Level')
            axes[i-1].legend()
            axes[i-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "correction_detail_by_region.png")
        plt.close(fig)

def plot_anomaly_evolution(data, anomaly_masks, output_path):
    """Plot the evolution of anomaly detection across iterations."""
    if len(anomaly_masks) <= 1:
        return
        
    fig, axes = plt.subplots(len(anomaly_masks), 1, figsize=(14, 2*len(anomaly_masks)), sharex=True)
    
    if len(anomaly_masks) == 1:
        axes = [axes]
    
    for i, mask in enumerate(anomaly_masks):
        axes[i].fill_between(data.index, 0, mask, color='r', alpha=0.3)
        axes[i].set_title(f'Anomalies Detected - Iteration {i+1}')
        axes[i].set_ylabel('Anomaly')
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(output_path / "anomaly_detection_evolution.png")
    plt.close(fig)

def plot_rmse_convergence(correction_history, output_path):
    """Plot RMSE convergence between iterations."""
    if len(correction_history) <= 1:
        return
        
    rmse_values = []
    for i in range(1, len(correction_history)):
        prev = correction_history[i-1]
        curr = correction_history[i]
        
        valid_mask = ~np.isnan(prev) & ~np.isnan(curr)
        if valid_mask.sum() > 0:
            rmse = np.sqrt(((prev[valid_mask] - curr[valid_mask]) ** 2).mean())
            rmse_values.append(rmse)
        else:
            rmse_values.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(rmse_values)+1), rmse_values, 'bo-', linewidth=2)
    ax.set_title('Convergence of Corrections (RMSE between consecutive iterations)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "rmse_convergence.png")
    plt.close(fig)

def plot_final_comparison(original_data, error_data, results, output_path):
    """Plot final comparison of original, error-injected, and corrected data."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot all versions
    ax.plot(original_data.index, original_data['vst_raw'], 'b-', 
            label='Original Data', alpha=0.6)
    if error_data is not None:
        ax.plot(error_data.index, error_data['vst_raw'], 'r-', 
                label='Data with Errors', alpha=0.6)
    ax.plot(results.index, results['corrected'], 'g-', 
            label='Corrected Data', linewidth=2)
    
    # Add detected anomalies
    anomaly_points = results[results['anomaly'] == 1]
    ax.scatter(anomaly_points.index, anomaly_points['actual'], 
              color='purple', marker='o', s=30, label='Detected Anomalies')
    
    ax.legend()
    ax.set_title('Comparison: Original vs Error-Injected vs Corrected Data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Water Level')
    plt.tight_layout()
    
    plt.savefig(output_path / "final_comparison.png")
    plt.close(fig)

def visualize_all_results(original_data, error_data, correction_history, 
                         anomaly_masks, results, output_path):
    """Generate all visualizations for the correction process."""
    output_path.mkdir(parents=True, exist_ok=True)
    print("\nGenerating visualizations...")
    
    # Plot iteration convergence
    plot_iteration_convergence(original_data, error_data, correction_history, output_path)
    
    # Plot correction regions
    plot_correction_regions(original_data, error_data, correction_history, output_path)
    
    # Plot anomaly evolution
    plot_anomaly_evolution(original_data, anomaly_masks, output_path)
    
    # Plot RMSE convergence
    plot_rmse_convergence(correction_history, output_path)
    
    # Plot final comparison
    plot_final_comparison(original_data, error_data, results, output_path)
    
    print("All visualizations saved to:", output_path)

def visualize_anomalies(result_df, title='Anomaly Detection Results', figsize=(12, 10), output_path=None):
    """
    Visualize the anomaly detection results.
    
    Args:
        result_df: DataFrame with actual values, predictions, residuals, and anomaly indicators
        title: plot title (default 'Anomaly Detection Results')
        figsize: figure size as tuple (width, height) (default (12, 10))
        output_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Actual vs Predicted values
    axes[0].plot(result_df.index, result_df['actual'], 'b-', label='Actual')
    axes[0].plot(result_df.index, result_df['predicted'], 'r-', label='Predicted')
    
    # Highlight anomalies in the actual data
    anomaly_points = result_df[result_df['anomaly'] == 1]
    axes[0].scatter(anomaly_points.index, anomaly_points['actual'], 
                   color='orange', marker='o', s=50, label='Anomalies')
    
    axes[0].set_title(f'{title} - Actual vs Predicted')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Residuals with anomalies highlighted
    if 'residual' in result_df.columns:
        axes[1].plot(result_df.index, result_df['residual'], 'g-', label='Residual')
        if 'smoothed_residual' in result_df.columns:
            axes[1].plot(result_df.index, result_df['smoothed_residual'], 
                        'k--', alpha=0.5, label='Smoothed Residual')
        
        # Highlight anomalies in residuals
        axes[1].scatter(anomaly_points.index, anomaly_points['residual'], 
                       color='red', marker='x', s=50, label='Anomalies')
        
        axes[1].set_title('Residuals (Actual - Predicted)')
        axes[1].set_ylabel('Residual')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot 3: Anomaly indicator (binary)
    axes[2].fill_between(result_df.index, 0, result_df['anomaly'], color='red', alpha=0.3)
    axes[2].set_title('Anomaly Indicator')
    axes[2].set_ylabel('Anomaly (1=Yes)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
        return None
    return fig

def visualize_correction_results(result_df, output_path):
    """
    Generate visualizations of the anomaly correction results.
    
    Args:
        result_df: DataFrame with actual, predicted, anomaly, and corrected values
        output_path: Path to save visualizations
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Actual vs Predicted
    axes[0].plot(result_df.index, result_df['actual'], 'b-', label='Actual')
    axes[0].plot(result_df.index, result_df['predicted'], 'r-', label='Predicted')
    
    # Highlight anomalies
    anomaly_points = result_df[result_df['anomaly'] == 1]
    axes[0].scatter(anomaly_points.index, anomaly_points['actual'], 
                  color='orange', marker='o', s=30, label='Anomalies')
    
    axes[0].set_title('Water Level: Actual vs Predicted')
    axes[0].set_ylabel('Water Level')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Actual vs Corrected
    axes[1].plot(result_df.index, result_df['actual'], 'b-', label='Actual')
    axes[1].plot(result_df.index, result_df['corrected'], 'g-', label='Corrected')
    
    # Highlight differences
    correction_diff = (result_df['actual'] != result_df['corrected'])
    corrected_points = result_df[correction_diff]
    axes[1].scatter(corrected_points.index, corrected_points['corrected'], 
                  color='purple', marker='x', s=30, label='Corrections')
    
    axes[1].set_title('Water Level: Actual vs Corrected')
    axes[1].set_ylabel('Water Level')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Anomaly Indicator
    axes[2].fill_between(result_df.index, 0, result_df['anomaly'], color='red', alpha=0.3)
    axes[2].set_title('Anomaly Indicator')
    axes[2].set_ylabel('Anomaly (1=Yes)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True)
    axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(output_dir / "anomaly_correction_results.png")
    plt.close(fig)
    
    # Save results to CSV
    result_df.to_csv(output_dir / "anomaly_correction_results.csv")
    
    print(f"Visualizations saved to {output_dir}")

def visualize_correction_segments(result_df, segments, output_dir, max_segments=5):
    """
    Generate detailed visualizations for individual anomaly segments.
    
    Args:
        result_df: DataFrame with actual, predicted, anomaly, and corrected values
        segments: List of tuples (start_idx, end_idx) for each anomaly segment
        output_dir: Directory to save visualizations
        max_segments: Maximum number of segments to visualize
    """
    # Limit to max_segments
    if len(segments) > max_segments:
        # Select segments evenly distributed across the dataset
        indices = np.linspace(0, len(segments)-1, max_segments).astype(int)
        segments = [segments[i] for i in indices]
    
    for i, (start_idx, end_idx) in enumerate(segments):
        # Get segment data with some context (add 24 hours before and after)
        context_start = max(0, start_idx - 24)
        context_end = min(len(result_df), end_idx + 24)
        
        segment_indices = result_df.index[context_start:context_end]
        segment_data = result_df.loc[segment_indices]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot actual, predicted, and corrected
        axes[0].plot(segment_data.index, segment_data['actual'], 'b-', label='Actual')
        axes[0].plot(segment_data.index, segment_data['predicted'], 'r--', label='Predicted')
        axes[0].plot(segment_data.index, segment_data['corrected'], 'g-', label='Corrected')
        
        # Highlight anomaly region
        anomaly_start = result_df.index[start_idx]
        anomaly_end = result_df.index[end_idx]
        axes[0].axvspan(anomaly_start, anomaly_end, color='yellow', alpha=0.3, label='Anomaly Period')
        
        axes[0].set_title(f'Anomaly Segment {i+1}: {anomaly_start.strftime("%Y-%m-%d %H:%M")} to {anomaly_end.strftime("%Y-%m-%d %H:%M")}')
        axes[0].set_ylabel('Water Level')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot difference between actual and corrected
        axes[1].plot(segment_data.index, segment_data['actual'] - segment_data['corrected'], 'r-', label='Actual - Corrected')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].axvspan(anomaly_start, anomaly_end, color='yellow', alpha=0.3)
        
        axes[1].set_title('Difference between Actual and Corrected Values')
        axes[1].set_ylabel('Difference')
        axes[1].set_xlabel('Time')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"anomaly_segment_{i+1}.png")
        plt.close(fig) 