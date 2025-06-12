import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_anomaly_confidence(z_scores, threshold):
    """
    Calculate confidence levels for detected anomalies based on z-score distance.
    
    Args:
        z_scores: Array of z-scores
        threshold: Base threshold for anomaly detection
        
    Returns:
        Array of confidence levels ('High', 'Medium', 'Low', 'Normal')
    """
    abs_z_scores = np.abs(z_scores)
    confidence = np.full_like(z_scores, 'Normal', dtype=object)
    
    # Define confidence thresholds
    high_threshold = 2.0 * threshold      # High confidence: > 2x threshold
    medium_threshold = 1.5 * threshold    # Medium confidence: 1.5x to 2x threshold  
    low_threshold = threshold             # Low confidence: threshold to 1.5x threshold
    
    # Assign confidence levels
    confidence[abs_z_scores >= high_threshold] = 'High'
    confidence[(abs_z_scores >= medium_threshold) & (abs_z_scores < high_threshold)] = 'Medium'
    confidence[(abs_z_scores >= low_threshold) & (abs_z_scores < medium_threshold)] = 'Low'
    
    return confidence

def create_anomaly_zoom_plots(val_data, predictions, z_scores, anomalies, confidence, 
                            error_generator, station_id, config, output_dir, original_val_data=None):
    """
    Create zoomed plots for each type of injected error showing model behavior during anomalous events.
    Shows original data, modified data (with injected errors), and model predictions.
    
    Args:
        val_data: Validation data with synthetic errors
        predictions: Model predictions
        z_scores: Calculated z-scores
        anomalies: Boolean array of detected anomalies
        confidence: Confidence levels for each point
        error_generator: SyntheticErrorGenerator instance with error_periods
        station_id: Station identifier
        config: Configuration dictionary
        output_dir: Output directory for plots
        original_val_data: Original validation data before error injection
    """
    if not hasattr(error_generator, 'error_periods') or not error_generator.error_periods:
        print("No error periods found for zoom plots")
        return
    
    print(f"\nCreating zoom plots for {len(error_generator.error_periods)} injected errors...")
    print(f"DEBUG: val_data index range: {val_data.index[0]} to {val_data.index[-1]}")
    
    # Group error periods by type
    error_types = {}
    for period in error_generator.error_periods:
        error_type = period.error_type
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(period)
        print(f"DEBUG: Found {error_type} error from {period.start_time} to {period.end_time}")
    
    # Create zoom plots for each error type (take first occurrence of each type)
    for error_type, periods in error_types.items():
        try:
            # Take the first period of this error type
            period = periods[0]
            
            print(f"Creating zoom plot for {error_type} error...")
            print(f"DEBUG: Error period: {period.start_time} to {period.end_time}")
            
            # Calculate buffer (2 hours before and after)
            buffer_hours = 22
            buffer_steps = buffer_hours * 4  # 15-min intervals
            
            # Find the period in the validation data
            start_time = period.start_time
            end_time = period.end_time
            
            # Get indices for the error period
            period_mask = (val_data.index >= start_time) & (val_data.index <= end_time)
            if not period_mask.any():
                print(f"ERROR: Error period not found in validation data for {error_type}")
                print(f"  Looking for: {start_time} to {end_time}")
                print(f"  Data range: {val_data.index[0]} to {val_data.index[-1]}")
                continue
                
            # Find start and end indices
            period_indices = np.where(period_mask)[0]
            start_idx = max(0, period_indices[0] - buffer_steps)
            end_idx = min(len(val_data), period_indices[-1] + buffer_steps)
            
            print(f"DEBUG: Period indices: {period_indices[0]} to {period_indices[-1]}")
            print(f"DEBUG: Zoom indices (with buffer): {start_idx} to {end_idx}")
            
            # Create zoom data with buffer
            zoom_data = val_data.iloc[start_idx:end_idx].copy()
            
            # Replace the error period with the actual modified values
            error_period_start_idx = period_indices[0] - start_idx
            error_period_end_idx = period_indices[-1] - start_idx + 1
            zoom_data.iloc[error_period_start_idx:error_period_end_idx, zoom_data.columns.get_loc('vst_raw')] = period.modified_values
            
            # Extract other data for plotting
            zoom_predictions = predictions[start_idx:end_idx] if len(predictions) > end_idx else predictions[start_idx:]
            zoom_z_scores = z_scores[start_idx:end_idx] if len(z_scores) > end_idx else z_scores[start_idx:]
            zoom_anomalies = anomalies[start_idx:end_idx] if len(anomalies) > end_idx else anomalies[start_idx:]
            zoom_confidence = confidence[start_idx:end_idx] if len(confidence) > end_idx else confidence[start_idx:]
            
            # Extract original data if available
            zoom_original_data = None
            if original_val_data is not None:
                zoom_original_data = original_val_data.iloc[start_idx:end_idx].copy()
            
            # DEBUG: Check if the data shows the synthetic error
            print(f"DEBUG: Zoom data range: {zoom_data['vst_raw'].min():.1f} to {zoom_data['vst_raw'].max():.1f} mm")
            print(f"DEBUG: Original values in error period: {period.original_values.min():.1f} to {period.original_values.max():.1f} mm")
            print(f"DEBUG: Modified values in error period: {period.modified_values.min():.1f} to {period.modified_values.max():.1f} mm")
            
            # ADDITIONAL DEBUG: Check specific values at error period indices
            error_period_data = zoom_data.iloc[error_period_start_idx:error_period_end_idx]
            print(f"DEBUG: Actual zoom data during error period: {error_period_data['vst_raw'].min():.1f} to {error_period_data['vst_raw'].max():.1f} mm")
            print(f"DEBUG: Expected modified range should be: {period.modified_values.min():.1f} to {period.modified_values.max():.1f} mm")
            
            # Check if the data matches original or modified values
            if abs(error_period_data['vst_raw'].mean() - np.mean(period.original_values)) < abs(error_period_data['vst_raw'].mean() - np.mean(period.modified_values)):
                print(f"🚨 BUG DETECTED: Zoom data matches ORIGINAL values, not MODIFIED values!")
            else:
                print(f"✅ Zoom data correctly shows MODIFIED values")
            
            # Create zoom plot
            zoom_title = f"Zoom: {error_type.title()} Error - Station {station_id}"
            zoom_title += f"\nPeriod: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
            
            # Count anomalies by confidence in this zoom
            high_conf = np.sum((zoom_anomalies) & (zoom_confidence == 'High'))
            med_conf = np.sum((zoom_anomalies) & (zoom_confidence == 'Medium'))
            low_conf = np.sum((zoom_anomalies) & (zoom_confidence == 'Low'))
            
            zoom_title += f"\nDetected: {high_conf} High, {med_conf} Medium, {low_conf} Low confidence anomalies"
            
            zoom_png, zoom_html = plot_water_level_anomalies(
                test_data=zoom_data,
                predictions=zoom_predictions,
                z_scores=zoom_z_scores,
                anomalies=zoom_anomalies,
                threshold=config['threshold'],
                title=zoom_title,
                output_dir=output_dir,
                save_png=True,
                save_html=False,
                show_plot=False,
                filename_prefix=f"zoom_{error_type}_",
                confidence=zoom_confidence,
                original_data=zoom_original_data  # Pass original data for comparison
            )
            
            print(f"  Zoom plot for {error_type} saved to: {zoom_png}")
            
        except Exception as e:
            print(f"Error creating zoom plot for {error_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def create_simple_anomaly_zoom_plots(val_data, predictions, error_generator, station_id, output_dir, original_val_data=None):
    """
    Create simplified zoom plots for each type of injected error showing only:
    - Original data (if available)
    - Modified data (with injected errors) 
    - Model predictions
    - Error period boundaries
    
    No z-scores, detection markers, or confidence levels - just pure model behavior analysis.
    
    Args:
        val_data: Validation data with synthetic errors
        predictions: Model predictions
        error_generator: SyntheticErrorGenerator instance with error_periods
        station_id: Station identifier
        output_dir: Output directory for plots
        original_val_data: Original validation data before error injection
    """
    if not hasattr(error_generator, 'error_periods') or not error_generator.error_periods:
        print("No error periods found for simple zoom plots")
        return
    
    print(f"\nCreating simple zoom plots for {len(error_generator.error_periods)} injected errors...")
    
    # Group error periods by type
    error_types = {}
    for period in error_generator.error_periods:
        error_type = period.error_type
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(period)
    
    # Create zoom plots for each error type (take first occurrence of each type)
    for error_type, periods in error_types.items():
        try:
            period = periods[0]  # Take first period of this error type
            
            print(f"Creating simple zoom plot for {error_type} error...")
            
            # Calculate buffer (6 hours before and after for more context)
            buffer_hours = 22  # Increased from 2 to 6 hours
            buffer_steps = buffer_hours * 4  # 15-min intervals
            
            # Find the period in the validation data
            start_time = period.start_time
            end_time = period.end_time
            
            # Get indices for the error period
            period_mask = (val_data.index >= start_time) & (val_data.index <= end_time)
            if not period_mask.any():
                print(f"ERROR: Error period not found in validation data for {error_type}")
                continue
                
            # Find start and end indices with buffer
            period_indices = np.where(period_mask)[0]
            start_idx = max(0, period_indices[0] - buffer_steps)
            end_idx = min(len(val_data), period_indices[-1] + buffer_steps)
            
            # Create zoom data
            zoom_data = val_data.iloc[start_idx:end_idx].copy()
            zoom_predictions = predictions[start_idx:end_idx] if len(predictions) > end_idx else predictions[start_idx:]
            
            # Extract original data if available
            zoom_original_data = None
            if original_val_data is not None:
                zoom_original_data = original_val_data.iloc[start_idx:end_idx].copy()
            
            # Create the plot with clean thesis-friendly style
            plt.style.use('default')  # Use clean default style
            plt.rcParams['font.size'] = 16          # Larger base font
            plt.rcParams['axes.labelsize'] = 18     # Larger axis labels
            plt.rcParams['xtick.labelsize'] = 14    # Larger tick labels
            plt.rcParams['ytick.labelsize'] = 14    # Larger tick labels
            plt.rcParams['legend.fontsize'] = 16    # Larger legend
            plt.rcParams['lines.linewidth'] = 2.5   # Thicker lines
            
            fig, ax = plt.subplots(figsize=(16, 10))  # Larger figure
            
            # Plot original data (if available)
            if zoom_original_data is not None:
                ax.plot(zoom_original_data.index, zoom_original_data['vst_raw'], 
                       'b-', label='Original Data', linewidth=2.5, alpha=0.8)
            
            # Plot modified data (with synthetic errors)
            ax.plot(zoom_data.index, zoom_data['vst_raw'], 
                   'r-', label='Modified Data', linewidth=2.5, alpha=0.8)
            
            # Plot model predictions
            ax.plot(zoom_data.index, zoom_predictions, 
                   'g-', label='Model Predictions', linewidth=3.0, alpha=0.9)
            
            # NO colored background for anomaly period - removed axvspan
            
            # NO title - removed set_title
            
            # Clean formatting
            ax.set_xlabel('Date', fontweight='bold', fontsize=18)
            ax.set_ylabel('Water Level [mm]', fontweight='bold', fontsize=18)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(False)  # NO grid lines
            
            # Clean axis styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            
            # Format x-axis with clean time labels
          #  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
           #ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  # Every day
           # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # Minor ticks every 12 hours
          #  plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
            
            # Save the plot
            plt.tight_layout()
            
            output_path = Path(output_dir) / f"simple_zoom_{error_type}_{station_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Simple zoom plot for {error_type} saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating simple zoom plot for {error_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue 

def plot_water_level_anomalies(
    test_data,
    predictions,
    z_scores,
    anomalies,
    threshold,
    title="Water Level Forecasting with Anomalies",
    output_dir=None,
    save_png=True,
    save_html=True,
    show_plot=True,
    sequence_length=None,
    filename_prefix="",
    confidence=None,
    original_data=None,  # Add parameter for original data
    ground_truth_flags=None,  # New: ground truth anomaly flags
    edt_data=None  # Add parameter for EDT reference data
):
    """
    Creates a plot showing water level data, predictions, z-scores, and detected anomalies.
    Now includes original data (before error injection) and EDT reference data if provided.
    
    Args:
        test_data (pd.DataFrame): DataFrame containing water level data with datetime index.
                                 Should have a 'vst_raw' column for water levels.
        predictions (pd.Series or np.array): Predicted water level values.
        z_scores (np.array): Z-scores calculated from residuals.
        anomalies (np.array): Boolean array indicating anomaly points.
        threshold (float): Anomaly detection threshold.
        title (str): Plot title.
        output_dir (str or Path): Directory to save output files.
        save_png (bool): Whether to save plot as PNG.
        save_html (bool): Whether to save interactive plot as HTML.
        show_plot (bool): Whether to display the plot.
        sequence_length (int or None): Length of the sequence to pad.
        filename_prefix (str): Prefix for output filenames.
        confidence (np.array or None): Confidence levels for anomalies.
        original_data (pd.DataFrame or None): Original data before error injection.
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path("results/anomaly_detection")
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate meaningful filename
    base_filename = f"{filename_prefix}water_level_anomalies" if filename_prefix else "water_level_anomalies"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Ensure data is in the right format
    if isinstance(test_data, pd.DataFrame):
        modified_values = test_data['vst_raw']
    else:
        modified_values = test_data

    # Get original values if provided
    original_values = original_data['vst_raw'] if original_data is not None else None

    n = len(modified_values)

    # Infer sequence_length if not provided
    if sequence_length is None:
        # If predictions is shorter than actual_values, infer the offset
        if len(predictions) < n:
            sequence_length = n - len(predictions)
        else:
            sequence_length = 0

    # Pad predictions, z_scores, anomalies to match test_data length
    full_predictions = pd.Series(np.full(n, np.nan), index=modified_values.index)
    full_z_scores = np.full(n, np.nan)
    full_anomalies = np.zeros(n, dtype=bool)
    full_confidence = np.full(n, 'Normal', dtype=object) if confidence is not None else None

    # Place predictions/z_scores/anomalies after sequence_length
    valid_len = min(n - sequence_length, len(predictions))
    if valid_len > 0:
        full_predictions.iloc[sequence_length:sequence_length+valid_len] = predictions[:valid_len]
        full_z_scores[sequence_length:sequence_length+valid_len] = z_scores[:valid_len]
        full_anomalies[sequence_length:sequence_length+valid_len] = anomalies[:valid_len]
        if confidence is not None:
            full_confidence[sequence_length:sequence_length+valid_len] = confidence[:valid_len]

    # Plot using matplotlib (for PNG)
    if save_png or show_plot:
        # Set up figure with single plot (removed z-score subplot)
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot original data if available
        if original_values is not None:
            ax1.plot(original_values.index, original_values.values, color='lightblue', linewidth=1, 
                    label='Original Water Levels', alpha=0.7)
        
        # Plot EDT reference data if available
        if edt_data is not None:
            ax1.plot(edt_data.index, edt_data.values, color='purple', linewidth=1.5, 
                    label='EDT Reference', alpha=0.8, linestyle='--')
        
        # Plot modified water levels
        ax1.plot(modified_values.index, modified_values.values, color='blue', linewidth=1, 
                label='Modified Water Levels')
        
        # Plot predictions
        ax1.plot(modified_values.index, full_predictions.values, color='green', linewidth=1, 
                label='Model Predictions')
        
        # Mark anomalies with different colors based on confidence
        if confidence is not None:
            # Define confidence colors
            confidence_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
            
            for conf_level, color in confidence_colors.items():
                # Get indices for this confidence level
                conf_mask = (full_anomalies) & (full_confidence == conf_level) & ~np.isnan(full_predictions)
                if np.any(conf_mask):
                    conf_indices = modified_values.index[conf_mask]
                    ax1.scatter(conf_indices, modified_values.loc[conf_indices], 
                               color=color, s=50, marker='o', label=f'{conf_level} Confidence Anomalies',
                               edgecolors='black', linewidth=0.5)
        else:
            # Mark anomalies only where predictions exist
            valid_anomaly_indices = modified_values.index[full_anomalies & ~np.isnan(full_predictions)]
            if len(valid_anomaly_indices) > 0:
                ax1.scatter(valid_anomaly_indices, modified_values.loc[valid_anomaly_indices], 
                           color='red', s=50, marker='o', label='Detected Anomalies')
        
        # Format plot
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('Date', fontsize=14)
        ax1.set_ylabel('Water Level [mm]', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(False)  # Remove grid lines
        
        # Format date axis to show month-year only
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks every month
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the figure if requested
        png_path = None
        if save_png:
            png_path = output_dir / f"{base_filename}_{timestamp}.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {png_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    # Create interactive Plotly visualization (for HTML)
    html_path = None
    if save_html:
        # Debug prints for ground truth flags
        if ground_truth_flags is not None:
            print("DEBUG: Number of ground truth anomalies:", np.sum(ground_truth_flags))
            print("DEBUG: Indices of ground truth anomalies:", test_data.index[ground_truth_flags == 1])
        
        # Create figure with single plot (removed z-score subplot)
        fig = go.Figure()
        
        # Original water levels if available
        if original_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=original_values.index,
                    y=original_values.values,
                    name="Original Water Levels",
                    line=dict(color='lightblue', width=1)
                )
            )
        
        # EDT reference data if available
        if edt_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=edt_data.index,
                    y=edt_data.values,
                    name="EDT Reference",
                    line=dict(color='purple', width=1.5, dash='dash')
                )
            )
        
        # Modified water levels
        fig.add_trace(
            go.Scatter(
                x=modified_values.index,
                y=modified_values.values,
                name="Modified Water Levels",
                line=dict(color='blue', width=1)
            )
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=modified_values.index,
                y=full_predictions.values,
                name="Model Predictions",
                line=dict(color='green', width=1)
            )
        )
        
        # Ground truth anomalies if available
        if ground_truth_flags is not None:
            # Plot ground truth anomalies as blue dots
            if hasattr(test_data, 'index'):
                gt_indices = test_data.index[ground_truth_flags == 1]
                gt_values = test_data['vst_raw'].loc[gt_indices]
            else:
                gt_indices = np.where(ground_truth_flags == 1)[0]
                gt_values = test_data[gt_indices]
            fig.add_trace(
                go.Scatter(
                    x=gt_indices,
                    y=gt_values,
                    mode='markers',
                    marker=dict(color='blue', size=7, symbol='diamond'),
                    name='Ground Truth Anomalies',
                )
            )
        
        # Add detected anomalies as scatter points only where predictions exist
        valid_anomaly_indices = modified_values.index[full_anomalies & ~np.isnan(full_predictions)]
        if len(valid_anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_anomaly_indices,
                    y=modified_values.loc[valid_anomaly_indices],
                    mode='markers',
                    marker=dict(color='red', size=5, symbol='circle'),
                    name=f"Detected Anomalies ({len(valid_anomaly_indices)})"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,  # Reduced height since only one plot
            width=1200,
            xaxis_title="Date",
            yaxis_title="Water Level [mm]",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Remove grid lines and format x-axis for month-year
        fig.update_xaxes(showgrid=False, dtick="M3", tickformat="%b %Y")  # Every 3 months, month-year format
        fig.update_yaxes(showgrid=False)
        
        # Save HTML file
        html_path = output_dir / f"{base_filename}_{timestamp}.html"
        fig.write_html(str(html_path))
        print(f"Interactive plot saved as {html_path}")
    
    return png_path, html_path 