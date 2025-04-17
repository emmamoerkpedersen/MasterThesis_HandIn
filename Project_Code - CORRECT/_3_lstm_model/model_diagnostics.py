import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from matplotlib.gridspec import GridSpec

def analyze_residuals(actual, predictions, output_dir=None, station_id='main'):
    """
    Analyze the residuals (errors) between actual and predicted water levels.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate residuals
    residuals = actual - predictions
    
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # 1. Time series of residuals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(residuals.index, residuals.values, 'o-', markersize=2, alpha=0.6, color='#1f77b4')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_title(f'Residuals Over Time - Station {station_id}', fontweight='bold')
    ax1.set_ylabel('Residual (Actual - Predicted) [mm]', fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of residuals
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(residuals, kde=True, ax=ax2, color='#1f77b4', stat='density')
    ax2.set_title('Distribution of Residuals', fontweight='bold')
    ax2.set_xlabel('Residual [mm]', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    
    # Add normal distribution fit
    from scipy import stats
    mu, std = stats.norm.fit(residuals.dropna())
    x = np.linspace(residuals.min(), residuals.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    ax2.plot(x, p, 'k--', linewidth=1.5, label=f'Normal: μ={mu:.1f}, σ={std:.1f}')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax2.legend()
    
    # 3. Q-Q plot to check normality
    ax3 = fig.add_subplot(gs[1, 1])
    from scipy import stats
    stats.probplot(residuals.dropna(), plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals', fontweight='bold')
    ax3.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontweight='bold')
    
    # 4. Residuals vs Actual
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(actual, residuals, alpha=0.3, s=10, color='#1f77b4')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_title('Residuals vs Actual Values', fontweight='bold')
    ax4.set_xlabel('Actual Water Level [mm]', fontweight='bold')
    ax4.set_ylabel('Residual [mm]', fontweight='bold')
    
    # 5. Residuals vs Predicted
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(predictions, residuals, alpha=0.3, s=10, color='#1f77b4')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax5.set_title('Residuals vs Predicted Values', fontweight='bold')
    ax5.set_xlabel('Predicted Water Level [mm]', fontweight='bold')
    ax5.set_ylabel('Residual [mm]', fontweight='bold')
    
    # Add overall statistics as text
    stats_text = (
        f"RMSE: {np.sqrt(mean_squared_error(actual, predictions)):.2f} mm\n"
        f"MAE: {mean_absolute_error(actual, predictions):.2f} mm\n"
        f"R²: {r2_score(actual, predictions):.4f}\n"
        f"Mean Error: {np.mean(residuals):.2f} mm\n"
        f"Error Std Dev: {np.std(residuals):.2f} mm"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'residual_analysis_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved residual analysis to: {output_path}")
    return output_path

def analyze_peak_detection(actual, predictions, rainfall=None, output_dir=None, station_id='main', threshold_percentile=80):
    """
    Analyze how well the model detects and predicts peak water levels.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        rainfall: Optional series of rainfall data for the same period
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        threshold_percentile: Percentile to define a "peak" (default: 80)
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Define peaks as values above the threshold percentile
    peak_threshold = np.percentile(actual, threshold_percentile)
    
    # Identify peaks
    peak_mask = actual > peak_threshold
    peak_dates = actual.index[peak_mask]
    
    # Calculate errors for peak and non-peak periods
    peak_errors = np.abs(actual[peak_mask] - predictions[peak_mask])
    non_peak_errors = np.abs(actual[~peak_mask] - predictions[~peak_mask])
    
    # Calculate metrics
    peak_mae = np.mean(peak_errors)
    peak_rmse = np.sqrt(np.mean(np.square(peak_errors)))
    non_peak_mae = np.mean(non_peak_errors)
    non_peak_rmse = np.sqrt(np.mean(np.square(non_peak_errors)))
    
    # Create figure with subplots
    if rainfall is not None:
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(3, 2, height_ratios=[0.5, 2, 1])
        
        # Rainfall subplot
        ax_rain = fig.add_subplot(gs[0, :])
        ax_rain.bar(rainfall.index, rainfall.values, width=0.5, alpha=0.6, color='#1f77b4')
        ax_rain.set_title('Rainfall', fontweight='bold')
        ax_rain.set_ylabel('Rainfall [mm]', fontweight='bold')
        ax_rain.grid(True, alpha=0.3)
        ax_rain.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_rain.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_rain.xaxis.get_majorticklabels(), visible=False)
    else:
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, height_ratios=[2, 1])
    
    # Time series with peaks highlighted
    ax1 = fig.add_subplot(gs[1, :] if rainfall is not None else gs[0, :])
    ax1.plot(actual.index, actual.values, label='Actual', color='#1f77b4', linewidth=1.5)
    ax1.plot(predictions.index, predictions.values, label='Predicted', color='#d62728', linewidth=1.5, alpha=0.8)
    
    # Highlight peaks
    ax1.fill_between(actual.index, peak_threshold, actual.max() + 100, alpha=0.2, color='red', 
                    label=f'Peak Regions (>{threshold_percentile}th percentile)')
    ax1.axhline(y=peak_threshold, color='red', linestyle='--', alpha=0.8, 
               label=f'Peak Threshold: {peak_threshold:.0f} mm')
    
    ax1.set_title(f'Water Level Peaks Analysis - Station {station_id}', fontweight='bold')
    ax1.set_ylabel('Water Level [mm]', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Error comparison: peak vs non-peak
    ax2 = fig.add_subplot(gs[2, 0] if rainfall is not None else gs[1, 0])
    bar_data = [peak_mae, non_peak_mae]
    bars = ax2.bar(['Peak Periods', 'Non-Peak Periods'], bar_data, color=['#ff9999', '#66b3ff'])
    ax2.set_title('MAE Comparison', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error [mm]', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add text labels on bars
    for bar, value in zip(bars, bar_data):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 5, f'{value:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Error comparison: RMSE
    ax3 = fig.add_subplot(gs[2, 1] if rainfall is not None else gs[1, 1])
    bar_data = [peak_rmse, non_peak_rmse]
    bars = ax3.bar(['Peak Periods', 'Non-Peak Periods'], bar_data, color=['#ff9999', '#66b3ff'])
    ax3.set_title('RMSE Comparison', fontweight='bold')
    ax3.set_ylabel('Root Mean Squared Error [mm]', fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add text labels on bars
    for bar, value in zip(bars, bar_data):
        ax3.text(bar.get_x() + bar.get_width()/2, value + 5, f'{value:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add overall statistics as text
    peak_count = sum(peak_mask)
    total_count = len(actual)
    peak_percent = (peak_count / total_count) * 100
    
    stats_text = (
        f"Total observations: {total_count}\n"
        f"Peak observations: {peak_count} ({peak_percent:.1f}%)\n"
        f"Peak threshold: {peak_threshold:.1f} mm\n"
        f"Peak MAE: {peak_mae:.1f} mm (vs {non_peak_mae:.1f} mm for non-peak)\n"
        f"Peak RMSE: {peak_rmse:.1f} mm (vs {non_peak_rmse:.1f} mm for non-peak)"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'peak_analysis_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved peak detection analysis to: {output_path}")
    return output_path

def create_error_heatmap(actual, predictions, output_dir=None, station_id='main'):
    """
    Create a heatmap showing prediction errors by month and day of week.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate errors
    errors = actual - predictions
    errors_df = pd.DataFrame({'error': errors, 'abs_error': np.abs(errors)})
    
    # Add time features
    errors_df['month'] = errors.index.month
    errors_df['day_of_week'] = errors.index.dayofweek
    errors_df['hour'] = errors.index.hour
    
    # Month-Day of Week Mean Absolute Error
    month_dow_pivot = errors_df.pivot_table(
        values='abs_error', 
        index='month', 
        columns='day_of_week', 
        aggfunc='mean'
    )
    
    # Month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Month-Day of Week heatmap
    sns.heatmap(
        month_dow_pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd', 
        ax=axes[0],
        xticklabels=day_names,
        yticklabels=month_names,
        cbar_kws={'label': 'Mean Absolute Error [mm]'}
    )
    axes[0].set_title('Mean Absolute Error by Month and Day of Week', fontweight='bold')
    axes[0].set_xlabel('Day of Week', fontweight='bold')
    axes[0].set_ylabel('Month', fontweight='bold')
    
    # Create month-only error boxplot
    month_errors = errors_df.groupby('month')['error'].apply(list)
    
    # Plot boxplot of errors by month
    month_plot = axes[1]
    
    # Convert to format needed for boxplot
    month_data = [month_errors[m] for m in range(1, 13)]
    
    # Create boxplot
    boxplot = month_plot.boxplot(
        month_data, 
        labels=month_names,
        patch_artist=True,
        medianprops={'color': 'black'},
        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3}
    )
    
    # Customize boxplot colors - gradient from green to red
    colors = plt.cm.YlOrRd(np.linspace(0.1, 0.8, 12))
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    
    month_plot.set_title('Error Distribution by Month', fontweight='bold')
    month_plot.set_xlabel('Month', fontweight='bold')
    month_plot.set_ylabel('Error (Actual - Predicted) [mm]', fontweight='bold')
    month_plot.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    month_plot.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(month_plot.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'error_heatmap_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error heatmap to: {output_path}")
    return output_path

def analyze_response_to_rainfall(actual, predictions, rainfall, output_dir=None, station_id='main'):
    """
    Analyze how well the model responds to rainfall events.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        rainfall: Series containing rainfall data
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Identify significant rainfall events (e.g., > 10mm)
    significant_rainfall = 10  # mm
    rainfall_events = rainfall[rainfall > significant_rainfall]
    
    if len(rainfall_events) == 0:
        print("No significant rainfall events found. Adjusting threshold...")
        # If no events with > 10mm, use the top 10 rainfall events
        rainfall_events = rainfall.nlargest(10)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 2])
    
    # Rainfall subplot
    ax_rain = fig.add_subplot(gs[0])
    ax_rain.bar(rainfall.index, rainfall.values, width=0.5, alpha=0.6, color='#1f77b4')
    
    # Highlight significant rainfall events
    for event_time in rainfall_events.index:
        ax_rain.axvline(x=event_time, color='red', alpha=0.5, linestyle='--')
    
    ax_rain.set_title('Rainfall with Significant Events', fontweight='bold')
    ax_rain.set_ylabel('Rainfall [mm]', fontweight='bold')
    ax_rain.grid(True, alpha=0.3)
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rain.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_rain.xaxis.get_majorticklabels(), visible=False)
    
    # Water level subplot
    ax_wl = fig.add_subplot(gs[1])
    ax_wl.plot(actual.index, actual.values, label='Actual', color='#1f77b4', linewidth=1.5)
    ax_wl.plot(predictions.index, predictions.values, label='Predicted', color='#d62728', linewidth=1.5, alpha=0.8)
    
    # Highlight the same rainfall events
    for event_time in rainfall_events.index:
        ax_wl.axvline(x=event_time, color='red', alpha=0.5, linestyle='--')
    
    ax_wl.set_title(f'Water Level Response to Rainfall - Station {station_id}', fontweight='bold')
    ax_wl.set_ylabel('Water Level [mm]', fontweight='bold')
    ax_wl.set_xlabel('Date', fontweight='bold')
    ax_wl.legend(loc='upper right')
    ax_wl.grid(True, alpha=0.3)
    ax_wl.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_wl.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_wl.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Calculate post-rainfall errors
    window_days = 3  # Days after rainfall to analyze
    post_rain_errors = []
    normal_errors = []
    
    # Create a mask for all post-rainfall periods
    post_rain_mask = pd.Series(False, index=actual.index)
    
    for event_time in rainfall_events.index:
        window_end = event_time + pd.Timedelta(days=window_days)
        post_rain_period = (actual.index > event_time) & (actual.index <= window_end)
        post_rain_mask = post_rain_mask | post_rain_period
        
        # Calculate errors for this event
        if any(post_rain_period):
            event_errors = actual[post_rain_period] - predictions[post_rain_period]
            post_rain_errors.extend(event_errors.values)
    
    # Normal periods are those not in post-rainfall windows
    normal_period = ~post_rain_mask
    normal_errors = (actual[normal_period] - predictions[normal_period]).values
    
    # Calculate metrics
    post_rain_mae = np.mean(np.abs(post_rain_errors)) if post_rain_errors else np.nan
    normal_mae = np.mean(np.abs(normal_errors)) if len(normal_errors) > 0 else np.nan
    post_rain_rmse = np.sqrt(np.mean(np.square(post_rain_errors))) if post_rain_errors else np.nan
    normal_rmse = np.sqrt(np.mean(np.square(normal_errors))) if len(normal_errors) > 0 else np.nan
    
    # Add statistics as text
    stats_text = (
        f"Significant rainfall threshold: {significant_rainfall} mm\n"
        f"Number of rainfall events: {len(rainfall_events)}\n"
        f"Post-rainfall window: {window_days} days\n"
        f"Post-rainfall MAE: {post_rain_mae:.1f} mm\n"
        f"Normal period MAE: {normal_mae:.1f} mm\n"
        f"Post-rainfall RMSE: {post_rain_rmse:.1f} mm\n"
        f"Normal period RMSE: {normal_rmse:.1f} mm"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'rainfall_response_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved rainfall response analysis to: {output_path}")
    return output_path

def create_taylor_diagram(actual, predictions, station_id='main', output_dir=None):
    """
    Create a Taylor diagram to visualize model performance.
    If skill_metrics package is not available, creates a simplified alternative plot.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        station_id: Station identifier for plot title
        output_dir: Optional output directory path
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Drop NaN values
    valid_mask = ~(np.isnan(actual) | np.isnan(predictions))
    actual_valid = actual[valid_mask].values
    predictions_valid = predictions[valid_mask].values
    
    # Calculate statistics
    std_ref = np.std(actual_valid)
    std_pred = np.std(predictions_valid)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(actual_valid, predictions_valid)[0, 1]
    
    # Calculate normalized RMSD (unbiased)
    centered_actual = actual_valid - np.mean(actual_valid)
    centered_pred = predictions_valid - np.mean(predictions_valid)
    norm_rmsd = np.sqrt(np.mean((centered_actual - centered_pred) ** 2)) / std_ref
    
    # Calculate metrics for statistics text
    rmse = np.sqrt(mean_squared_error(actual_valid, predictions_valid))
    mae = mean_absolute_error(actual_valid, predictions_valid)
    r2 = r2_score(actual_valid, predictions_valid)
    
    # Try to use skill_metrics if available
    try:
        import skill_metrics as sm
        
        # Setup figure
        plt.figure(figsize=(10, 8))
        
        # Define labels and colors
        labels = ['Observations', 'Predictions']
        colors = ['k', 'r']  # Use single-letter codes for black and red
        
        # Data for Taylor diagram
        sdev = np.array([std_ref, std_pred])
        crmsd = np.array([0.0, norm_rmsd * std_ref])
        ccoef = np.array([1.0, corr])
        
        # Create Taylor diagram
        sm.taylor_diagram(sdev, crmsd, ccoef, 
                         markerLabel=labels, markerColor=colors,
                         markerLegend='on', styleOBS='-', 
                         titleOBS='Reference', showlabelsRMS='on')
        
        # Add title
        plt.title(f'Taylor Diagram - Station {station_id}', fontweight='bold', fontsize=14)
        
    except ImportError:
        # Create alternative visualization
        print("SkillMetrics package not found. Creating alternative performance visualization.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a scatter plot
        ax.scatter([std_ref], [1.0], s=150, c='blue', marker='*', label='Observations')
        ax.scatter([std_pred], [corr], s=100, c='red', marker='o', label='Predictions')
        
        # Add labels for points
        ax.annotate('Observations', xy=(std_ref, 1.0), xytext=(std_ref + 10, 1.0),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax.annotate('Predictions', xy=(std_pred, corr), xytext=(std_pred + 10, corr - 0.1),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        # Set up the axes
        ax.set_xlabel('Standard Deviation', fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontweight='bold')
        ax.set_title(f'Model Performance Metrics - Station {station_id}', fontweight='bold')
        
        # Add reference lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=std_ref, color='gray', linestyle='--', alpha=0.7)
        
        # Set axis limits
        ax.set_ylim(0, 1.1)
        max_std = max(std_ref, std_pred) * 1.2
        ax.set_xlim(0, max_std)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right')
    
    # Add statistics text
    stats_text = (
        f"Model Performance Metrics\n"
        f"---------------------------\n"
        f"RMSE: {rmse:.2f} mm\n"
        f"MAE: {mae:.2f} mm\n"
        f"R²: {r2:.4f}\n"
        f"Correlation: {corr:.4f}\n"
        f"Std. Dev. (Obs): {std_ref:.2f} mm\n"
        f"Std. Dev. (Pred): {std_pred:.2f} mm"
    )
    
    # Position statistics text in the upper left corner
    plt.figtext(0.02, 0.95, stats_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    output_path = output_dir / f'performance_metrics_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance metrics visualization to: {output_path}")
    return output_path

def analyze_specific_events(actual, predictions, rainfall=None, output_dir=None, station_id='main', n_events=5):
    """
    Create detailed visualizations of top flood events to analyze model performance during these critical periods.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        rainfall: Optional series of rainfall data for the same period
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        n_events: Number of top events to analyze
        
    Returns:
        Path to the saved PNG file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Find top n_events peaks in the data
    # First, we need to identify isolated peaks by smoothing the data a bit
    from scipy.signal import find_peaks
    
    # Use a rolling window to smooth the data
    actual_smoothed = actual.rolling(window=24, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    
    # Find peaks in the smoothed data
    peaks, _ = find_peaks(actual_smoothed, distance=72, prominence=50)  # 72 hours between peaks, prominence of 50mm
    
    if len(peaks) == 0:
        print("No significant peaks found. Adjusting parameters...")
        peaks, _ = find_peaks(actual_smoothed, distance=48, prominence=20)  # Relax criteria
        
    if len(peaks) == 0:
        print("Still no peaks found. Using the highest points in the data.")
        # Just use the top n_events points
        peaks = actual.nlargest(n_events).index
        peak_values = actual[peaks]
    else:
        peak_indices = actual.index[peaks]
        peak_values = actual[peak_indices]
        
        # Sort peaks by height and get the top n_events
        sorted_peaks = sorted(zip(peak_indices, peak_values), key=lambda x: x[1], reverse=True)
        peak_indices = [p[0] for p in sorted_peaks[:n_events]]
        peak_values = [p[1] for p in sorted_peaks[:n_events]]
    
    # Create figure for all events
    fig, axes = plt.subplots(n_events, 1, figsize=(12, 4*n_events))
    
    # If only one event, make axes iterable
    if n_events == 1:
        axes = [axes]
    
    # Analyze each peak event
    for i, peak_time in enumerate(peak_indices[:n_events]):
        # Define window around peak (7 days before and after)
        window_start = peak_time - pd.Timedelta(days=7)
        window_end = peak_time + pd.Timedelta(days=7)
        
        # Get data in window
        window_mask = (actual.index >= window_start) & (actual.index <= window_end)
        actual_window = actual[window_mask]
        predictions_window = predictions[window_mask]
        
        # Calculate metrics for this event
        event_rmse = np.sqrt(mean_squared_error(actual_window, predictions_window.loc[actual_window.index]))
        event_mae = mean_absolute_error(actual_window, predictions_window.loc[actual_window.index])
        peak_error = abs(actual_window.max() - predictions_window.loc[actual_window.index].max())
        timing_error = abs((actual_window.idxmax() - predictions_window.loc[actual_window.index].idxmax()).total_seconds() / 3600)  # hours
        
        # Plot this event
        ax = axes[i]
        ax.plot(actual_window.index, actual_window, label='Actual', color='#1f77b4', linewidth=1.5)
        ax.plot(predictions_window.index, predictions_window, label='Predicted', color='#d62728', linewidth=1.5, alpha=0.8)
        
        # Mark the peak points
        ax.scatter([actual_window.idxmax()], [actual_window.max()], color='blue', s=80, zorder=5, label='Actual Peak')
        ax.scatter([predictions_window.idxmax()], [predictions_window.max()], color='red', s=80, zorder=5, label='Predicted Peak')
        
        # Add rainfall if available
        if rainfall is not None:
            rainfall_window = rainfall[window_mask]
            ax2 = ax.twinx()
            ax2.bar(rainfall_window.index, rainfall_window, alpha=0.3, color='#2ca02c', width=0.5)
            ax2.set_ylabel('Rainfall (mm)', color='#2ca02c')
            ax2.tick_params(axis='y', labelcolor='#2ca02c')
            ax2.set_ylim(bottom=0)
            # Invert y-axis for rainfall to show bars going down from top
            current_ylim = ax2.get_ylim()
            ax2.set_ylim(current_ylim[1], current_ylim[0])
        
        # Set labels and title
        event_date = peak_time.strftime('%Y-%m-%d')
        ax.set_title(f'Flood Event {i+1}: {event_date} - Peak: {actual_window.max():.0f}mm', fontweight='bold')
        ax.set_ylabel('Water Level (mm)', fontweight='bold')
        
        if i == n_events - 1:  # Only add xlabel to the bottom plot
            ax.set_xlabel('Date', fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        # Add metrics as text
        metrics_text = (
            f"Event RMSE: {event_rmse:.1f}mm\n"
            f"Event MAE: {event_mae:.1f}mm\n"
            f"Peak Magnitude Error: {peak_error:.1f}mm\n"
            f"Peak Timing Error: {timing_error:.1f} hours"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'event_analysis_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved specific event analysis to: {output_path}")
    return output_path

def generate_all_diagnostics(actual, predictions, output_dir=None, station_id='main', rainfall=None, n_event_plots=3):
    """
    Generate a comprehensive set of diagnostic visualizations for a water level prediction model.
    
    Args:
        actual: Series containing actual water level values
        predictions: Series containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot titles
        rainfall: Optional series of rainfall data for the same period
        n_event_plots: Number of top water level events to analyze (default: 3)
        
    Returns:
        dict: Dictionary with paths to all generated visualization files
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
    
    # Make sure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all visualization paths
    visualization_paths = {}
    
    # Generate timestamp for logging
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Generating diagnostic visualizations for station {station_id}...")
    
    try:
        # 1. Analyze residuals
        print("  - Analyzing prediction residuals...")
        residuals_path = analyze_residuals(
            actual=actual,
            predictions=predictions,
            output_dir=output_dir,
            station_id=station_id
        )
        visualization_paths['residuals'] = residuals_path
        
        # 2. Analyze peak detection performance
        print("  - Analyzing peak detection performance...")
        if rainfall is not None:
            peak_path = analyze_peak_detection(
                actual=actual,
                predictions=predictions,
                rainfall=rainfall,
                output_dir=output_dir,
                station_id=station_id
            )
        else:
            peak_path = analyze_peak_detection(
                actual=actual,
                predictions=predictions,
                output_dir=output_dir,
                station_id=station_id
            )
        visualization_paths['peak_detection'] = peak_path
        
        # 3. Create error heatmap by time patterns
        print("  - Creating error pattern heatmap...")
        heatmap_path = create_error_heatmap(
            actual=actual,
            predictions=predictions,
            output_dir=output_dir,
            station_id=station_id
        )
        visualization_paths['error_heatmap'] = heatmap_path
        
        # 4. Analyze specific flood events
        print(f"  - Analyzing top {n_event_plots} water level events...")
        events_path = analyze_specific_events(
            actual=actual,
            predictions=predictions,
            rainfall=rainfall,
            output_dir=output_dir,
            station_id=station_id,
            n_events=n_event_plots
        )
        visualization_paths['specific_events'] = events_path
        
        # 5. Analyze model response to rainfall if rainfall data is available
        if rainfall is not None:
            print("  - Analyzing model response to rainfall events...")
            rainfall_path = analyze_response_to_rainfall(
                actual=actual,
                predictions=predictions,
                rainfall=rainfall,
                output_dir=output_dir,
                station_id=station_id
            )
            visualization_paths['rainfall_response'] = rainfall_path
        
        # 6. Create performance metrics visualization (Taylor diagram or alternative)
        try:
            print("  - Creating performance metrics visualization...")
            metrics_path = create_taylor_diagram(
                actual=actual,
                predictions=predictions,
                station_id=station_id,
                output_dir=output_dir
            )
            visualization_paths['performance_metrics'] = metrics_path
        except Exception as e:
            print(f"    Error creating performance metrics visualization: {str(e)}")
        
        # Success message
        print(f"\nAll diagnostic visualizations saved to: {output_dir}")
        
        # Create an HTML index file that links to all visualizations
        try:
            index_path = create_diagnostics_index(
                visualization_paths=visualization_paths,
                station_id=station_id,
                output_dir=output_dir
            )
            visualization_paths['index'] = index_path
        except Exception as e:
            print(f"Error creating index file: {str(e)}")
        
    except Exception as e:
        print(f"Error during diagnostic visualization generation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return visualization_paths

def generate_comparative_diagnostics(actual, predictions_dict, output_dir=None, station_id='main', rainfall=None, n_event_plots=3):
    """
    Generate comparative diagnostic visualizations for multiple prediction models.
    
    Args:
        actual: Series containing actual water level values
        predictions_dict: Dictionary of prediction Series with model names as keys
        output_dir: Optional output directory path
        station_id: Station identifier for plot titles
        rainfall: Optional series of rainfall data for the same period
        n_event_plots: Number of top water level events to analyze (default: 3)
        
    Returns:
        dict: Dictionary with paths to all generated visualization files, organized by model
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("Project_Code - CORRECT/results/diagnostics")
    
    # Make sure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for comparative visualizations
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Dictionary to store visualization paths for all models
    all_visualization_paths = {}
    
    # Generate timestamp for logging
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Generating comparative diagnostic visualizations for {len(predictions_dict)} models...")
    
    try:
        # Generate individual diagnostics for each model
        for model_name, predictions in predictions_dict.items():
            print(f"\nAnalyzing model: {model_name}")
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            visualization_paths = generate_all_diagnostics(
                actual=actual,
                predictions=predictions,
                output_dir=model_dir,
                station_id=f"{station_id}_{model_name}",
                rainfall=rainfall,
                n_event_plots=n_event_plots
            )
            
            all_visualization_paths[model_name] = visualization_paths
        
        # Create a comparative index HTML file
        index_path = create_comparative_index(
            all_visualization_paths=all_visualization_paths,
            station_id=station_id,
            output_dir=comparison_dir
        )
        
        print(f"\nComparative analysis complete. Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during comparative diagnostic visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return all_visualization_paths

def create_diagnostics_index(visualization_paths, station_id, output_dir):
    """
    Create an HTML index file linking to all diagnostic visualizations.
    
    Args:
        visualization_paths: Dictionary with paths to visualization files
        station_id: Station identifier
        output_dir: Output directory
        
    Returns:
        Path to the created index HTML file
    """
    output_dir = Path(output_dir)
    index_path = output_dir / f"diagnostics_index_{station_id}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Water Level Model Diagnostics - Station {station_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .visualization {{ margin-bottom: 30px; }}
            .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
            .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            @media (max-width: 1200px) {{ .grid-container {{ grid-template-columns: 1fr; }} }}
        </style>
    </head>
    <body>
        <h1>Water Level Model Diagnostics - Station {station_id}</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="grid-container">
    """
    
    # Add visualizations to the HTML content
    for viz_type, viz_path in visualization_paths.items():
        if viz_type == 'index':
            continue
            
        rel_path = viz_path.relative_to(output_dir)
        html_content += f"""
            <div class="visualization">
                <h2>{viz_type.replace('_', ' ').title()}</h2>
                <a href="{rel_path}" target="_blank">
                    <img src="{rel_path}" alt="{viz_type} visualization" />
                </a>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created diagnostics index at: {index_path}")
    return index_path

def create_comparative_index(all_visualization_paths, station_id, output_dir):
    """
    Create an HTML index file for comparing model diagnostics.
    
    Args:
        all_visualization_paths: Dictionary with visualization paths for each model
        station_id: Station identifier
        output_dir: Output directory
        
    Returns:
        Path to the created index HTML file
    """
    output_dir = Path(output_dir)
    index_path = output_dir / f"comparative_index_{station_id}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparative Model Diagnostics - Station {station_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .visualization {{ margin-bottom: 50px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Comparative Model Diagnostics - Station {station_id}</h1>
        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Get common visualization types across all models
    all_viz_types = set()
    for model_paths in all_visualization_paths.values():
        all_viz_types.update(model_paths.keys())
    
    # Remove 'index' from visualization types
    if 'index' in all_viz_types:
        all_viz_types.remove('index')
    
    # Organize visualizations by type for comparison
    for viz_type in sorted(all_viz_types):
        html_content += f"""
        <div class="visualization">
            <h2>{viz_type.replace('_', ' ').title()} Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Visualization</th>
                </tr>
        """
        
        for model_name, model_paths in all_visualization_paths.items():
            if viz_type in model_paths:
                viz_path = model_paths[viz_type]
                # Get path relative to the comparison directory
                try:
                    rel_path = Path("..") / viz_path.relative_to(output_dir.parent)
                except ValueError:
                    # If the path is not relative to output_dir.parent, use the absolute path
                    rel_path = viz_path
                
                html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>
                        <a href="{rel_path}" target="_blank">
                            <img src="{rel_path}" alt="{viz_type} for {model_name}" />
                        </a>
                    </td>
                </tr>
                """
        
        html_content += """
            </table>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created comparative index at: {index_path}")
    return index_path 