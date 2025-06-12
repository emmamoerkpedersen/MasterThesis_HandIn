import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
from typing import Optional, Dict, Any
from synthetic_error_config import SYNTHETIC_ERROR_PARAMS
from models.lstm_traditional.config import LSTM_CONFIG

# Add project root to sys.path for imports
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

from models.lstm_traditional.preprocessing_LSTM1 import DataPreprocessor

def mad_outlier_flags(train_series, val_series=None, threshold=3.0, window_size=16):
    """
    Detect outliers in a dataset using the Median Absolute Deviation (MAD) method with a rolling window.
    
    Args:
        train_series (pd.Series or np.ndarray): Training data to fit the MAD model.
        val_series (pd.Series or np.ndarray, optional): Validation data to apply the MAD model. If None, only train_series is used.
        threshold (float): Threshold for standardized MAD score to flag outliers (default: 3.0).
        window_size (int): Number of points in the rolling window (default: 16 for 4 hours at 15-min intervals).
    
    Returns:
        train_flags (np.ndarray): Boolean array (1=anomaly, 0=non-anomaly) for training data.
        val_flags (np.ndarray or None): Boolean array for validation data, or None if val_series is None.
        medians (np.ndarray): Rolling medians for training data.
        mads (np.ndarray): Rolling MADs for training data.
    """
    train_arr = np.asarray(train_series)
    n = len(train_arr)
    medians = np.full(n, np.nan)
    mads = np.full(n, np.nan)
    train_flags = np.zeros(n, dtype=int)
    for i in range(window_size, n):
        window = train_arr[i-window_size:i]
        window = window[~np.isnan(window)]
        if len(window) < window_size * 0.5:
            continue
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        medians[i] = median
        mads[i] = mad
        if mad > 0 and not np.isnan(train_arr[i]):
            z = np.abs(train_arr[i] - median) / mad
            if z > threshold:
                train_flags[i] = 1
    # Treat NaNs as non-anomalies
    train_flags[np.isnan(train_arr)] = 0

    val_flags = None
    val_medians = None
    val_mads = None
    if val_series is not None:
        val_arr = np.asarray(val_series)
        n_val = len(val_arr)
        val_flags = np.zeros(n_val, dtype=int)
        val_medians = np.full(n_val, np.nan)
        val_mads = np.full(n_val, np.nan)
        # Use rolling window from training data's last window
        combined = np.concatenate([train_arr[-window_size:], val_arr])
        for i in range(window_size, window_size + n_val):
            window = combined[i-window_size:i]
            window = window[~np.isnan(window)]
            if len(window) < window_size * 0.5:
                continue
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            val_medians[i-window_size] = median
            val_mads[i-window_size] = mad
            if mad > 0 and not np.isnan(val_arr[i-window_size]):
                z = np.abs(val_arr[i-window_size] - median) / mad
                if z > threshold:
                    val_flags[i-window_size] = 1
        val_flags[np.isnan(val_arr)] = 0
    return train_flags, val_flags, medians, mads, val_medians, val_mads

def plot_mad_anomalies(series, flags, medians, mads, threshold, window_size, title="Water Level with MAD Anomalies", save_path=None):
    times = series.index
    values = series.values
    lower = medians - threshold * mads
    upper = medians + threshold * mads
    plt.figure(figsize=(15, 5)) # dimension: height = 5, width = 10
    plt.fill_between(times, lower, upper, color='grey', alpha=0.3, label='Prediction Interval (PI)')
    plt.plot(times, values, color='black', label='Water Level')
    plt.scatter(times[flags == 1], values[flags == 1], color='red', label='Detected Anomalies', zorder=5, s=20)
    plt.xlabel("Time")
    plt.ylabel("Water Level (m msl)")
    plt.ylim(-500, 2000)
    plt.legend()
    plt.title(f"{title}\n(Rolling window: {window_size} steps)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.show()

def plot_mad_anomalies_html(series, flags, medians, mads, threshold, window_size, title="Water Level with MAD Anomalies", save_path=None):
    times = series.index
    values = series.values
    lower = medians - threshold * mads
    upper = medians + threshold * mads
    fig = go.Figure()
    # Prediction interval (shaded)
    fig.add_traces([
        go.Scatter(
            x=times, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'),
        go.Scatter(
            x=times, y=lower, mode='lines', fill='tonexty', fillcolor='rgba(128,128,128,0.3)',
            line=dict(width=0), name='Prediction Interval (PI)', hoverinfo='skip')
    ])
    # Water level
    fig.add_trace(go.Scatter(x=times, y=values, mode='lines', name='Water Level', line=dict(color='black')))
    # Detected anomalies
    fig.add_trace(go.Scatter(
        x=times[flags == 1], y=values[flags == 1], mode='markers',
        name='Detected Anomalies', marker=dict(color='red', size=7),
        hovertemplate='Time: %{x}<br>Value: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title=f"{title}<br>(Rolling window: {window_size} steps)",
        xaxis_title="Time",
        yaxis_title="Water Level (m msl)",
        yaxis=dict(range=[-500, 2000]),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white"
    )
    if save_path is not None:
        pio.write_html(fig, file=save_path, auto_open=False)
        print(f"HTML plot saved to: {save_path}")
    fig.show()

def create_anomaly_zoom_plots(series, flags, medians, mads, threshold, window_size, save_dir):
    """
    Create zoomed-in plots for 15 random detected anomalies (no grouping).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all anomaly indices
    anomaly_indices = np.where(flags == 1)[0]
    if len(anomaly_indices) == 0:
        print("No anomalies found to plot.")
        return
    
    # Randomly select up to 15 anomalies
    n_plots = min(15, len(anomaly_indices))
    selected_indices = random.sample(list(anomaly_indices), n_plots)
    #selected_indices.sort()  # Optional: sort for reproducibility
    print(f"\nGenerating zoom plots for {n_plots} random anomaly points...")
    
    for i, idx in enumerate(selected_indices):
        # Calculate buffer (4 hours before and after)
        buffer_steps = 50  # 4 hours at 15-min intervals
        start_idx = max(0, idx - buffer_steps)
        end_idx = min(len(series), idx + buffer_steps)
        
        # Create zoom data
        zoom_series = series.iloc[start_idx:end_idx]
        zoom_flags = flags[start_idx:end_idx]
        zoom_medians = medians[start_idx:end_idx]
        zoom_mads = mads[start_idx:end_idx]
        
        # Extract year(s) from the zoomed period
        if hasattr(zoom_series.index, 'year'):
            years = zoom_series.index.year
            unique_years = np.unique(years)
            year_str = ', '.join(str(y) for y in unique_years)
        else:
            year_str = ''
        
        # Calculate prediction intervals
        lower = zoom_medians - threshold * zoom_mads
        upper = zoom_medians + threshold * zoom_mads
        
        # Create plot
        plt.figure(figsize=(12, 5))
        plt.fill_between(zoom_series.index, lower, upper, color='grey', alpha=0.3, label='Prediction Interval (PI)')
        plt.plot(zoom_series.index, zoom_series.values, color='black', label='Water Level')
        plt.scatter(zoom_series.index[zoom_flags == 1], zoom_series.values[zoom_flags == 1], 
                   color='red', label='Detected Anomalies', zorder=5, s=20)
        
        # Format x-axis: hour:min at each tick
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Add a single date label below the x-axis (centered)
        if hasattr(zoom_series.index, 'to_pydatetime'):
            center_idx = len(zoom_series) // 2
            center_time = zoom_series.index[center_idx]
            date_str = center_time.strftime('%d %b %Y')
            # Place the date label below the x-axis
            plt.xlabel(f"Time\n{date_str}")
        else:
            plt.xlabel("Time")
        
        plt.ylabel("Water Level (m msl)")
        plt.ylim(0, 1000)
        plt.legend()
        plt.title(f"Zoom: Anomaly {i+1} (Year(s): {year_str})\n(Rolling window: {window_size} steps, Threshold: {threshold})")
        plt.tight_layout()
        
        # Save plot
        save_path = save_dir / f"zoom_anomaly_{i+1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zoom plot {i+1} saved to: {save_path}")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAD Outlier Detection for vst_raw")
    parser.add_argument('--station_id', type=str, default='21006846', help='Station ID to process')
    parser.add_argument('--threshold', type=float, default=50, help='MAD threshold for anomaly detection')
    parser.add_argument('--window_size', type=int, default=6*4, help='Rolling window size (default: 16 for 4 hours)')
    parser.add_argument('--error_multiplier', type=float, default=None, help='Error multiplier for synthetic errors. If not provided, no errors are injected.')
    parser.add_argument('--error_type', type=str, default='both', choices=['both', 'train', 'validation', 'none'], help='Which datasets to inject errors into (both, train, validation, or none)')
    args = parser.parse_args()

    config = LSTM_CONFIG.copy()
    if 'feature_stations' not in config:
        config['feature_stations'] = []

    preprocessor = DataPreprocessor(config)
    data = pd.read_pickle("/Users/emmamork/Desktop/Master Thesis/MasterThesis/Project_Code - CORRECT/data_utils/Sample data/preprocessed_data.pkl")    
    station_data = data.get(args.station_id)
    if not station_data:
        raise ValueError(f"Station ID {args.station_id} not found in the data.")
    df = pd.concat(station_data.values(), axis=1)
    start_date = pd.Timestamp('2010-01-04')
    end_date = pd.Timestamp('2025-01-07')
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    vst_raw = df['vst_raw']
    train_data = vst_raw[vst_raw.index.year < 2022]
    val_data = vst_raw[(vst_raw.index.year >= 2022) & (vst_raw.index.year <= 2023)]

    # --- Inject synthetic errors if requested ---
    if args.error_multiplier is not None and args.error_type != 'none':
        from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
        from utils.error_utils import configure_error_params, inject_errors_into_dataset
        print(f"\nInjecting synthetic errors with multiplier {args.error_multiplier:.1f}x...")
        print(f"Error injection mode: {args.error_type}")
        error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, args.error_multiplier)
        water_level_cols = ['vst_raw','vst_raw_feature']
        # Inject into train_data
        
        if args.error_type in ['both', 'train']:
            print("\nProcessing TRAINING data - injecting errors...")
            error_generator = SyntheticErrorGenerator(error_config)
            train_df = df[df.index.year < 2022].copy()
            train_data_with_errors, train_error_report = inject_errors_into_dataset(
                train_df, error_generator, f"{args.station_id}_train", water_level_cols
            )
            train_data = train_data_with_errors['vst_raw']
        # Inject into val_data
        if args.error_type in ['both', 'validation']:
            print("\nProcessing VALIDATION data - injecting errors...")
            validation_error_generator = SyntheticErrorGenerator(error_config)
            val_df = df[(df.index.year >= 2022) & (df.index.year <= 2023)].copy()
            val_data_with_errors, val_error_report = inject_errors_into_dataset(
                val_df, validation_error_generator, f"{args.station_id}_val", water_level_cols
            )
            val_data = val_data_with_errors['vst_raw']

    # --- Run MAD outlier detection and plot/save as before ---
    train_flags, val_flags, medians, mads, val_medians, val_mads = mad_outlier_flags(
        train_data, val_data, threshold=args.threshold, window_size=args.window_size)
    print(f"\nMAD Outlier Detection for station {args.station_id}")
    print(f"Threshold: {args.threshold}")
    print(f"Window size: {args.window_size} (steps)")
    print(f"Training set: {np.sum(train_flags)} anomalies out of {len(train_flags)}")
    print(f"Validation set: {np.sum(val_flags)} anomalies out of {len(val_flags)}")

    # Save path for training plot
    save_dir = Path("/Users/emmamork/Desktop/Master Thesis/MasterThesis/Project_Code - CORRECT/results/anomaly_detection")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"0_MAD_detection_{args.window_size}_{args.threshold}.png"
    plot_mad_anomalies(train_data, train_flags, medians, mads, args.threshold, args.window_size, title=f"Training Data: Water Level with MAD Anomalies. Anomalies detected {np.sum(train_flags)}", save_path=save_path)
    # Save HTML version
    html_save_path = save_dir / f"0_MAD_detection_{args.window_size}_{args.threshold}.html"
    plot_mad_anomalies_html(train_data, train_flags, medians, mads, args.threshold, args.window_size, title=f"Training Data: Water Level with MAD Anomalies. Anomalies detected {np.sum(train_flags)}", save_path=html_save_path)

    # Create zoom plots for validation data
    zoom_save_dir = Path("/Users/emmamork/Desktop/Master Thesis/MasterThesis/Project_Code - CORRECT/results/anomaly_detection/Zoom")
    create_anomaly_zoom_plots(
        val_data, val_flags, val_medians, val_mads, 
        threshold=args.threshold, 
        window_size=args.window_size,
        save_dir=zoom_save_dir
    )
    
    # Continue with existing plotting code... 