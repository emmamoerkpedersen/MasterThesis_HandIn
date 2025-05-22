import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from plotly.offline import plot
import copy
from plotly_resampler import FigureResampler
import numpy as np
# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from data_utils.data_loading import load_all_station_data, save_data_Dict


def rename_columns(dictionary):
    for station_id, station_data in dictionary.items():
        for key, time_series in station_data.items():
            # Skip if time_series is None
            if time_series is None:
                continue
                
            # Find the column that is not 'Date'
            non_date_column = [col for col in time_series.keys() if col != 'Date'][0]
            
            # Create a new entry with the key name
            time_series[key] = time_series[non_date_column]
            
            # Delete the old column
            del time_series[non_date_column]
    
    return dictionary

def detect_frost_periods(temperature_data, vst_data):
    """
    Detect frost periods based on temperature data and remove corresponding data from vst_data.
    
    Args:
        temperature_data (pd.DataFrame): DataFrame with a datetime index and a 'temperature' column.
        vst_data (pd.DataFrame): DataFrame with a datetime index to be filtered during frost periods.
        
    Returns:
        tuple: (list of frost periods, vst_data with frost periods removed, number of data points removed)
    """

    # Drop NaNs in temperature column
    temperature_data = temperature_data.dropna(subset=['temperature'])

    frost_periods = []
    frost_sum = 0
    current_period_start = None
    current_period_end = None
    points_removed_frost = 0  # Initialize counter for removed points
    threshold = 150

    for idx in range(len(temperature_data)):
        current_time = temperature_data.index[idx]
        current_temp = temperature_data['temperature'].iloc[idx]

        if pd.isna(current_temp):
            continue  # Skip NaNs explicitly, though dropna already handled most

        if current_temp < 0:
            if current_period_start is None:
                current_period_start = current_time
            current_period_end = current_time
            frost_sum += current_temp
        else:
            if current_period_start is not None:
                if frost_sum < -threshold:
                    extended_end = current_period_end + pd.Timedelta(hours=24)
                    frost_periods.append((current_period_start, extended_end))
                current_period_start = None
                current_period_end = None
                frost_sum = 0

    # Handle case where data ends in a frost period
    if current_period_start is not None:
        if frost_sum < -threshold:
            extended_end = current_period_end + pd.Timedelta(hours=24)
            frost_periods.append((current_period_start, extended_end))

    # Merge overlapping frost periods
    merged_frost_periods = []
    for start, end in sorted(frost_periods):
        if not merged_frost_periods or merged_frost_periods[-1][1] < start:
            merged_frost_periods.append((start, end))
        else:
            merged_frost_periods[-1] = (merged_frost_periods[-1][0], max(merged_frost_periods[-1][1], end))

    # Remove data from vst_data during merged frost periods
    for start, end in merged_frost_periods:
        points_removed_frost += vst_data[(vst_data.index >= start) & (vst_data.index <= end)].shape[0]
        vst_data = vst_data[~((vst_data.index >= start) & (vst_data.index <= end))]

    return merged_frost_periods, vst_data, points_removed_frost

def remove_spikes(vst_data):
    """
    Detect and remove spikes in VST data using IQR method.
    
    Args:
        vst_data (pd.DataFrame): DataFrame containing VST measurements
        
    Returns:
        tuple: (filtered_data, n_spikes, bounds, avg_spike_intensity, avg_spike_duration, neg_pos_spike_ratio)
            - filtered_data: DataFrame with spikes removed
            - n_spikes: number of spikes removed
            - bounds: tuple of (lower_bound, upper_bound)
            - avg_spike_intensity: average intensity of spikes removed
            - avg_spike_duration: average duration of spikes removed
            - neg_pos_spike_ratio: ratio of negative to positive spikes
    """
    # Calculate IQR
    Q1 = vst_data['vst_raw'].quantile(0.25)
    Q3 = vst_data['vst_raw'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 4 * IQR
    
    if lower_bound < 0:
        lower_bound = 0
    # Create a mask for valid (non-NaN) values within bounds
    is_spike = (vst_data['vst_raw'] < lower_bound) | (vst_data['vst_raw'] > upper_bound)
    is_not_spike_or_nan = ~is_spike | vst_data['vst_raw'].isna()
    
    # Calculate average intensity of spikes using absolute values
    avg_spike_intensity = vst_data['vst_raw'][is_spike].abs().mean()
    
    # Calculate average duration of spikes
    spike_durations = vst_data['vst_raw'][is_spike].groupby((~is_spike).cumsum()).size()
    avg_spike_duration = spike_durations.mean() if not spike_durations.empty else 0
    
    # Calculate ratio of negative to positive spikes
    negative_spikes = vst_data['vst_raw'][is_spike & (vst_data['vst_raw'] < 0)].count()
    positive_spikes = vst_data['vst_raw'][is_spike & (vst_data['vst_raw'] > 0)].count()
    neg_pos_spike_ratio = negative_spikes / positive_spikes if positive_spikes > 0 else float('inf')
    
    # Count spikes (excluding NaNs)
    n_spikes = is_spike.sum()
    
    # Filter data (preserving NaNs)
    filtered_data = vst_data[is_not_spike_or_nan]
    
    return filtered_data, n_spikes, (lower_bound, upper_bound), avg_spike_intensity, avg_spike_duration, neg_pos_spike_ratio


def remove_flatlines(vst_data, threshold=30):
    """
    Detect and remove flatline sequences longer than the threshold, 
    keeping only the first value in each sequence.

    Args:
        vst_data (pd.DataFrame): DataFrame with a 'vst_raw' column.
        threshold (int): Minimum number of consecutive equal values to consider a flatline.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with flatline sequences removed (except first value).
            - int: Number of flatline points removed.
            - float: Average duration of flatline segments removed.
    """
    vst_series = vst_data['vst_raw']
    
    # Identify where values change
    value_change = vst_series != vst_series.shift()
    group_id = value_change.cumsum()
    
    # Group by sequences of repeated values
    groups = vst_series.groupby(group_id)
    
    # Build a mask: keep all rows initially
    keep_mask = pd.Series(True, index=vst_data.index)
    n_flatline = 0
    flatline_durations = []
    
    for _, group in groups:
        if len(group) >= threshold:
            # Mark all but the first value for removal
            indices_to_remove = group.index[1:]
            keep_mask[indices_to_remove] = False
            n_flatline += len(indices_to_remove)
            flatline_durations.append(len(group))
    
    filtered_data = vst_data[keep_mask]
    
    # Calculate average duration of flatline segments
    avg_flatline_duration = sum(flatline_durations) / len(flatline_durations) if flatline_durations else 0
    
    return filtered_data, n_flatline, avg_flatline_duration


def align_data(data):
    aligned_data = {}

    # Step 1: Find the global min and max timestamp
    all_timestamps = []
    for key in data:
        for subkey, df in data[key].items():
            # Convert all timestamps to timezone-naive before adding to the list
            if df is not None and not df.empty:
                # Make a copy of the index and convert to timezone-naive
                timestamps = df.index.copy()
                if timestamps.tz is not None:
                    timestamps = timestamps.tz_localize(None)
                all_timestamps.extend(timestamps)

    # Ensure we have timestamps to work with
    if not all_timestamps:
        print("Warning: No timestamps found in the data")
        return data
        
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)

    # Step 2: Create a common time index with 15-minute intervals
    common_index = pd.date_range(start=min_time, end=max_time, freq='15min')

    # Step 3: Align data to the common index
    for key in data:
        aligned_data[key] = {}
        for i, (subkey, df) in enumerate(data[key].items()):
            if df is None or df.empty:
                aligned_data[key][subkey] = None
                continue
                
            df = df.copy()

            # Ensure index is timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Round timestamps **only** for the vst_raw, vst_edt, vinge data
            if subkey == 'vst_raw' or subkey == 'vst_edt' or subkey == 'vinge':
                df.index = df.index.round('15min')

            # Remove duplicates after rounding (if any)
            df = df[~df.index.duplicated(keep='first')]

            # Reindex to match the common 15-minute intervals
            df = df.reindex(common_index)  # Default fill value is NaN

            aligned_data[key][subkey] = df

    return aligned_data

def preprocess_data():
    """
    Preprocess the data and save to pickle files.
    
    Returns:
        tuple: (preprocessed_data, original_data, frost_periods)
    """
    print("Loading raw station data...")
    All_station_data_original = load_all_station_data()

    # Save the original data before any processing
    # Use absolute path from project root
    save_path = Path(__file__).parent.parent / "data_utils" / "Sample data"
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving original data to {save_path}")
    save_data_Dict(All_station_data_original, filename=save_path / 'original_data.pkl')

    print("Processing data...")
    # Create a deep copy for processing to keep original data intact
    All_station_data = copy.deepcopy(All_station_data_original)
    
    # Rename columns before processing
    All_station_data = rename_columns(All_station_data)

    # Align data to common time index
    All_station_data = align_data(All_station_data)
    
    # Collect frost periods from all stations
    all_frost_periods = []
    
    # Process each station's data
    for station_name, station_data in All_station_data.items():
        # Detect and remove spikes
        station_data['vst_raw'], n_spikes, (lower_bound, upper_bound), avg_spike_intensity, avg_spike_duration, neg_pos_spike_ratio = remove_spikes(station_data['vst_raw'])
        # Detect and remove flatlines
        station_data['vst_raw'], n_flatlines, avg_flatline_duration = remove_flatlines(station_data['vst_raw'])

        # # # Detect freezing periods and remove from vst_raw
        # temp_data = station_data['temperature']
        # frost_periods, station_data['vst_raw'], points_removed_frost = detect_frost_periods(temp_data, station_data['vst_raw'])
        # # Add to the combined list
        # all_frost_periods.extend(frost_periods)
              
    
        # Create vst_raw_feature as a separate feature
        # This will be used as an input feature, independent of the target vst_raw
        station_data['vst_raw_feature'] = station_data['vst_raw'].copy()
        station_data['vst_raw_feature'].columns = ['vst_raw_feature']  # Rename the column
        # Fill any remaining NaN values with -1 for the feature
        station_data['vst_raw_feature'] = station_data['vst_raw_feature'].fillna(-1)
        
          # Resample temperature data if it exists
        if station_data['temperature'] is not None:
            station_data['temperature'] = station_data['temperature'].resample('15min').ffill().bfill()  # Hold mean temperature constant but divide by 4
          
        # Resample rainfall data 
        if station_data['rainfall'] is not None:
            station_data['rainfall'] = station_data['rainfall'].fillna(-1)

        

        print(f"  - Total data points before processing: {len(All_station_data_original[station_name]['vst_raw'])}")
        print(f"  - Total data points after processing: {len(station_data['vst_raw'])}")
        print(f"  - Total data points removed: {n_spikes + n_flatlines }")
        print(f"Percentage of data points removed: {(n_spikes + n_flatlines ) / len(All_station_data_original[station_name]['vst_raw']) * 100:.2f}%")

        #print(f"  - Removed {points_removed_frost} data points from {len(frost_periods)} frost periods")
        print(f"  - IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  - Removed {n_spikes} spikes")
        print(f"  - Removed {int(n_flatlines)} flatline points")
        # print(f"  - Average flatline duration: {avg_flatline_duration:.2f} points")
        # print(f"  - Average spike intensity: {avg_spike_intensity:.2f}")    
        # print(f"  - Average spike duration: {avg_spike_duration:.2f} points")    
        # print(f"  - Negative to positive spike ratio: {neg_pos_spike_ratio:.2f}")    
        

    # Save the preprocessed data
    save_data_Dict(All_station_data, filename=save_path / 'preprocessed_data.pkl')
    # Save the frost periods
    save_data_Dict(all_frost_periods, filename=save_path / 'frost_periods.pkl')
  
    return All_station_data, All_station_data_original, all_frost_periods

def create_interactive_station_plot(processed_data, original_data, station_id, frost_periods):
    """
    Create an interactive plot showing the processed vst_raw data with frost periods marked,
    a subplot for temperature data, and a subplot for the original data.
    
    Args:
        processed_data: Dictionary containing processed station data
        original_data: Dictionary containing original station data
        station_id: ID of the station to plot
        frost_periods: List of tuples containing (start_time, end_time) for each frost period
    """
    # Create subplots with shared x-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Processed VST Raw Data", "Temperature Data", "Original VST Raw Data"))
    
    # Convert to FigureResampler for better performance with large datasets
    fig = FigureResampler(fig)
    
    # Add processed VST raw data trace
    fig.add_trace(
        go.Scattergl(
            name='Processed VST Raw Data',
            line=dict(color='blue', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
        ),
        hf_x=processed_data[station_id]['vst_raw'].index,
        hf_y=processed_data[station_id]['vst_raw']['vst_raw'],
        row=1, col=1
    )
    
    # Add temperature data trace
    fig.add_trace(
        go.Scattergl(
            name='Temperature',
            line=dict(color='red', width=1),
            hovertemplate='Date: %{x}<br>Temperature: %{y:.2f} °C<extra></extra>'
        ),
        hf_x=processed_data[station_id]['temperature'].index,
        hf_y=processed_data[station_id]['temperature']['temperature'],
        row=2, col=1
    )
    
    # Add original VST raw data trace
    fig.add_trace(
        go.Scattergl(
            name='Original VST Raw Data',
            line=dict(color='green', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
        ),
        hf_x=original_data[station_id]['vst_raw'].index,
        hf_y=original_data[station_id]['vst_raw']['Value'],
        row=3, col=1
    )
    
    # Add shaded regions for frost periods
    for start, end in frost_periods:
        fig.add_shape(
            type="rect",
            x0=start, x1=end, y0=0, y1=1, xref="x", yref="paper",
            fillcolor="lightblue", opacity=0.5, layer="below", line_width=0
        )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1200,
        title_text=f"Station {station_id} - VST Raw Data, Temperature, and Original Data with Frost Periods",
        hovermode='x',
        yaxis_title="Water Level (mm)",
        xaxis_title="Date",
        plot_bgcolor='white',  # Set plot background to white
        paper_bgcolor='white',  # Set paper background to white
        xaxis=dict(showgrid=False),  # Remove x-axis grid lines
        yaxis=dict(showgrid=False)   # Remove y-axis grid lines
    )
    
    return fig

if __name__ == "__main__":
    processed_data, original_data, frost_periods = preprocess_data()
    station_id = '21006846'  # You can change this to any station ID
    
    # Create interactive plot showing processed and original vst_raw data with frost periods
    interactive_fig = create_interactive_station_plot(processed_data, original_data, station_id, frost_periods)
    
    # Show using Dash (interactive mode)
    interactive_fig.show_dash(mode='inline', port=8050)

