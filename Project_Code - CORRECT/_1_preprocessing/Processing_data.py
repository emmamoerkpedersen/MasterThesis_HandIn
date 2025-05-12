import sys
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio
from plotly.offline import plot
import copy

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

def detect_frost_periods(temperature_data):
    """
    Detect frost periods based on temperature data.
    
    Args:
        temperature_data (pd.DataFrame): DataFrame containing temperature measurements
        
    Returns:
        list: List of tuples containing (start_time, end_time) for each frost period
    """
    frost_periods = []
    frost_sum = 0
    current_period_start = None
    current_period_end = None
    
    for idx in range(len(temperature_data)):
        current_time = temperature_data.index[idx]
        current_temp = temperature_data['temperature'].iloc[idx]
        
        if current_temp < 0:
            # Start or continue tracking frost period
            if current_period_start is None:
                current_period_start = current_time
            current_period_end = current_time
            frost_sum += current_temp
        else:
            # Temperature is above 0, check if we were tracking a frost period
            if current_period_start is not None:
                # Check against single threshold
                if frost_sum < -15:
                    # Add 24 hours to the end of the frost period
                    extended_end = current_period_end + pd.Timedelta(hours=24)
                    # Convert times to timezone-naive if they're not already
                    if current_period_start.tzinfo is not None:
                        current_period_start = current_period_start.tz_localize(None)
                    if extended_end.tzinfo is not None:
                        extended_end = extended_end.tz_localize(None)
                    frost_periods.append((current_period_start, extended_end))
                # Reset tracking regardless of whether threshold was met
                current_period_start = None
                current_period_end = None
                frost_sum = 0
    
    return frost_periods

def remove_spikes(vst_data):
    """
    Detect and remove spikes in VST data using IQR method.
    
    Args:
        vst_data (pd.DataFrame): DataFrame containing VST measurements
        
    Returns:
        tuple: (filtered_data, n_spikes, bounds)
            - filtered_data: DataFrame with spikes removed
            - n_spikes: number of spikes removed
            - bounds: tuple of (lower_bound, upper_bound)
    """
    # Calculate IQR
    Q1 = vst_data['vst_raw'].quantile(0.25)
    Q3 = vst_data['vst_raw'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 4 * IQR
    
    # Create a mask for valid (non-NaN) values within bounds
    is_spike = (vst_data['vst_raw'] < lower_bound) | (vst_data['vst_raw'] > upper_bound)
    is_not_spike_or_nan = ~is_spike | vst_data['vst_raw'].isna()

    # Count spikes (excluding NaNs)
    n_spikes = is_spike.sum()

    # Filter data (preserving NaNs)
    filtered_data = vst_data[is_not_spike_or_nan]
    
    return filtered_data, n_spikes, (lower_bound, upper_bound)



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

    for _, group in groups:
        if len(group) >= threshold:
            # Mark all but the first value for removal
            indices_to_remove = group.index[1:]
            keep_mask[indices_to_remove] = False
            n_flatline += len(indices_to_remove)

    filtered_data = vst_data[keep_mask]

    return filtered_data, n_flatline


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
            if subkey == 'vst_raw' or subkey == 'vst_edt':
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
    
    # Process each station's data
    for station_name, station_data in All_station_data.items():
        # Detect and remove spikes
        station_data['vst_raw'], n_spikes, (lower_bound, upper_bound) = remove_spikes(station_data['vst_raw'])
        # Detect and remove flatlines
        station_data['vst_raw'], n_flatlines = remove_flatlines(station_data['vst_raw'])
        
        # Detect freezing periods
        temp_data = station_data['temperature']
        frost_periods = detect_frost_periods(temp_data)
        # Count points before frost period removal
        points_before = len(station_data['vst_raw'])
        #Remove VST data during frost periods
        for start, end in frost_periods:
            station_data['vst_raw'] = station_data['vst_raw'][
                ~((station_data['vst_raw'].index >= start) & 
                    (station_data['vst_raw'].index <= end))
            ]
        
        # Create vst_raw_feature as a separate feature
        # This will be used as an input feature, independent of the target vst_raw
        station_data['vst_raw_feature'] = station_data['vst_raw'].copy()
        station_data['vst_raw_feature'].columns = ['vst_raw_feature']  # Rename the column
        # Fill any remaining NaN values with -1 for the feature
        station_data['vst_raw_feature'] = station_data['vst_raw_feature'].fillna(-1)
        

        # Count points removed during frost periods
        points_removed_frost = points_before - len(station_data['vst_raw'])
          # Resample temperature data if it exists
        if station_data['temperature'] is not None:
            station_data['temperature'] = station_data['temperature'].resample('15min').ffill().bfill()  # Hold mean temperature constant but divide by 4
          
        # Resample rainfall data 
        if station_data['rainfall'] is not None:
            station_data['rainfall'] = station_data['rainfall'].fillna(-1)



        print(f"\nProcessed {station_name}:")
        print(f"  - Total data points before processing: {len(All_station_data_original[station_name]['vst_raw'])}")
        print(f"  - Total data points after processing: {len(station_data['vst_raw'])}")
        print(f"  - Total data points removed: {n_spikes + n_flatlines +points_removed_frost}")
        print(f"Percentage of data points removed: {(n_spikes + n_flatlines +points_removed_frost) / len(All_station_data_original[station_name]['vst_raw']) * 100:.2f}%")

        print(f"  - Removed {points_removed_frost} data points from {len(frost_periods)} frost periods")
        print(f"  - IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  - Removed {n_spikes} spikes")
        print(f"  - Removed {int(n_flatlines)} flatline points")
      

    # Save the preprocessed data
    save_data_Dict(All_station_data, filename=save_path / 'preprocessed_data.pkl')
    #save_data_Dict(frost_periods, filename=save_path / 'frost_periods.pkl')
  
    return All_station_data, All_station_data_original

if __name__ == "__main__":
    processed_data, original_data  = preprocess_data()
    station_id = '21006846'
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=('Original VST Raw Data', 'Processed VST Raw Data with Frost Periods', 'Rainfall'),
                        vertical_spacing=0.1)

    # Add original VST raw data trace to top subplot
    fig.add_trace(
        go.Scatter(
            x=original_data[station_id]['vst_raw'].index,
            y=original_data[station_id]['vst_raw']['Value'],
            name='VST Raw Original',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    # Add processed VST raw data trace to bottom subplot
    fig.add_trace(
        go.Scatter(
            x=processed_data[station_id]['vst_raw'].index,
            y=processed_data[station_id]['vst_raw']['vst_raw'],
            name='VST Raw Processed',
            line=dict(color='green')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=processed_data[station_id]['rainfall'].index,
            y=processed_data[station_id]['rainfall']['rainfall'],
            name='Rainfall',
            line=dict(color='red')
        ), 
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=processed_data[station_id]['temperature'].index,
            y=processed_data[station_id]['temperature']['temperature'],
            name='Temperature',
            line=dict(color='purple')
        ), 
        row=4, col=1
    )


    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        hovermode='x unified',
        title_text="VST Raw Data"
    )

    # Link x-axes of all subplots
    fig.update_xaxes(matches='x')
    fig.update_xaxes(range=['2010-01-01', '2025-01-01'])

    # Update y-axis labels
    fig.update_yaxes(title_text="VST Value", row=1, col=1)
    fig.update_yaxes(title_text="VST Value", row=2, col=1)

    # Open the plot in browser
    plot(fig, filename='station_data_comparison.html')



