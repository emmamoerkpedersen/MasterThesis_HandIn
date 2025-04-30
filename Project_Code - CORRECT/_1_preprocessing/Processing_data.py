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
                if frost_sum < -35:
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

def detect_spikes(vst_data):
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
    
    # Count spikes before removal
    n_spikes = len(vst_data) - len(
        vst_data[
            (vst_data['vst_raw'] >= lower_bound) & 
            (vst_data['vst_raw'] <= upper_bound)
        ]
    )
    
    # Remove spikes
    filtered_data = vst_data[
        (vst_data['vst_raw'] >= lower_bound) & 
        (vst_data['vst_raw'] <= upper_bound)
    ]
    
    return filtered_data, n_spikes, (lower_bound, upper_bound)

def detect_flatlines(vst_data, window=30):
    """
    Detect and remove flatlines in VST data.
    
    Args:
        vst_data (pd.DataFrame): DataFrame containing VST measurements
        window (int): Number of consecutive identical values to consider as flatline
        
    Returns:
        tuple: (filtered_data, n_flatlines)
            - filtered_data: DataFrame with flatlines removed
            - n_flatlines: number of flatline points removed
    """
    # Detect flatlines (identical consecutive values)
    rolling_count = vst_data['vst_raw'].rolling(window=window).apply(
        lambda x: len(x.unique()) == 1
    ).fillna(False).astype(bool)
    
    # Keep non-flatline points
    filtered_data = vst_data[~rolling_count]
    
    # Count removed flatline points
    n_flatlines = rolling_count.sum()
    
    return filtered_data, n_flatlines

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
            
            # Fill missing values using forward fill then backward fill
            # This ensures continuous data without gaps
            df = df.ffill().bfill()

            # Fill rainfall data with -1
            if subkey == 'rainfall':
                df = df.fillna(-1)

            aligned_data[key][subkey] = df

    return aligned_data

def distribute_hourly_rainfall(rainfall_df):
    """
    Distribute hourly cumulated rainfall values across previous 15-minute intervals.
    
    Args:
        rainfall_df: Pandas DataFrame with hourly rainfall data
    Returns:
        Pandas DataFrame with 15-minute distributed rainfall data
    """
    # Get the rainfall column name (should be 'rainfall')
    rainfall_col = rainfall_df.columns[0]
    
    # Convert to series for easier handling
    rainfall_series = rainfall_df[rainfall_col]
    
    # Resample to 15-minute intervals
    resampled = rainfall_series.resample('15min').asfreq()
    
    # For each non-NaN hourly value
    for timestamp in rainfall_series.dropna().index:
        hourly_value = rainfall_series.loc[timestamp]
        
        # Get the previous hour's timestamps (4 fifteen-minute intervals)
        prev_timestamps = pd.date_range(end=timestamp, periods=4, freq='15min')
        
        # Distribute the hourly value equally (divide by 4)
        distributed_value = hourly_value / 4
        
        # Assign the distributed value to each 15-minute interval
        for prev_ts in prev_timestamps:
            resampled.loc[prev_ts] = distributed_value
    
    # Fill remaining NaN with -1
    resampled = resampled.fillna(-1)
    
    # Convert back to DataFrame with the same column name
    return pd.DataFrame(resampled, columns=[rainfall_col])


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
    
    # Ensure all timestamps are timezone-naive before alignment
    for station_name, station_data in All_station_data.items():
        for key, data in station_data.items():
            if data is not None and not data.empty:
                # Convert to timezone-naive if needed
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
    
    # Align data to common time index
    All_station_data = align_data(All_station_data)
    
    # Process each station's data
    for station_name, station_data in All_station_data.items():
        # Process all datasets in the station_data dictionary
        for key, data in station_data.items():
            if data is not None:
                # Convert index to datetime if it's not already
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                
                # Ensure data is timezone aware (UTC)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                
                # Make data timezone-naive for further processing
                station_data[key].index = station_data[key].index.tz_localize(None)
        
        # Detect and remove spikes
        station_data['vst_raw'], n_spikes, (lower_bound, upper_bound) = detect_spikes(station_data['vst_raw'])
        # Detect and remove flatlines
        station_data['vst_raw'], n_flatlines = detect_flatlines(station_data['vst_raw'])
        # # Detect freezing periods
        # temp_data = station_data['temperature']
        # frost_periods = detect_frost_periods(temp_data)
        # # Count points before frost period removal
        # points_before = len(station_data['vst_raw'])
        # #Remove VST data during frost periods
        # for start, end in frost_periods:
        #     station_data['vst_raw'] = station_data['vst_raw'][
        #         ~((station_data['vst_raw'].index >= start) & 
        #             (station_data['vst_raw'].index <= end))
        #     ]
        # Count points removed during frost periods
        #points_removed_frost = points_before - len(station_data['vst_raw'])
        

        print(f"\nProcessed {station_name}:")
        print(f"  - Total data points before processing: {len(All_station_data_original[station_name]['vst_raw'])}")
        print(f"  - Total data points after processing: {len(station_data['vst_raw'])}")
        print(f"  - Total data points removed: {len(All_station_data_original[station_name]['vst_raw']) - len(station_data['vst_raw'])}")
        # print(f"  - Removed {points_removed_frost} data points from {len(frost_periods)} frost periods")

        print(f"  - IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  - Removed {n_spikes} spikes")
        print(f"  - Removed {int(n_flatlines)} flatline points")
        
        # Resample temperature data if it exists
        if station_data['temperature'] is not None:
            station_data['temperature'] = station_data['temperature'].resample('15min').ffill().bfill()  # Hold mean temperature constant but divide by 4
            print(f"  - Resampled temperature data to 15-minute intervals with ffill and bfill")

        # Resample rainfall data 
        if station_data['rainfall'] is not None:
            station_data['rainfall'] = station_data['rainfall'].fillna(-1)
            print(f"  - Filled rainfall data with -1")

    # Save the preprocessed data
    save_data_Dict(All_station_data, filename=save_path / 'preprocessed_data.pkl')
    #save_data_Dict(frost_periods, filename=save_path / 'frost_periods.pkl')
  
    return All_station_data, All_station_data_original

if __name__ == "__main__":
    processed_data, original_data  = preprocess_data()
    station_id = '21006846'
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1,
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

      # Debug: Print rainfall data info
    print("Rainfall data info:")
    print(processed_data[station_id]['rainfall'].info())
    print("\nFirst few rows of rainfall data:")
    print(processed_data[station_id]['rainfall'].head())

    fig.add_trace(
        go.Scatter(
            x=processed_data[station_id]['rainfall'].index,
            y=processed_data[station_id]['rainfall']['rainfall'],
            name='Rainfall',
            line=dict(color='red')
        ), 
        row=3, col=1
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



