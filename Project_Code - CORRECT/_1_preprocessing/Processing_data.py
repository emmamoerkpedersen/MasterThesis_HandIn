import sys
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio
from plotly.offline import plot

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_utils.data_loading import load_all_station_data


def preprocess_data():
    All_station_data = load_all_station_data()
    freezing_periods = {}  # Add this to store freezing periods
    
    # Process each station's data
    for station_name, station_data in All_station_data.items():
        if station_data['vst_raw'] is not None and station_data['temperature'] is not None:
            # Calculate IQR for this station
            Q1 = station_data['vst_raw']['Value'].quantile(0.25)
            Q3 = station_data['vst_raw']['Value'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Count spikes before removal
            n_spikes = len(station_data['vst_raw']) - len(
                station_data['vst_raw'][
                    (station_data['vst_raw']['Value'] >= lower_bound) & 
                    (station_data['vst_raw']['Value'] <= upper_bound)
                ]
            )
            
            # Remove spikes
            station_data['vst_raw'] = station_data['vst_raw'][
                (station_data['vst_raw']['Value'] >= lower_bound) & 
                (station_data['vst_raw']['Value'] <= upper_bound)
            ]
            
            # Detect and remove flatlines (20 or more identical consecutive values)
            rolling_count = station_data['vst_raw']['Value'].rolling(window=20).apply(
                lambda x: len(x.unique()) == 1
            ).fillna(False).astype(bool)
            
            # Keep non-flatline points
            station_data['vst_raw'] = station_data['vst_raw'][~rolling_count]
            
            # Count removed flatline points
            n_flatlines = rolling_count.sum()
            
            # Detect freezing periods
            temp_data = station_data['temperature']
            below_zero = temp_data['temperature (C)'] < 0
            
            # Calculate duration of freezing periods
            freezing_start = None
            freezing_duration = pd.Timedelta(hours=0)
            periods_to_remove = []
            
            for idx in range(len(temp_data)):
                current_time = temp_data.index[idx]
                
                if below_zero.iloc[idx]:
                    if freezing_start is None:
                        freezing_start = current_time
                    freezing_duration = current_time - freezing_start
                    
                    # If freezing duration reaches 36 hours, mark period for removal
                    if freezing_duration >= pd.Timedelta(hours=36):
                        # Find when temperature goes above 0 again
                        for future_idx in range(idx, len(temp_data)):
                            if not below_zero.iloc[future_idx]:
                                end_time = temp_data.index[future_idx] + pd.Timedelta(hours=24)
                                # Convert times to timezone-naive if they're not already
                                if current_time.tzinfo is not None:
                                    current_time = current_time.tz_localize(None)
                                if end_time.tzinfo is not None:
                                    end_time = end_time.tz_localize(None)
                                periods_to_remove.append((current_time, end_time))
                                break
                        # Reset tracking
                        freezing_start = None
                        freezing_duration = pd.Timedelta(hours=0)
                else:
                    # Reset tracking when temperature goes above 0
                    freezing_start = None
                    freezing_duration = pd.Timedelta(hours=0)
            
            # Ensure VST data dates are timezone-naive
            if station_data['vst_raw']['Date'].dt.tz is not None:
                station_data['vst_raw']['Date'] = station_data['vst_raw']['Date'].dt.tz_localize(None)
            
            # Remove VST data during freezing periods
            for start, end in periods_to_remove:
                station_data['vst_raw'] = station_data['vst_raw'][
                    ~((station_data['vst_raw']['Date'] >= start) & 
                      (station_data['vst_raw']['Date'] <= end))
                ]
            
            # Store freezing periods for this station
            freezing_periods[station_name] = periods_to_remove
            
            print(f"\nProcessed {station_name}:")
            print(f"  - Removed data from {len(periods_to_remove)} freezing periods")
            print(f"  - IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
            print(f"  - Removed {n_spikes} spikes")
            print(f"  - Removed {int(n_flatlines)} flatline points")
    
    return All_station_data, freezing_periods  # Return both

### CODE FOR FREEZING PERIODS ###
def find_freezing_periods(preprocessed_data):
    """
    Identify periods where temperature remains below 0°C for 48 hours or more.
    
    Args:
        preprocessed_data (dict): Dictionary containing station data
        
    Returns:
        dict: Dictionary with station names as keys and list of freezing periods as values
    """
    freezing_periods = {}
    
    for station_name, station_data in preprocessed_data.items():
        if 'temperature' not in station_data or station_data['temperature'] is None:
            continue
            
        temp_data = station_data['temperature']
        
        # Create a boolean mask for temperatures below 0
        below_zero = temp_data['temperature (C)'] < 0
        
        # Calculate the duration of each temperature reading (in hours)
        time_diff = pd.Series(temp_data.index).diff().dt.total_seconds() / 3600
        
        # Initialize variables for tracking freezing periods
        current_start = None
        current_duration = 0
        station_periods = []
        
        for i in range(len(temp_data)):
            if below_zero.iloc[i]:
                if current_start is None:
                    current_start = temp_data.index[i]
                current_duration += time_diff.iloc[i] if i > 0 else 0
            else:
                if current_start is not None and current_duration >= 48:
                    station_periods.append({
                        'start': current_start,
                        'end': temp_data.index[i-1],
                        'duration_hours': current_duration
                    })
                current_start = None
                current_duration = 0
        
        # Check if we ended in a freezing period
        if current_start is not None and current_duration >= 48:
            station_periods.append({
                'start': current_start,
                'end': temp_data.index[-1],
                'duration_hours': current_duration
            })
        
        if station_periods:
            freezing_periods[station_name] = station_periods
            print(f"\nFound {len(station_periods)} freezing periods for {station_name}:")
            for period in station_periods:
                print(f"  - From {period['start']} to {period['end']} ({period['duration_hours']:.1f} hours)")
    
    return freezing_periods
