import sys
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_utils.data_loading import load_all_station_data


def preprocess_data():
    All_station_data = load_all_station_data()
    
    # Process each station's data
    for station_name, station_data in All_station_data.items():
        if station_data['vst_raw'] is not None:
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
            
            print(f"Processed {station_name}:")
            print(f"  - IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
            print(f"  - Removed {n_spikes} spikes")
            print(f"  - Removed {int(n_flatlines)} flatline points")
            
    
    return All_station_data

# preprocess_data()
# All_station_data = load_all_station_data()



# # Access temperature data for a specific station
# station_name = next(iter(All_station_data.keys()))  # gets first station name
# temp_data = All_station_data[station_name]['temperature']
# plt.figure(figsize=(12, 6))
# plt.plot(temp_data.index, temp_data['temperature (C)'])
# plt.title('Temperature Data - All Stations')
# plt.xlabel('Date')
# plt.ylabel('Temperature (°C)')
# plt.grid(True)
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


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

## Example usage:
if __name__ == "__main__":
    processed_data = preprocess_data()
    freezing_periods = find_freezing_periods(processed_data)
    


### CODE FOR BAR PLOT FOR COVER ###
# def plot_station_data(station_data, station_name):
#     """
#     Create an elegant, minimalist plot showing VST raw and edited data.
    
#     Args:
#         station_data (dict): Dictionary containing the station's data
#         station_name (str): Name of the station for the plot title
#     """
#     # Create figure with transparent background
#     fig = plt.figure(figsize=(15, 8), facecolor='none')
#     ax = plt.gca()
#     ax.set_facecolor('none')
    
#     # Plot VST edited data first (sophisticated burgundy)
#     if station_data['vst_edt'] is not None:
#         # Resample data to daily mean
#         vst_edt_resampled = station_data['vst_edt'].resample('D', on='Date').mean()
#         plt.bar(vst_edt_resampled.index, 
#                vst_edt_resampled['Value'],
#                color='#722F37',  # Rich burgundy
#                alpha=0.85, 
#                label='VST EDT',
#                width=0.7)  # Thinner bars
    
#     # Plot VST raw data on top (deep navy)
#     if station_data['vst_raw'] is not None:
#         # Resample data to daily mean
#         vst_raw_resampled = station_data['vst_raw'].resample('D', on='Date').mean()
#         plt.bar(vst_raw_resampled.index, 
#                vst_raw_resampled['Value'],
#                color='#1B3F8B',  # Deep navy
#                alpha=0.85, 
#                label='VST Raw',
#                width=0.7)  # Thinner bars
    
#     # Remove grid
#     plt.grid(False)
    
#     # Minimal legend with custom styling
#    # plt.legend(frameon=False, loc='upper right')
    
#     # Set date range
#     plt.xlim(pd.Timestamp('2009-01-01'), pd.Timestamp('2020-01-31'))
    
#     # Make all spines (border lines) transparent
#     for spine in ax.spines.values():
#         spine.set_alpha(0)

#     plt.tight_layout()
#     plt.show()


# # Example usage:
# if __name__ == "__main__":
#     processed_data = preprocess_data()
#     #freezing_periods = find_freezing_periods(processed_data)
    
#     # Plot data for each station
#     for station_name, station_data in processed_data.items():
#         plot_station_data(station_data, station_name)

