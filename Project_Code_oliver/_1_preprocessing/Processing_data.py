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


def preprocess_data():
    All_station_data_original = load_all_station_data()

    # Rename columns before processing
    All_station_data = rename_columns(All_station_data_original)
    
    # Process each station's data
    for station_name, station_data in All_station_data.items():
        if station_data['vst_raw'] is not None and station_data['temperature'] is not None:
            # Convert temperature index to datetime if it's not already
            if not isinstance(station_data['temperature'].index, pd.DatetimeIndex):
                station_data['temperature'].index = pd.to_datetime(station_data['temperature'].index)
            
            # Ensure VST data index is datetime if it's not already
            if not isinstance(station_data['vst_raw'].index, pd.DatetimeIndex):
                station_data['vst_raw'].index = pd.to_datetime(station_data['vst_raw'].index)
            
            # Ensure VST data dates are timezone-naive
            if station_data['vst_raw'].index.tz is not None:
                station_data['vst_raw'].index = station_data['vst_raw'].index.tz_localize(None)
            
            if station_data['temperature'].index.tz is not None:
                station_data['temperature'].index = station_data['temperature'].index.tz_localize(None)
            
            # Calculate IQR for this station
            Q1 = station_data['vst_raw']['vst_raw'].quantile(0.25)
            Q3 = station_data['vst_raw']['vst_raw'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Count spikes before removal
            n_spikes = len(station_data['vst_raw']) - len(
                station_data['vst_raw'][
                    (station_data['vst_raw']['vst_raw'] >= lower_bound) & 
                    (station_data['vst_raw']['vst_raw'] <= upper_bound)
                ]
            )
            
            # Remove spikes
            station_data['vst_raw'] = station_data['vst_raw'][
                (station_data['vst_raw']['vst_raw'] >= lower_bound) & 
                (station_data['vst_raw']['vst_raw'] <= upper_bound)
            ]
            
            # Detect and remove flatlines (20 or more identical consecutive values)
            rolling_count = station_data['vst_raw']['vst_raw'].rolling(window=20).apply(
                lambda x: len(x.unique()) == 1
            ).fillna(False).astype(bool)
            
            # Keep non-flatline points
            station_data['vst_raw'] = station_data['vst_raw'][~rolling_count]
            
            # Count removed flatline points
            n_flatlines = rolling_count.sum()
            
            # Detect freezing periods
            temp_data = station_data['temperature']
            below_zero = temp_data['temperature'] < 0
            
            # Calculate cumulative frost sum for each location
            frost_sum = 0
            frost_periods = []
            current_period_start = None
            current_period_end = None
            
            for idx in range(len(temp_data)):
                current_time = temp_data.index[idx]
                current_temp = temp_data['temperature'].iloc[idx]
                
                if current_temp < 0:
                    if current_period_start is None:
                        current_period_start = current_time
                    current_period_end = current_time
                    frost_sum += current_temp
                    
                    # If frost sum exceeds threshold, mark period for removal
                    if frost_sum < -10:
                        # Add 24 hours to the end of the frost period
                        extended_end = current_period_end + pd.Timedelta(hours=24)
                        # Convert times to timezone-naive if they're not already
                        if current_period_start.tzinfo is not None:
                            current_period_start = current_period_start.tz_localize(None)
                        if extended_end.tzinfo is not None:
                            extended_end = extended_end.tz_localize(None)
                        frost_periods.append((current_period_start, extended_end))
                        # Reset tracking
                        current_period_start = None
                        current_period_end = None
                        frost_sum = 0
                else:
                    # Reset tracking when temperature goes above 0
                    if current_period_start is not None and frost_sum < -15:
                        # Add 24 hours to the end of the frost period
                        extended_end = current_period_end + pd.Timedelta(hours=24)
                        # Convert times to timezone-naive if they're not already
                        if current_period_start.tzinfo is not None:
                            current_period_start = current_period_start.tz_localize(None)
                        if extended_end.tzinfo is not None:
                            extended_end = extended_end.tz_localize(None)
                        frost_periods.append((current_period_start, extended_end))
                    current_period_start = None
                    current_period_end = None
                    frost_sum = 0
            
            # Remove VST data during frost periods
            for start, end in frost_periods:
                station_data['vst_raw'] = station_data['vst_raw'][
                    ~((station_data['vst_raw'].index >= start) & 
                      (station_data['vst_raw'].index <= end))
                ]
            
            print(f"\nProcessed {station_name}:")
            print(f"  - Removed data from {len(frost_periods)} frost periods")
            print(f"  - IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
            print(f"  - Removed {n_spikes} spikes")
            print(f"  - Removed {int(n_flatlines)} flatline points")
            
            # Resample temperature data if it exists
            if station_data['temperature'] is not None:
                temperature = station_data['temperature']
                station_data['temperature'] = temperature.resample('15min').ffill() / 4  # Hold mean temperature constant but divide by 4
                print(f"  - Resampled temperature data to 15-minute intervals")
            
            # Resample rainfall data if it exists
            if station_data['rainfall'] is not None:
                # Ensure rainfall index is datetime
                if not isinstance(station_data['rainfall'].index, pd.DatetimeIndex):
                    station_data['rainfall'].index = pd.to_datetime(station_data['rainfall'].index)
                
                # Resample rainfall to 15-minute intervals
                rainfall = station_data['rainfall']
                station_data['rainfall'] = rainfall.resample('15min').ffill()  # Distribute accumulated rainfall
                print(f"  - Resampled rainfall data to 15-minute intervals")


    # Save the preprocessed data
    save_data_Dict(All_station_data, filename='preprocessed_data.pkl')
    save_data_Dict(All_station_data_original, filename='original_data.pkl')

    
    
    return All_station_data, All_station_data_original

if __name__ == "__main__":
    processed_data, original_data = preprocess_data()



# # For plotting the raw vs. the processed data + temperature
# processed_data = preprocess_data()
# original_data = load_all_station_data()

# # Create interactive plot using Plotly with subplots
# fig = make_subplots(rows=2, cols=1, 
#                     subplot_titles=('Temperature', 'VST Raw Data Comparison'),
#                     vertical_spacing=0.15)

# # Add temperature trace to top subplot
# fig.add_trace(
#     go.Scatter(
#         x=processed_data['21006847']['temperature'].index,
#         y=processed_data['21006847']['temperature']['temperature'],
#         name='Temperature',
#         line=dict(color='red')
#     ),
#     row=1, col=1
# )

# # Add raw data trace to bottom subplot
# fig.add_trace(
#     go.Scatter(
#         x=original_data['21006847']['rainfall'].index,
#         y=original_data['21006847']['rainfall']['rainfall'],
#         name='Raw Data',
#         line=dict(color='blue')
#     ),
#     row=2, col=1
# )

# # Add processed data trace to bottom subplot
# fig.add_trace(
#     go.Scatter(
#         x=processed_data['21006847']['rainfall'].index,
#         y=processed_data['21006847']['rainfall']['rainfall'],
#         name='Processed Data',
#         line=dict(color='green')
#     ),
#     row=2, col=1
# )

# # Update layout
# fig.update_layout(
#     height=800,  # Increase overall height to accommodate both plots
#     showlegend=True,
#     hovermode='x unified'
# )

# # Link x-axes of both subplots
# fig.update_xaxes(matches='x')

# # Update y-axis labels
# fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
# fig.update_yaxes(title_text="VST Value", row=2, col=1)

# # Open the plot in browser
# plot(fig, filename='vst47_comparison.html')