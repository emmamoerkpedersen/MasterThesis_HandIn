from pathlib import Path
import pandas as pd
from data_loading import (
    load_all_folders, 
    load_rainfall_data, 
    get_data_path, 
    get_plot_path
)
from plotting import (
    plot_data_overview,
    plot_vst_raw_overview,
    plot_vst_vs_vinge_comparison,
    plot_vst_files_comparison,
    create_detailed_plot,
    plot_all_errors
)
from analysis import ErrorAnalyzer

def analyze_data_availability(df):
    """Analyze and print data availability information."""
    print("\nData Availability Analysis:")
    print(f"Dataset starts at: {df['Date'].min()}")
    print(f"Dataset ends at: {df['Date'].max()}")
    
    df_sorted = df.sort_values('Date')
    time_diff = df_sorted['Date'].diff()
    gaps = time_diff[time_diff > pd.Timedelta(days=7)]
    
    print("\nMajor gaps in data:")
    for idx in gaps.index:
        gap_start = df_sorted['Date'][idx - 1]
        gap_end = df_sorted['Date'][idx]
        print(f"Gap from {gap_start} to {gap_end} ({(gap_end - gap_start).days} days)")

def get_time_windows():
    """Define time windows for anomaly analysis."""
    return [
        {
            "title": "Data gaps",
            "start_date": pd.to_datetime('1994-01-01'),
            "end_date": pd.to_datetime('1995-01-01'),
            "y_range": None
        },
        {
            "title": "Linear Interpolation segment",
            "start_date": pd.to_datetime('1998-01-01'),
            "end_date": pd.to_datetime('2002-03-01'),
            "y_range": None
        },
        {
            "title": "Offset error",
            "start_date": pd.to_datetime('2007-03-19'),
            "end_date": pd.to_datetime('2007-03-30'),
            "y_range": (-7810, -7770)
        },
        {
            "title": "Spike error",
            "start_date": pd.to_datetime('2011-01-01'),
            "end_date": pd.to_datetime('2011-05-01'),
            "y_range": None
        },
        {
            "title": "Long offset error",
            "start_date": pd.to_datetime('2016-08-16'),
            "end_date": pd.to_datetime('2016-09-02'),
            "y_range": (22, 26)
        },
        {
            "title": "Spike fluctuations & flatline",
            "start_date": pd.to_datetime('2016-12-11'),
            "end_date": pd.to_datetime('2016-12-23'),
            "y_range": None
        }
    ]

def main():
    """Main function to orchestrate the data exploration."""
    folders = ['21006845', '21006846', '21006847']
    data_dir = get_data_path()
    plot_dir = get_plot_path()
    
    # Load rain station mapping
    rain_stations = pd.read_csv('data/closest_rain_stations.csv')
    
    # Load all data
    all_data = load_all_folders(data_dir, folders)
    
    # Create plots
    for folder, data in all_data.items():
        if data['vst_raw'] is not None:
            # Get corresponding rain station
            rain_station = rain_stations[
                rain_stations['Station_of_Interest'] == int(folder)
            ]['Closest_Rain_Station'].iloc[0]
            rain_data = load_rainfall_data(data_dir, rain_station)
            """
            # Create overview plots
            plot_data_overview(
                data['vst_raw'], 
                data['vst_edt'], 
                rain_data, 
                data['vinge']
            )
            """
            
         #   if data['vinge'] is not None:
         #       plot_vst_vs_vinge_comparison(data, folder)
            
         #   if data['vst_edt'] is not None:
         #       plot_vst_files_comparison(data, folder)
            
            # Create detailed analysis plots
           # time_windows = get_time_windows()
           # create_detailed_plot(data, time_windows, folder)
            
            # Create error analysis plot
            plot_all_errors(data, folder)

if __name__ == "__main__":
    main()