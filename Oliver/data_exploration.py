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
from analysis.error_analysis import ErrorAnalyzer  # Only import the ErrorAnalyzer class

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
    
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Use direct path for rain stations file
    rain_stations = pd.read_csv(Path('C:/Users/olive/OneDrive/GitHub/MasterThesis/data/closest_rain_stations.csv'))
    
    # Load all data
    all_data = load_all_folders(data_dir, folders)
    
    # Filter data from 2000 onwards (timezone-naive)
    start_date = pd.to_datetime('2000-01-01')
    for folder, data in all_data.items():
        if data['vst_raw'] is not None:
            data['vst_raw'] = data['vst_raw'][data['vst_raw']['Date'] >= start_date]
        if data['vst_edt'] is not None:
            data['vst_edt'] = data['vst_edt'][data['vst_edt']['Date'] >= start_date]
        if data['vinge'] is not None:
            data['vinge'] = data['vinge'][data['vinge']['Date'] >= start_date]
    
    # Collect statistics from all stations
    all_stats = []
    
    # Create plots and analyze data
    for folder, data in all_data.items():
        if data['vst_raw'] is not None:
            print(f"\nAnalyzing station {folder}:")
            
            # Create analyzer instance and run analysis with filtered data
            analyzer = ErrorAnalyzer(data['vst_raw'])            


            # Detect drift if VINGE data is available
            if data['vinge'] is not None:
                analyzer.drift_stats = analyzer.detect_drift(data['vinge'])
            
            # Save individual station statistics and collect for combined analysis
            analyzer.save_error_statistics(folder, results_dir)
            all_stats.append(analyzer.generate_error_statistics())

            # Get corresponding rain station and add to data dictionary
            rain_station = rain_stations[
                rain_stations['Station_of_Interest'] == int(folder)
            ]['Closest_Rain_Station'].iloc[0]
            rain_data = load_rainfall_data(data_dir, rain_station)
            if rain_data is not None:
                # Convert rain data to timezone-naive for consistency
                rain_data['datetime'] = rain_data['datetime'].dt.tz_localize(None)
                rain_data = rain_data[rain_data['datetime'] >= start_date]
            data['rain'] = rain_data
        
            # Add plot_data_overview call
            #plot_data_overview(
            #    df=data['vst_raw'],
            #    edt_df=data['vst_edt'],
            #    rain_df=data['rain'],
            #    vinge_df=data['vinge']
            #)

            
            
            # Create detailed analysis plots
            time_windows = get_time_windows()
            #create_detailed_plot(data, time_windows, folder)

            # Create error analysis plot with all data
            #plot_all_errors(data, folder)  # Default: include both rain and edt
            
            # Or without rainfall data
            #plot_all_errors(data, folder, include_rain=False)

            # Or without edited data
            #plot_all_errors(data, folder, include_edt=False)

            # Or without both
            #plot_all_errors(data, folder, include_rain=False, include_edt=False)
            
    # Generate combined statistics
    ErrorAnalyzer.save_combined_statistics(all_stats, folders, results_dir)

if __name__ == "__main__":
    main()
