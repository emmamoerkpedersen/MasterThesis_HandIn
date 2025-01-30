import os
import matplotlib.pyplot as plt
import pandas as pd
from data_loading import load_all_folders
from plot_code.overview_plots import plot_vst_raw_overview, plot_datasets_overview
from plot_code.comparison_plots import plot_vst_vs_vinge_comparison, plot_vst_files_comparison
from plot_code.anomaly_plots import create_detailed_plot

def get_data_path():
    """Get the path to the data directory."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, 'Sample data')

def get_plot_path():
    """Get the path to the plots directory."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_dir = os.path.join(repo_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

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

def load_rainfall_data(data_dir, station_id):
    """Load rainfall data for a specific station.
    
    Args:
        data_dir (str): Base directory path
        station_id (int): Rain station ID
        
    Returns:
        pd.DataFrame: DataFrame with datetime and rainfall values, or None if file not found
    """
    rain_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    station_id_padded = f"{int(station_id):05d}"
    rain_file = os.path.join(rain_data_dir, f'RainData_{station_id_padded}.csv')
    
    if os.path.exists(rain_file):
        df = pd.read_csv(rain_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Keep the original column names - they'll be handled in the plotting function
        return df
    return None

def main():
    """Main function to orchestrate the data exploration."""
    folders = ['21006845', '21006846', '21006847']
    data_dir = get_data_path()
    plot_dir = get_plot_path()
    
    # Load rain station mapping
    rain_stations = pd.read_csv('data/closest_rain_stations.csv')
    
    # Load all data
    all_data = load_all_folders(data_dir, folders)
    
    # Create VST raw data plots with resampling
    plot_vst_raw_overview(all_data)
    
    # Create other plots
    for folder, data in all_data.items():
        print(folder)
        if data['vst_raw'] is not None:
            # Get corresponding rain station
            rain_station = rain_stations[rain_stations['Station_of_Interest'] == int(folder)]['Closest_Rain_Station'].iloc[0]
            print(rain_station)
            rain_data = load_rainfall_data(data_dir, rain_station)
            print(rain_data)
            
            # Create overview plot with rainfall data
            plot_datasets_overview(data, folder, rain_data)
            plt.savefig(os.path.join(plot_dir, f'datasets_overview_{folder}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create comparison plots
            if data['vinge'] is not None:
                plot_vst_vs_vinge_comparison(data, folder)
                plt.savefig(os.path.join(plot_dir, f'vst_vs_vinge_comparison_{folder}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Create VST files comparison
            if data['vst_edt'] is not None:
                plot_vst_files_comparison(data, folder)
                plt.savefig(os.path.join(plot_dir, f'vst_files_comparison_{folder}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Create detailed analysis plot
            time_windows = get_time_windows()
            create_detailed_plot(data, time_windows, folder)
            plt.savefig(os.path.join(plot_dir, f'detailed_analysis_{folder}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()