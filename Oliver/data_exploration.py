import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
from plots.overview_plots import plot_datasets_overview
from plots.comparison_plots import plot_vst_vs_vinge_comparison, plot_vst_files_comparison
from plots.anomaly_plots import create_detailed_plot

def get_data_path():
    """Get the path to the data directory."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, 'Sample data')

def load_vst_data(folders, data_dir):
    """Load VST_RAW.txt files from multiple folders."""
    vst_dfs = {}
    
    for folder in folders:
        file_path = os.path.join(data_dir, folder, 'VST_RAW.txt')
        try:
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    # First try to read a few lines to check the format
                    with open(file_path, 'r', encoding=encoding) as f:
                        first_lines = [next(f) for _ in range(5)]
                        print(f"\nFirst few lines of {folder}:")
                        print(''.join(first_lines))
                    
                    df = pd.read_csv(file_path, 
                                   encoding=encoding,
                                   delimiter=';',
                                   decimal=',',
                                   skiprows=3,
                                   names=['Date', 'Value'])
                    vst_dfs[folder] = df
                    break
                except UnicodeDecodeError:
                    continue
                except StopIteration:
                    print(f"File {folder} has fewer than 5 lines")
                    break
            if folder not in vst_dfs:
                print(f"Warning: Could not read file in folder {folder} with any supported encoding")
        except FileNotFoundError:
            print(f"Warning: VST_RAW.txt not found in folder {folder}")
    
    return vst_dfs

def load_vinge_data(data_dir):
    """Load and process VINGE.txt data."""
    vinge_df = pd.read_csv(os.path.join(data_dir, '21006845', 'VINGE.txt'), 
                          delimiter='\t',
                          encoding='latin1',
                          decimal=',',
                          quotechar='"')
    
    vinge_df['Date'] = pd.to_datetime(vinge_df['Date'], format='%d.%m.%Y %H:%M')
    vinge_df['W.L [cm]'] = pd.to_numeric(vinge_df['W.L [cm]'], errors='coerce') * 10
    vinge_df = vinge_df[vinge_df['Date'].dt.year >= 1990]
    
    return vinge_df

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
    
    # Load data
    vst_dfs = load_vst_data(folders, data_dir)
    vinge_df = load_vinge_data(data_dir)
    
    # Create plots
    plot_datasets_overview(vst_dfs, vinge_df)
    plot_vst_vs_vinge_comparison(vst_dfs, vinge_df)
    
    # Create VST files comparison for each folder
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        plot_vst_files_comparison(folder_path)
    
    # Analyze first dataset in detail
    folder = list(vst_dfs.keys())[0]
    df = vst_dfs[folder]
    
    # Convert data types
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Analyze data availability
    analyze_data_availability(df)
    
    # Create detailed plot
    time_windows = get_time_windows()
    create_detailed_plot(df, vinge_df, time_windows, folder)
    
    plt.show()

if __name__ == "__main__":
    main()