"""Diagnostic tools for data splitting step."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_split_visualization(split_data: dict, output_dir: Path):
    """
    Create visualization of how the data was split for each station.
    
    Args:
        split_data: Dictionary containing the split data
        output_dir: Directory to save diagnostic plots
    """
    # Create diagnostics directory if it doesn't exist
    diagnostic_dir = output_dir / "diagnostics" / "split"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    if "windows" in split_data:
        # Group windows by station
        stations_grouped = {}
        for year, stations in split_data["windows"].items():
            for station_name, station_data in stations.items():
                stations_grouped.setdefault(station_name, []).append((int(year), station_data))
        
        # Create one plot per station showing all windows
        for station, windows in stations_grouped.items():
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Sort windows by year
            windows_sorted = sorted(windows, key=lambda x: x[0])
            
            # Plot all data first
            all_data = pd.concat([window_data['vst_raw'] for _, window_data in windows_sorted 
                                if 'vst_raw' in window_data and not window_data['vst_raw'].empty])
            if not all_data.empty:
                ax.plot(all_data.index, all_data['Value'],
                       'b-', label='Data', alpha=0.7, linewidth=0.5)
                
                # Add vertical lines for each window boundary
                for year, window_data in windows_sorted:
                    if 'window_start' in window_data and 'window_end' in window_data:
                        ax.axvline(x=window_data['window_start'], color='k', 
                                 linestyle='--', alpha=0.5)
                        ax.axvline(x=window_data['window_end'], color='grey', 
                                 linestyle='--', alpha=0.5)
                        # Add year label at the top of the plot
                        ax.text(window_data['window_start'], ax.get_ylim()[1], 
                               f' {year}', rotation=0, va='bottom')
                
                ax.set_title(f"{station} - All Windows")
                ax.set_xlabel("Date")
                ax.set_ylabel("Water Level (mm)")
                ax.grid(True)
                plt.tight_layout()
                plt.savefig(diagnostic_dir / f"{station}_yearly_splits.png", dpi=300)
                plt.close(fig)
    else:
        # Normal mode: use existing code
        for station_name in split_data['train'].keys():
            if 'vst_raw' in split_data['train'][station_name]:
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.plot(split_data['train'][station_name]['vst_raw'].index,
                        split_data['train'][station_name]['vst_raw']['Value'],
                        'b-', label='Training', alpha=0.7, linewidth=0.5)
                ax.plot(split_data['validation'][station_name]['vst_raw'].index,
                        split_data['validation'][station_name]['vst_raw']['Value'],
                        'g-', label='Validation', alpha=0.7, linewidth=0.5)
                ax.plot(split_data['test'][station_name]['vst_raw'].index,
                        split_data['test'][station_name]['vst_raw']['Value'],
                        'r-', label='Test', alpha=0.7, linewidth=0.5)
                val_start = split_data['validation'][station_name]['vst_raw'].index[0]
                test_start = split_data['test'][station_name]['vst_raw'].index[0]
                ax.axvline(x=val_start, color='k', linestyle='--', alpha=0.5)
                ax.axvline(x=test_start, color='k', linestyle='--', alpha=0.5)
                ax.set_title(f'Data Split Visualization - {station_name}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Water Level (mm)')
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.savefig(diagnostic_dir / f"{station_name}_split.png", dpi=300)
                plt.close()

def generate_split_report(split_data: dict, output_dir: Path):
    """
    Generate a detailed report of the data splits.
    
    Args:
        split_data: Dictionary containing the split data
        output_dir: Directory to save diagnostic reports
    """
    report_dir = output_dir / "diagnostics" / "split"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "split_report.txt", "w") as f:
        f.write("Data Split Diagnostics Report\n")
        f.write("===========================\n\n")
        
        if "windows" in split_data:
            f.write("Yearly Split Diagnostics Report\n")
            f.write("===============================\n\n")
            # Group by station and only report a couple windows per station
            stations_grouped = {}
            for year, stations in split_data["windows"].items():
                for station_name, station_data in stations.items():
                    stations_grouped.setdefault(station_name, []).append((year, station_data))

            max_plots = 2  # maximum plots per station (e.g., earliest and latest window)
            for station, windows in stations_grouped.items():
                windows_sorted = sorted(windows, key=lambda x: int(x[0]))
                if len(windows_sorted) > max_plots:
                    windows_to_report = [windows_sorted[0], windows_sorted[-1]]
                else:
                    windows_to_report = windows_sorted

                for year, station_data in windows_to_report:
                    f.write(f"\nYear: {year}\n")
                    f.write("-" * (len(year) + 9) + "\n")
                    f.write(f"\nStation: {station}\n")
                    f.write("-" * (len(station) + 9) + "\n")
                    for data_type, data in station_data.items():
                        if data_type in ['window_start', 'window_end']:
                            continue
                        if data is not None and not data.empty:
                            f.write(f"  {data_type}:\n")
                            f.write(f"    - Points: {len(data)}\n")
                            f.write(f"    - Time range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}\n")
                            if data_type == 'vst_raw':
                                f.write(f"    - Value range: {data['Value'].min():.2f} to {data['Value'].max():.2f}\n")
                    if 'window_start' in station_data and 'window_end' in station_data:
                        f.write(f"  Window: {station_data['window_start']} to {station_data['window_end']}\n")
                    f.write("\n" + "="*50 + "\n")
        else:
            for station_name in split_data['train'].keys():
                f.write(f"\nStation: {station_name}\n")
                f.write("-" * (len(station_name) + 9) + "\n")
                for split_name in ['train', 'validation', 'test']:
                    station_split = split_data[split_name][station_name]
                    f.write(f"\n{split_name.capitalize()} Set:\n")
                    for data_type, data in station_split.items():
                        if data is not None and not data.empty:
                            f.write(f"  {data_type}:\n")
                            f.write(f"    - Points: {len(data)}\n")
                            f.write(f"    - Time range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}\n")
                            if data_type == 'vst_raw':
                                f.write(f"    - Value range: {data['Value'].min():.2f} to {data['Value'].max():.2f}\n")
                f.write("\n" + "="*50 + "\n") 