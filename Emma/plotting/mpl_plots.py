import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.dates as mdates

def calculate_differences(sensor_df, manual_df, max_time_diff='1H'):
    """Calculate differences between sensor and manual measurements."""
    differences = []
    dates = []
    water_levels = []
    
    for idx, manual_row in manual_df.iterrows():
        time_window = pd.Timedelta(max_time_diff)
        mask = (sensor_df['Date'] >= manual_row['Date'] - time_window) & \
               (sensor_df['Date'] <= manual_row['Date'] + time_window)
        
        if mask.any():
            window_data = sensor_df[mask]
            closest_idx = (manual_row['Date']-window_data['Date']).abs().idxmin()
            sensor_value = window_data.loc[closest_idx, 'Value']
            
            difference = sensor_value - manual_row['W.L [cm]']
            differences.append(difference)
            dates.append(manual_row['Date'])
            water_levels.append(manual_row['W.L [cm]'])
    
    return pd.DataFrame({
        'Date': dates,
        'Difference': differences,
        'WaterLevel': water_levels
    })

def plot_vst_vs_vinge_comparison(data, folder):
    """Create comparison plot between VST and VINGE data using Matplotlib."""
    if data['vst_raw'] is None or data['vinge'] is None:
        print(f"Missing required data for VST vs VINGE comparison in folder {folder}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    fig.suptitle(f'VST vs VINGE Comparison - {folder}', fontsize=14)
    
    # Convert dates to matplotlib format
    raw_dates = mdates.date2num(data['vst_raw']['Date'].values)
    vinge_dates = mdates.date2num(data['vinge']['Date'].values)
    
    # Top plot: Raw data comparison
    ax1.plot_date(raw_dates, data['vst_raw']['Value'],
                 'b-', label='Sensor data (VST_RAW)', alpha=0.7)
    ax1.scatter(vinge_dates, data['vinge']['W.L [cm]'],
                color='red', s=30, alpha=0.7, label='Manual measurements (VINGE)')
    ax1.set_ylabel('Water level (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate and plot differences
    diff_df = calculate_differences(data['vst_raw'], data['vinge'])
    diff_dates = mdates.date2num(diff_df['Date'].values)
    
    # Bottom plot: Differences
    scatter = ax2.scatter(diff_dates, diff_df['Difference'],
                         c=diff_df['WaterLevel'], cmap='viridis',
                         s=30, alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('VST_RAW - VINGE (mm)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Format dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Water level (mm)')
    
    # Add statistics
    stats_text = (f'Mean diff: {diff_df["Difference"].mean():.1f}mm\n'
                  f'Std dev: {diff_df["Difference"].std():.1f}mm\n'
                  f'Max abs diff: {diff_df["Difference"].abs().max():.1f}mm\n'
                  f'Measurements: {len(diff_df)}')
    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_dir = Path(r"C:\Users\olive\OneDrive\GitHub\MasterThesis\plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / f'vst_vs_vinge_comparison_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_vst_files_comparison(data, folder):
    """Create comparison plot between VST_RAW and VST_EDT files using Matplotlib."""
    if data['vst_raw'] is None:
        print(f"Missing VST_RAW data for folder {folder}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    fig.suptitle(f'VST Files Comparison - {folder}', fontsize=14)
    
    # Convert dates to matplotlib format
    raw_dates = mdates.date2num(data['vst_raw']['Date'].values)
    
    # Plot VST datasets in top subplot
    ax1.plot_date(raw_dates, data['vst_raw']['Value'],
                 '-', color='blue', alpha=0.7, label='VST_RAW')
    
    if data['vst_edt'] is not None:
        edt_dates = mdates.date2num(data['vst_edt']['Date'].values)
        ax1.plot_date(edt_dates, data['vst_edt']['Value'],
                     '-', color='red', alpha=0.7, label='VST_EDT')
    
    if data['vinge'] is not None:
        vinge_dates = mdates.date2num(data['vinge']['Date'].values)
        ax1.scatter(vinge_dates, data['vinge']['W.L [cm]'],
                   color='black', s=30, alpha=0.7, label='Manual measurements (VINGE)')
    
    ax1.set_ylabel('Water level (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate and plot differences if VST_EDT exists
    if data['vst_edt'] is not None:
        # Merge datasets on Date
        merged = pd.merge(data['vst_raw'], data['vst_edt'],
                         on='Date',
                         suffixes=('_raw', '_edt'))
        
        # Calculate difference
        difference = merged['Value_raw'] - merged['Value_edt']
        merged_dates = mdates.date2num(merged['Date'].values)
        
        # Plot difference
        ax2.plot_date(merged_dates, difference,
                     '-', color='purple', alpha=0.7,
                     label='VST_RAW - VST_EDT')
        
        # Add statistics
        stats_text = (f'Mean diff: {difference.mean():.1f}mm\n'
                     f'Std dev: {difference.std():.1f}mm\n'
                     f'Max abs diff: {abs(difference).max():.1f}mm')
        
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Difference (mm)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format dates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_dir = Path(r"C:\Users\olive\OneDrive\GitHub\MasterThesis\plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / f'vst_files_comparison_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close() 