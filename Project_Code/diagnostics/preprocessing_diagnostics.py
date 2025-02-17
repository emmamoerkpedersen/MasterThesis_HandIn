"""Diagnostic tools for data preprocessing step."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_preprocessing_comparison(original_data: dict, preprocessed_data: dict, output_dir: Path):
    """Create comparison plots between original and preprocessed data for each station."""
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name in original_data.keys():
        if (original_data[station_name]['vst_raw'] is not None and 
            preprocessed_data[station_name]['vst_raw'] is not None):
            
            # Create figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))
            
            orig = original_data[station_name]['vst_raw']
            proc = preprocessed_data[station_name]['vst_raw']
            
            # Calculate IQR bounds
            Q1 = orig['Value'].quantile(0.25)
            Q3 = orig['Value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 4 * IQR
            
            # Original data with bounds
            ax1.plot(orig['Date'], orig['Value'], 'b-', label='Original', alpha=0.7, linewidth=0.5)
            ax1.axhline(y=lower_bound, color='r', linestyle='--', label='Lower bound (Q1 - 1.5*IQR)')
            ax1.axhline(y=upper_bound, color='r', linestyle='--', label='Upper bound (Q3 + 4*IQR)')
            ax1.set_title(f'Original Data with IQR Bounds - {station_name}')
            ax1.set_ylabel('Water Level (mm)')
            ax1.grid(True)
            ax1.legend()
            
            # Preprocessed data
            ax2.plot(proc['Date'], proc['Value'], 'g-', label='Preprocessed', alpha=0.7, linewidth=0.5)
            ax2.set_title('Preprocessed Data')
            ax2.set_ylabel('Water Level (mm)')
            ax2.grid(True)
            ax2.legend()
            
            # Removed points highlighted
            ax3.plot(orig['Date'], orig['Value'], 'b-', label='Original', alpha=0.3, linewidth=0.5)
            removed_mask = (orig['Value'] < lower_bound) | (orig['Value'] > upper_bound)
            removed_points = orig[removed_mask]
            ax3.scatter(removed_points['Date'], removed_points['Value'], 
                       color='r', s=20, alpha=0.5, label='Removed Points')
            ax3.set_title('Removed Data Points')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Water Level (mm)')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            plt.savefig(diagnostic_dir / f"{station_name}_preprocessing.png", dpi=300)
            plt.close()

def generate_preprocessing_report(original_data: dict, preprocessed_data: dict, output_dir: Path):
    """Generate detailed report including all data types and IQR statistics."""
    report_dir = output_dir / "diagnostics" / "preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "preprocessing_report.txt", "w") as f:
        f.write("Preprocessing Diagnostics Report\n")
        f.write("==============================\n\n")
        
        for station_name, station_data in preprocessed_data.items():
            f.write(f"\nStation: {station_name}\n")
            f.write("-" * (len(station_name) + 9) + "\n")
            
            # VST_RAW details with IQR statistics
            if station_data['vst_raw'] is not None:
                vst = station_data['vst_raw']
                orig_vst = original_data[station_name]['vst_raw']
                
                # Calculate IQR statistics
                Q1 = orig_vst['Value'].quantile(0.25)
                Q3 = orig_vst['Value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 4 * IQR
                
                f.write("\nVST_RAW Data:\n")
                f.write(f"  - Measurements: {len(vst)}\n")
                f.write(f"  - Time range: {pd.to_datetime(vst.index.min())} to {pd.to_datetime(vst.index.max())}\n")
                f.write(f"  - Value range: {vst['Value'].min():.2f} to {vst['Value'].max():.2f}\n")
                
                f.write("\nIQR Statistics:\n")
                f.write(f"  - Q1: {Q1:.2f}\n")
                f.write(f"  - Q3: {Q3:.2f}\n")
                f.write(f"  - IQR: {IQR:.2f}\n")
                f.write(f"  - Lower bound: {lower_bound:.2f}\n")
                f.write(f"  - Upper bound: {upper_bound:.2f}\n")
                
                points_below = len(orig_vst[orig_vst['Value'] < lower_bound])
                points_above = len(orig_vst[orig_vst['Value'] > upper_bound])
                f.write("\nPoints removed by bounds:\n")
                f.write(f"  - Below lower bound: {points_below}\n")
                f.write(f"  - Above upper bound: {points_above}\n")
                f.write(f"  - Total points removed: {points_below + points_above}\n")
            
            # VINGE details
            if station_data['vinge'] is not None:
                vinge = station_data['vinge']
                vinge_dates = pd.to_datetime(vinge.index)
                f.write("\nVINGE Measurements:\n")
                f.write(f"  - Number of measurements: {len(vinge)}\n")
                f.write(f"  - Average measurements per year: {len(vinge)/len(vinge_dates.year.unique()):.1f}\n")
                f.write(f"  - Years covered: {vinge_dates.year.min()} to {vinge_dates.year.max()}\n")
            
            # Rainfall details
            if station_data['rainfall'] is not None:
                rain = station_data['rainfall']
                f.write("\nRainfall Data:\n")
                f.write(f"  - Measurements: {len(rain)}\n")
                f.write(f"  - Time period: {rain.index.min().strftime('%Y-%m-%d')} to {rain.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"  - Total rainfall: {rain['precipitation (mm)'].sum():.1f} mm\n")
                f.write(f"  - Years covered: {rain.index.year.min()} to {rain.index.year.max()}\n")
            
            # Temperature details
            if station_data['temperature'] is not None:
                temp = station_data['temperature']
                f.write("\nTemperature Data:\n")
                f.write(f"  - Measurements: {len(temp)}\n")
                f.write(f"  - Time period: {temp.index.min().strftime('%Y-%m-%d')} to {temp.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"  - Years covered: {temp.index.year.min()} to {temp.index.year.max()}\n")
                f.write(f"  - Range: {temp['temperature (C)'].min():.1f}°C to {temp['temperature (C)'].max():.1f}°C\n")
            
            f.write("\n" + "="*50 + "\n")

def plot_station_data_overview(original_data: dict, preprocessed_data: dict, output_dir: Path):
    """
    Create comprehensive visualization of all data types for each station.
    
    Args:
        original_data: Dictionary of original station data
        preprocessed_data: Dictionary of preprocessed station data
        output_dir: Directory to save diagnostic plots
    """
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name in preprocessed_data.keys():
        if preprocessed_data[station_name]['vst_raw'] is not None:
            # Create figure with 4 subplots
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20), height_ratios=[3, 1, 1, 1])
            
            # 1. Water level measurements (VST_RAW, VST_EDT, VINGE)
            data = preprocessed_data[station_name]
            ax1.plot(data['vst_raw'].index, data['vst_raw']['Value'],
                    'b-', label='VST Raw', alpha=0.7, linewidth=0.5)
            
            if data['vst_edt'] is not None:
                ax1.plot(data['vst_edt'].index, data['vst_edt']['Value'],
                        'g-', label='VST EDT', alpha=0.7, linewidth=0.5)
            
            if data['vinge'] is not None:
                ax1.plot(data['vinge'].index, data['vinge']['W.L [cm]'],
                        'ro', label='VINGE', markersize=4)
            
            ax1.set_title(f'Water Level Measurements - {station_name}')
            ax1.set_ylabel('Water Level (mm)')
            ax1.grid(True)
            ax1.legend()
            
            # 2. Rainfall data
            if data['rainfall'] is not None:
                ax2.plot(data['rainfall'].index, data['rainfall']['precipitation (mm)'],
                        'b-', label='Rainfall', linewidth=0.5)
                ax2.set_title('Rainfall Data')
                ax2.set_ylabel('Precipitation (mm)')
                ax2.grid(True)
                ax2.legend()
            
            # 3. Temperature data
            if data['temperature'] is not None:
                ax3.plot(data['temperature'].index, data['temperature']['temperature (C)'],
                        'r-', label='Temperature', linewidth=0.5)
                ax3.set_title('Temperature Data')
                ax3.set_ylabel('Temperature (°C)')
                ax3.grid(True)
                ax3.legend()
            
            # 4. VINGE measurements detail
            if data['vinge'] is not None:
                ax4.plot(data['vinge'].index, data['vinge']['W.L [cm]'],
                        'ro-', label='VINGE', markersize=4)
                ax4.set_title('Manual Board Measurements (VINGE)')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Water Level (mm)')
                ax4.grid(True)
                ax4.legend()
            
            plt.tight_layout()
            plt.savefig(diagnostic_dir / f"{station_name}_data_overview.png", dpi=300)
            plt.close()

def plot_additional_data(preprocessed_data: dict, output_dir: Path):
    """
    Create additional plots for VINGE, rainfall and temperature data.
    
    Args:
        preprocessed_data: Dictionary of preprocessed station data
        output_dir: Directory to save diagnostic plots
    """
    diagnostic_dir = output_dir / "diagnostics" / "preprocessing"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    for station_name, station_data in preprocessed_data.items():
        if station_data['vst_raw'] is not None:
            # VST and VINGE comparison
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Make sure we're using datetime for x-axis
            vst_data = station_data['vst_raw'].copy()
            if not isinstance(vst_data.index, pd.DatetimeIndex):
                vst_data.set_index('Date', inplace=True)
            
            ax.plot(vst_data.index, vst_data['Value'],
                   'b-', label='VST Raw', alpha=0.7, linewidth=0.5)
            
            if station_data['vinge'] is not None:
                vinge_data = station_data['vinge'].copy()
                if not isinstance(vinge_data.index, pd.DatetimeIndex):
                    vinge_data.set_index('Date', inplace=True)
                    
                ax.plot(vinge_data.index, vinge_data['W.L [cm]'],
                       'ro', label='VINGE', markersize=6)
            
            ax.set_title(f'VST Raw and VINGE Measurements - {station_name}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Level (mm)')
            ax.grid(True)
            ax.legend()
            
            # Format x-axis to show dates properly
            plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
            
            plt.tight_layout()
            plt.savefig(diagnostic_dir / f"{station_name}_vst_vinge.png", dpi=300)
            plt.close()
            
            # Rainfall data
            if station_data['rainfall'] is not None:
                fig, ax = plt.subplots(figsize=(15, 6))
                ax.plot(station_data['rainfall'].index, 
                       station_data['rainfall']['precipitation (mm)'],
                       'b-', label='Rainfall', linewidth=0.5)
                ax.set_title(f'Rainfall Data - {station_name}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Precipitation (mm)')
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.savefig(diagnostic_dir / f"{station_name}_rainfall.png", dpi=300)
                plt.close()
            
            # Temperature data
            if station_data['temperature'] is not None:
                fig, ax = plt.subplots(figsize=(15, 6))
                ax.plot(station_data['temperature'].index,
                       station_data['temperature']['temperature (C)'],
                       'r-', label='Temperature', linewidth=0.5)
                ax.set_title(f'Temperature Data - {station_name}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Temperature (°C)')
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.savefig(diagnostic_dir / f"{station_name}_temperature.png", dpi=300)
                plt.close() 