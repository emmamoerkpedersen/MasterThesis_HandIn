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
    
    for station_name in split_data['train'].keys():
        if 'vst_raw' in split_data['train'][station_name]:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot each split with different colors
            ax.plot(split_data['train'][station_name]['vst_raw'].index,
                   split_data['train'][station_name]['vst_raw']['Value'],
                   'b-', label='Training', alpha=0.7, linewidth=0.5)
            
            ax.plot(split_data['validation'][station_name]['vst_raw'].index,
                   split_data['validation'][station_name]['vst_raw']['Value'],
                   'g-', label='Validation', alpha=0.7, linewidth=0.5)
            
            ax.plot(split_data['test'][station_name]['vst_raw'].index,
                   split_data['test'][station_name]['vst_raw']['Value'],
                   'r-', label='Test', alpha=0.7, linewidth=0.5)
            
            # Add vertical lines to show split points
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