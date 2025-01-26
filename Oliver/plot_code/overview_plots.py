import matplotlib.pyplot as plt
import pandas as pd

def plot_datasets_overview(data, folder):
    """Create overview plot for a single dataset including VST and VINGE data."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot VST_RAW data
    if data['vst_raw'] is not None:
        ax.plot(data['vst_raw']['Date'], data['vst_raw']['Value'], 
                label='Sensor data (VST_RAW)', alpha=0.7)
    
    # Plot VST_EDT data if available
    if data['vst_edt'] is not None:
        ax.plot(data['vst_edt']['Date'], data['vst_edt']['Value'],
                label='Edited sensor data (VST_EDT)', alpha=0.7)
    
    # Plot VINGE data if available
    if data['vinge'] is not None:
        ax.scatter(data['vinge']['Date'], data['vinge']['W.L [cm]'],
                  color='red', s=20, alpha=0.5, label='Manual measurements')
    
    ax.set_title(f'Dataset Overview - {folder}')
    ax.set_ylabel('Water level (mm)')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    plt.tight_layout() 