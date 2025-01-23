import matplotlib.pyplot as plt
import pandas as pd

def plot_datasets_overview(vst_dfs, vinge_df):
    """Create overview plots for all datasets including VINGE data."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    axes = [ax1, ax2, ax3]
    
    # Plot VST datasets
    for (folder, df), ax in zip(vst_dfs.items(), axes):
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        ax.plot(df['Date'], df['Value'], label='Sensor data')
        ax.scatter(vinge_df['Date'], vinge_df['W.L [cm]'], 
                  color='red', s=20, alpha=0.5, label='Manual measurements')
        ax.set_title(f'Dataset {folder}')
        ax.set_ylabel('Water level (mm)')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    
    plt.tight_layout() 