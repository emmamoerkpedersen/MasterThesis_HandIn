import matplotlib.pyplot as plt
import pandas as pd
import os

def load_all_vst_files(folder_path):
    """Load all VST-related files from a folder."""
    files = {
        'VST_RAW': 'VST_RAW.txt',
        'VST_RAW_LEVEL': 'VST_RAW_LEVEL.txt',
        'VST_EDT': 'VST_EDT.txt',
        'VST_EDT_LEVEL': 'VST_EDT_LEVEL.txt'
    }
    
    dfs = {}
    for key, filename in files.items():
        file_path = os.path.join(folder_path, filename)
        try:
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, 
                                   encoding=encoding,
                                   delimiter=';',
                                   decimal=',',
                                   skiprows=3,
                                   names=['Date', 'Value'])
                    
                    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
                    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                    
                    dfs[key] = df
                    print(f"Loaded {filename}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    break
        except FileNotFoundError:
            print(f"File not found: {filename}")
    
    return dfs

def plot_vst_vs_vinge_comparison(vst_dfs, vinge_df):
    """Create comparison plots between VST and VINGE data for each folder."""
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
                closest_idx = ( manual_row['Date']-window_data['Date']).abs().idxmin()
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

    for folder, vst_df in vst_dfs.items():
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        fig.suptitle(f'Dataset Comparison - {folder}', fontsize=14)
        
        # Convert data types
        vst_df['Date'] = pd.to_datetime(vst_df['Date'], format='%d-%m-%Y %H:%M')
        vst_df['Value'] = pd.to_numeric(vst_df['Value'], errors='coerce')
        
        # Top plot: Raw data comparison
        ax1.plot(vst_df['Date'], vst_df['Value'], 'b-', label='Sensor data (VST_RAW)', alpha=0.7)
        ax1.scatter(vinge_df['Date'], vinge_df['W.L [cm]'], 
                   color='red', s=30, alpha=0.7, label='Manual measurements (VINGE)')
        ax1.set_ylabel('Water level (mm)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Calculate and plot differences
        diff_df = calculate_differences(vst_df, vinge_df)
        
        # Bottom plot: Differences
        scatter = ax2.scatter(diff_df['Date'], diff_df['Difference'],
                            c=diff_df['WaterLevel'], cmap='viridis',
                            s=30, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel('VST_RAW - VINGE (mm)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for water levels
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Water level (mm)')
        
        # Add statistics annotation
        stats_text = (f'Mean diff: {diff_df["Difference"].mean():.1f}mm\n'
                     f'Std dev: {diff_df["Difference"].std():.1f}mm\n'
                     f'Max abs diff: {diff_df["Difference"].abs().max():.1f}mm\n'
                     f'Measurements: {len(diff_df)}')
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format dates on x-axis
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()

def plot_vst_files_comparison(folder_path):
    """Create comparison plot for all VST-related files."""
    # Load all VST files
    dfs = load_all_vst_files(folder_path)
    
    if not dfs:
        print("No data files were loaded successfully.")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    fig.suptitle('Comparison of VST Files', fontsize=14)
    
    # Colors and styles for different files
    styles = {
        'VST_RAW': {'color': 'blue', 'linestyle': '-', 'alpha': 0.7, 'label': 'VST_RAW'},
        'VST_RAW_LEVEL': {'color': 'lightblue', 'linestyle': '--', 'alpha': 0.7, 'label': 'VST_RAW_LEVEL'},
        'VST_EDT': {'color': 'red', 'linestyle': '-', 'alpha': 0.7, 'label': 'VST_EDT'},
        'VST_EDT_LEVEL': {'color': 'lightcoral', 'linestyle': '--', 'alpha': 0.7, 'label': 'VST_EDT_LEVEL'}
    }
    
    # Plot all datasets in top subplot
    for name, df in dfs.items():
        style = styles[name]
        ax1.plot(df['Date'], df['Value'], 
                color=style['color'], 
                linestyle=style['linestyle'], 
                alpha=style['alpha'], 
                label=style['label'])
    
    ax1.set_ylabel('Water level (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate and plot differences in bottom subplot
    if 'VST_RAW' in dfs:
        base_df = dfs['VST_RAW']
        for name, df in dfs.items():
            if name != 'VST_RAW':
                # Merge datasets on Date
                merged = pd.merge(base_df, df, 
                                on='Date', 
                                suffixes=('_raw', f'_{name.lower()}'))
                
                # Calculate difference
                difference = merged['Value_raw'] - merged[f'Value_{name.lower()}']
                
                # Plot difference
                style = styles[name]
                ax2.plot(merged['Date'], difference,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        alpha=style['alpha'],
                        label=f'VST_RAW - {name}')
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Difference (mm)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format dates on x-axis
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    # Add statistics annotation to bottom subplot
    stats_text = []
    if 'VST_RAW' in dfs:
        base_df = dfs['VST_RAW']
        for name, df in dfs.items():
            if name != 'VST_RAW':
                merged = pd.merge(base_df, df, on='Date', suffixes=('_raw', f'_{name.lower()}'))
                difference = merged['Value_raw'] - merged[f'Value_{name.lower()}']
                stats_text.append(
                    f'{name}:\n'
                    f'  Mean diff: {difference.mean():.1f}mm\n'
                    f'  Std dev: {difference.std():.1f}mm\n'
                    f'  Max abs diff: {abs(difference).max():.1f}mm'
                )
    
    if stats_text:
        ax2.text(0.02, 0.98, '\n'.join(stats_text),
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
    
    plt.tight_layout() 