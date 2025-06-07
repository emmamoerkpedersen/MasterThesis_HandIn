import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from Processing_data import preprocess_data

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def create_data_distribution_grid():
    """
    Create comprehensive distribution plots for all data series across all stations.
    """
    
    # Load the preprocessed data
    print("Loading preprocessed data...")
    processed_data, original_data, frost_periods = preprocess_data()
    
    # Define data types to analyze
    data_types = ['vst_raw', 'vst_edt', 'vinge', 'rainfall', 'temperature', 'vst_raw_feature']
    stations = list(processed_data.keys())
    
    # Create a large figure with subplots for each data type and station
    fig = plt.figure(figsize=(20, 16))
    
    # Calculate grid dimensions: 6 data types (rows) x 3 stations (columns)
    n_rows = len(data_types)
    n_cols = len(stations)
    
    # Create custom color palette for different data types
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_types)))
    
    plot_index = 1
    
    # Store statistics for summary
    stats_summary = {}
    
    for row, data_type in enumerate(data_types):
        for col, station in enumerate(stations):
            ax = plt.subplot(n_rows, n_cols, plot_index)
            
            # Get the data for this station and data type
            data = processed_data[station][data_type]
            
            if data is not None and not data.empty:
                # Extract the actual values (assuming single column after rename_columns)
                if data_type in data.columns:
                    values = data[data_type].dropna()
                else:
                    # Fallback to first column if the expected column name doesn't exist
                    values = data.iloc[:, 0].dropna()
                
                if len(values) > 0:
                    # Calculate statistics
                    stats = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75)
                    }
                    
                    stats_summary[f"{station}_{data_type}"] = stats
                    
                    # Create histogram with appropriate bins
                    n_bins = min(50, max(10, len(values) // 100))
                    
                    # Handle different data ranges for optimal visualization
                    if data_type == 'temperature':
                        bins = np.linspace(-20, 40, n_bins)
                    elif data_type == 'rainfall':
                        # Rainfall often has many zeros, so use custom binning
                        bins = np.logspace(0.1, np.log10(values.max() + 1), n_bins) if values.max() > 0 else n_bins
                    else:
                        bins = n_bins
                    
                    # Create histogram
                    n, bins_edges, patches = ax.hist(values, bins=bins, alpha=0.7, 
                                                   color=colors[row], density=True, 
                                                   edgecolor='black', linewidth=0.5)
                    
                    # Add statistics text
                    ax.axvline(stats['mean'], color='red', linestyle='--', alpha=0.8, 
                              linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
                    ax.axvline(stats['q25'], color='orange', linestyle=':', alpha=0.8, 
                              linewidth=1.5, label=f'Q25: {stats["q25"]:.2f}')
                    ax.axvline(stats['q75'], color='orange', linestyle=':', alpha=0.8, 
                              linewidth=1.5, label=f'Q75: {stats["q75"]:.2f}')
                    
                    # Set appropriate units for y-label
                    units = {
                        'vst_raw': 'mm',
                        'vst_edt': 'mm', 
                        'vinge': 'cm',
                        'rainfall': 'mm',
                        'temperature': '°C',
                        'vst_raw_feature': 'mm'
                    }
                    
                    if col == 0:  # Only label leftmost column
                        ax.set_ylabel('Density', fontsize=9)
                    if row == len(data_types) - 1:  # Only label bottom row
                        ax.set_xlabel(f'{data_type.replace("_", " ").title()} ({units.get(data_type, "")})', 
                                    fontsize=9)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
                    
                    # Add sample size annotation
                    ax.text(0.02, 0.98, f'n = {stats["count"]:,}', 
                           transform=ax.transAxes, fontsize=8, 
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Adjust tick label size
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    
                else:
                    ax.text(0.5, 0.5, 'No data available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
            
            plot_index += 1
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_dir = Path(__file__).parent.parent / "data_utils" / "Sample data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'data_distributions_grid.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Distribution plot saved to: {output_path}")
    
    # Create summary statistics table
    create_statistics_summary(stats_summary, output_dir)
    
    return fig, stats_summary

def create_statistics_summary(stats_summary, output_dir):
    """
    Create a comprehensive statistics summary table.
    """
    
    # Convert stats to DataFrame
    stats_df = pd.DataFrame(stats_summary).T
    
    # Add station and data_type columns
    stats_df['station'] = [key.split('_')[0] for key in stats_df.index]
    stats_df['data_type'] = ['_'.join(key.split('_')[1:]) for key in stats_df.index]
    
    # Reorder columns
    cols = ['station', 'data_type', 'count', 'mean', 'std', 'min', 'q25', 'q75', 'max']
    stats_df = stats_df[cols]
    
    # Round numerical values
    numerical_cols = ['mean', 'std', 'min', 'q25', 'q75', 'max']
    stats_df[numerical_cols] = stats_df[numerical_cols].round(2)
    
    # Save to CSV
    csv_path = output_dir / 'data_statistics_summary.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"Statistics summary saved to: {csv_path}")
    
    # Print summary to console
    print("\n" + "="*100)
    print("DATA DISTRIBUTION STATISTICS SUMMARY")
    print("="*100)
    print(stats_df.to_string(index=False))
    print("="*100)
    
    return stats_df

def create_comparative_boxplots():
    """
    Create comparative box plots for each data type across all stations.
    """
    
    # Load the preprocessed data
    processed_data, _, _ = preprocess_data()
    
    data_types = ['vst_raw', 'vst_edt', 'vinge', 'rainfall', 'temperature']
    stations = list(processed_data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, data_type in enumerate(data_types):
        ax = axes[i]
        
        data_for_boxplot = []
        labels = []
        
        for station in stations:
            data = processed_data[station][data_type]
            if data is not None and not data.empty:
                if data_type in data.columns:
                    values = data[data_type].dropna()
                else:
                    values = data.iloc[:, 0].dropna()
                
                if len(values) > 0:
                    data_for_boxplot.append(values)
                    labels.append(f'Station {station}')
        
        if data_for_boxplot:
            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.grid(True, alpha=0.3)
            
            # Set appropriate units
            units = {
                'vst_raw': 'mm',
                'vst_edt': 'mm', 
                'vinge': 'cm',
                'rainfall': 'mm',
                'temperature': '°C'
            }
            ax.set_ylabel(f'Values ({units.get(data_type, "")})')
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
    
    # Remove the extra subplot
    if len(data_types) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    
    # Save the comparative plot
    output_dir = Path(__file__).parent.parent / "data_utils" / "Sample data"
    output_path = output_dir / 'comparative_boxplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparative boxplot saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    print("Creating data distribution analysis...")
    
    # Create main distribution grid
    main_fig, stats = create_data_distribution_grid()
    
    # Create comparative boxplots
    comparative_fig = create_comparative_boxplots()
    
    # Show plots
    plt.show()
    
    print("\nAnalysis complete!") 