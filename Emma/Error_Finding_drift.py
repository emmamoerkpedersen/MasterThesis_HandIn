import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def load_station_data(station_id):
    """Load raw, edited and vinge data for a given station."""
    base_path = f'Sample data/{station_id}'
    
    # Load raw data
    raw_data = pd.read_csv(f'{base_path}/VST_RAW.txt', 
                          sep=';', decimal=',', skiprows=3, 
                          names=['Date', 'Value'], encoding='latin-1')
    
    # Load edited data
    edited_data = pd.read_csv(f'{base_path}/VST_EDT.txt', 
                            sep=';', decimal=',', skiprows=3, 
                            names=['Date', 'Value'], encoding='latin-1')
    
    # Load vinge data
    vinge_data = pd.read_excel(f'{base_path}/VINGE.xlsm', decimal=',', header=0)
    
    # Process data
    raw_data['Value'] = raw_data['Value'].astype(float)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d %H:%M:%S')
    edited_data['Date'] = pd.to_datetime(edited_data['Date'], format='%d-%m-%Y %H:%M')
    vinge_data['Date'] = pd.to_datetime(vinge_data['Date'], format='%d.%m.%Y %H:%M')
    
    # Convert Vinge water level to mm
    vinge_data['W.L [cm]'] = vinge_data['W.L [cm]'] * 10
    
    return raw_data, edited_data, vinge_data

def process_station_data(raw_data, vinge_data, start_date='2000-01-01', 
                        lower_threshold=5, upper_threshold=150):
    """Process and analyze data for drift detection."""
    # Crop data to start date
    raw_data = raw_data[raw_data['Date'] >= start_date]
    vinge_data = vinge_data[vinge_data['Date'] >= start_date]
    
    # Merge datasets
    merged_data = pd.merge_asof(
        vinge_data[['Date', 'W.L [cm]']].sort_values('Date'),
        raw_data.sort_values('Date'),
        on='Date',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=30)
    )
    
    # Calculate differences
    merged_data['difference'] = abs(merged_data['Value'].astype(float) - merged_data['W.L [cm]'])
    
    return analyze_drift(merged_data, lower_threshold, upper_threshold)

def analyze_drift(merged_data, lower_threshold, upper_threshold):
    """Analyze drift patterns in the data."""
    has_difference = (merged_data['difference'] > lower_threshold) & (merged_data['difference'] < upper_threshold)
    merged_data['drift_group'] = (~has_difference).cumsum()[has_difference]
    
    drift_stats = merged_data[has_difference].groupby('drift_group').agg({
        'Date': ['min', 'max', lambda x: (x.max() - x.min()).total_seconds() / (60 * 60 * 24)],
        'difference': ['mean', 'max', 'count']
    }).reset_index()
    
    drift_stats.columns = ['drift_group', 'start_date', 'end_date', 'duration_days', 
                          'mean_difference', 'max_difference', 'num_points']
    
    return merged_data, drift_stats

def create_drift_plot(raw_data, edited_data, vinge_data, drift_stats, station_id):
    """Create and save interactive plot."""
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=raw_data['Date'], y=raw_data['Value'],
                            name='Raw Data', line=dict(color='blue', width=1), opacity=0.7))
    
    fig.add_trace(go.Scatter(x=edited_data['Date'], y=edited_data['Value'],
                            name='Edited Data', line=dict(color='red', width=1), opacity=0.7))
    
    fig.add_trace(go.Scatter(x=vinge_data['Date'], y=vinge_data['W.L [cm]'],
                            name='Vinge Data', mode='markers',
                            marker=dict(color='green', size=5), opacity=0.7))
    
    # Add drift period highlights
    # for _, drift in drift_stats.iterrows():
    #     fig.add_vrect(x0=drift['start_date'], x1=drift['end_date'],
    #                  fillcolor="red", opacity=0.2, layer="below",
    #                  name="Drift Period", line_width=0)
    
    # Update layout
    fig.update_layout(
        title=f'Water Level Data Comparison - Station {station_id}',
        xaxis_title='Date',
        yaxis_title='Water Level (mm)',
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    # Save plot
    output_path = f'plots/drift_analysis_{station_id}.html'
    os.makedirs('plots', exist_ok=True)
    fig.write_html(output_path)
    return output_path

def create_cross_station_summary(station_ids):
    """Create a summary of drift statistics across all stations."""
    all_drift_stats = []
    
    for station_id in station_ids:
        drift_stats_path = f'Data Errors/drift_stats_{station_id}.csv'
        if os.path.exists(drift_stats_path):
            station_stats = pd.read_csv(drift_stats_path)
            # Filter out the summary row
            station_stats = station_stats[station_stats['drift_group'] != 'SUMMARY']
            station_stats['station_id'] = station_id
            all_drift_stats.append(station_stats)
    
    if not all_drift_stats:
        print("No drift statistics found for any station.")
        return
    
    combined_stats = pd.concat(all_drift_stats, ignore_index=True)
    
    cross_station_summary = pd.DataFrame({
        'metric': ['Cross-Station Summary'],
        'start_date': [combined_stats['start_date'].min()],
        'end_date': [combined_stats['end_date'].max()],
        'mean_duration_days': [combined_stats['duration_days'].mean()],
        'mean_difference': [combined_stats['mean_difference'].mean()],
        'median_difference': [combined_stats['mean_difference'].median()],
        'min_difference': [combined_stats['mean_difference'].min()],
        'max_difference': [combined_stats['max_difference'].max()],
        '25th_percentile': [combined_stats['mean_difference'].quantile(0.25)],
        '75th_percentile': [combined_stats['mean_difference'].quantile(0.75)],
        'avg_points_per_drift': [combined_stats['num_points'].mean()],
        'total_drift_periods': [len(combined_stats)],
        'total_points': [combined_stats['num_points'].sum()]
    })
    
    # Save cross-station summary
    os.makedirs('Data Errors', exist_ok=True)
    cross_station_summary.to_csv('Data Errors/cross_station_summary.csv', index=False)
    print("\nCross-station summary saved to 'Data Errors/cross_station_summary.csv'")
    
    return cross_station_summary

def print_station_statistics(drift_stats, station_id):
    """Print summary statistics for a single station."""
    print(f"\nStation {station_id} Statistics:")
    print("-" * 50)
    print(f"Date Range: {drift_stats['start_date'].min()} to {drift_stats['end_date'].max()}")
    print(f"Mean Duration of Drift Periods: {drift_stats['duration_days'].mean():.2f} days")
    print(f"Mean Difference: {drift_stats['mean_difference'].mean():.2f} mm")
    print(f"Median Difference: {drift_stats['mean_difference'].median():.2f} mm")
    print(f"Min Difference: {drift_stats['mean_difference'].min():.2f} mm")
    print(f"Max Difference: {drift_stats['max_difference'].max():.2f} mm")
    print(f"25th Percentile: {drift_stats['mean_difference'].quantile(0.25):.2f} mm")
    print(f"75th Percentile: {drift_stats['mean_difference'].quantile(0.75):.2f} mm")
    print(f"Average Points per Drift: {drift_stats['num_points'].mean():.2f}")
    print(f"Total Drift Periods: {len(drift_stats)}")
    print(f"Total Points: {drift_stats['num_points'].sum()}")
    print("-" * 50)

def main():
    """Main function to process all stations."""
    station_ids = ['21006845', '21006846', '21006847']
    
    for station_id in station_ids:
        print(f"\nProcessing station {station_id}")
        
        # Load data
        raw_data, edited_data, vinge_data = load_station_data(station_id)
        
        # Process data
        merged_data, drift_stats = process_station_data(raw_data, vinge_data)
        
        # Print station statistics
        print_station_statistics(drift_stats, station_id)
        
        # Create summary statistics
        summary_stats = pd.DataFrame({
            'drift_group': ['SUMMARY'],
            'start_date': [drift_stats['start_date'].min()],
            'end_date': [drift_stats['end_date'].max()],
            'duration_days': [drift_stats['duration_days'].mean()],
            'mean_difference': [drift_stats['mean_difference'].mean()],
            'median_difference': [drift_stats['mean_difference'].median()],
            'max_difference': [drift_stats['max_difference'].max()],
            'num_points': [drift_stats['num_points'].sum()]
        })
        
        # Add percentile statistics and save
        summary_stats['min_difference'] = drift_stats['mean_difference'].min()
        summary_stats['25th_percentile'] = drift_stats['mean_difference'].quantile(0.25)
        summary_stats['75th_percentile'] = drift_stats['mean_difference'].quantile(0.75)
        
        drift_stats_with_summary = pd.concat([drift_stats, summary_stats], ignore_index=True)
        drift_stats_with_summary.to_csv(f'Data Errors/drift_stats_{station_id}.csv', index=False)
        
        # Create and save plot
        plot_path = create_drift_plot(raw_data, edited_data, vinge_data, drift_stats, station_id)
        webbrowser.open('file://' + os.path.abspath(plot_path))
    
    # Create cross-station summary after processing all stations
    print("\nGenerating cross-station summary...")
    cross_station_summary = create_cross_station_summary(station_ids)
    if cross_station_summary is not None:
        print("\nCross-station summary statistics:")
        print("=" * 50)
        print(cross_station_summary.to_string(index=False))
        print("=" * 50)

if __name__ == "__main__":
    main()
