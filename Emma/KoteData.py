import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def load_station_data_level(station_id):
    """Load raw, edited and vinge data for a given station."""
    base_path = f'Sample data/{station_id}'
    
    # Load raw data
    raw_data_level = pd.read_csv(f'{base_path}/VST_RAW_LEVEL.txt', 
                          sep=';', decimal=',', skiprows=3, 
                          names=['Date', 'Value'], encoding='latin-1')
    
    # Load edited data
    edited_data_level = pd.read_csv(f'{base_path}/VST_EDT_LEVEL.txt', 
                            sep=';', decimal=',', skiprows=3, 
                            names=['Date', 'Value'], encoding='latin-1')
    
    # Load vinge data
    vinge_data_level = pd.read_excel(f'{base_path}/VINGE_level.xlsm', decimal=',', header=0)
    
    # Process data
    raw_data_level['Value'] = raw_data_level['Value'].astype(float)
    raw_data_level['Date'] = pd.to_datetime(raw_data_level['Date'], format='%d-%m-%Y %H:%M')
    edited_data_level['Date'] = pd.to_datetime(edited_data_level['Date'], format='%d-%m-%Y %H:%M')
    vinge_data_level['Date'] = pd.to_datetime(vinge_data_level['Date'], format='%d.%m.%Y %H:%M')
    
    # Convert Vinge water level to mm
    vinge_data_level['W.L [cm]'] = vinge_data_level['W.L [cm]'] * 10
    
    return raw_data_level, edited_data_level, vinge_data_level


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



station_id = '21006846'



raw_data_level, edited_data_level, vinge_data_level = load_station_data_level(station_id)
raw_data, edited_data, vinge_data = load_station_data(station_id)

# Filter data from year 2000
start_date = pd.to_datetime('2000-01-01')
raw_data_level_filtered = raw_data_level[raw_data_level['Date'] >= start_date]
edited_data_level_filtered = edited_data_level[edited_data_level['Date'] >= start_date]
vinge_data_level_filtered = vinge_data_level[vinge_data_level['Date'] >= start_date]

# Create an interactive plot with all three datasets
fig = go.Figure()

# Add raw data
fig.add_trace(
    go.Scatter(x=raw_data_level_filtered['Date'], y=raw_data_level_filtered['Value'],
               name='Raw Data',
               mode='lines',
               line=dict(color='blue', width=1))
)

# Add edited data
fig.add_trace(
    go.Scatter(x=edited_data_level_filtered['Date'], y=edited_data_level_filtered['Value'],
               name='Edited Data',
               mode='lines',
               line=dict(color='red', width=1))
)

# Add vinge data
fig.add_trace(
    go.Scatter(x=vinge_data_level_filtered['Date'], y=vinge_data_level_filtered['W.L [cm]'],
               name='Vinge Data',
               mode='markers',
               marker=dict(color='green', size=6))
)

# Update layout
fig.update_layout(
    title='Water Level Measurements Comparison (From 2000)',
    xaxis_title='Date',
    yaxis_title='Water Level (mm)',
    hovermode='x unified',
    showlegend=True
)

# Create a temporary HTML file and open it in the browser
temp_file = NamedTemporaryFile(delete=False, suffix='.html')
fig.write_html(temp_file.name)
webbrowser.open('file://' + temp_file.name)

difference = raw_data_level['Value']-raw_data['Value']


plt.plot(difference)