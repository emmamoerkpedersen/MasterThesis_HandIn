import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile


raw_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006845/VST_RAW_level.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
editted_level_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006845/VST_EDT_level.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')

# Divide Value column by 1000
raw_data['Value'] = raw_data['Value'] / 1000
editted_level_data['Value'] = editted_level_data['Value'] / 1000


# Convert Date column to datetime with specified format (DD-MM-YYYY HH:MM)
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d-%m-%Y %H:%M')
editted_level_data['Date'] = pd.to_datetime(editted_level_data['Date'], format='%d-%m-%Y %H:%M')

# Create the interactive plot using Plotly
fig = go.Figure()

# Add traces for both raw and edited data
fig.add_trace(
    go.Scatter(
        x=raw_data['Date'],
        y=raw_data['Value'],
        name='Raw Data',
        opacity=0.7
    )
)

fig.add_trace(
    go.Scatter(
        x=editted_level_data['Date'],
        y=editted_level_data['Value'],
        name='Edited Data',
        opacity=0.7
    )
)

# Update layout
fig.update_layout(
    title='Comparison of Raw and Edited Level Data',
    xaxis_title='Date',
    yaxis_title='Value',
    hovermode='x unified',
    showlegend=True
)

# Create a temporary HTML file and open it in the browser
temp = NamedTemporaryFile(delete=False, suffix='.html')
fig.write_html(temp.name)
webbrowser.open('file://' + temp.name)


