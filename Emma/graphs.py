import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import os
import webbrowser
from tempfile import NamedTemporaryFile
from plotly.subplots import make_subplots



raw_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006845/VST_RAW.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
editted_level_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Sample data/21006845/VST_EDT.txt', sep = ';', decimal = ',', skiprows = 3, names = ['Date', 'Value'], encoding = 'latin-1')
vinge_data = pd.read_excel('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Emma/Sample data/21006845/VINGE.xlsm', decimal = ',', header = 0)
precipitation_data = pd.read_csv('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Master Thesis/MasterThesis/Emma/Sample data/RainData_05135.csv', parse_dates=['datetime'])


# Convert Date column to datetime with specified format (DD-MM-YYYY HH:MM)
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d-%m-%Y %H:%M')
editted_level_data['Date'] = pd.to_datetime(editted_level_data['Date'], format='%d-%m-%Y %H:%M')
vinge_data['Date'] = pd.to_datetime(vinge_data['Date'], format='%d.%m.%Y %H:%M')

# Crop all dataframes to start from 2010-01-01
start_date = '2010-01-01'
raw_data = raw_data[raw_data['Date'] >= start_date]
editted_level_data = editted_level_data[editted_level_data['Date'] >= start_date]
vinge_data = vinge_data[vinge_data['Date'] >= start_date]
precipitation_data = precipitation_data[precipitation_data['datetime'] >= start_date]

# Multiply W.L [cm] by 100 to convert to mm
vinge_data['W.L [cm]'] = vinge_data['W.L [cm]']*10
# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.3, 0.7])

# Add precipitation trace on top subplot
fig.add_trace(
    go.Scatter(
        x=precipitation_data['datetime'],
        y=precipitation_data['precipitation (mm)'],
        name='Precipitation',
        opacity=0.7
    ),
    row=1, col=1
)

# Add water level traces on bottom subplot
fig.add_trace(
    go.Scatter(
        x=raw_data['Date'],
        y=raw_data['Value'],
        name='Raw Data',
        opacity=0.7
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=editted_level_data['Date'],
        y=editted_level_data['Value'],
        name='Edited Data',
        opacity=0.7
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=vinge_data['Date'],
        y=vinge_data['W.L [cm]'],
        name='Vinge Data',
        mode='markers',
        opacity=0.7
    ),
    row=2, col=1
)

# Find gaps larger than 15 minutes
time_diffs = raw_data['Date'].diff()
gaps = raw_data[time_diffs > pd.Timedelta(minutes=15)]

# Calculate average gap duration
gap_durations = time_diffs[time_diffs > pd.Timedelta(minutes=15)]
average_gap_duration = gap_durations.mean()
print(f"Number of gaps: {len(gaps)}")
print(f"Average gap duration: {average_gap_duration}")

# Add colored background rectangles for gaps
for idx, gap_row in gaps.iterrows():
    # Find the index position in raw_data
    raw_data_idx = raw_data.index.get_loc(idx)
    if raw_data_idx > 0:  # Only process if there's a previous row
        prev_time = raw_data.iloc[raw_data_idx-1]['Date']
        current_time = gap_row['Date']
        
        fig.add_vrect(
            x0=prev_time,
            x1=current_time,
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2, col=1
        )

# Update layout
fig.update_layout(
    title='Water Level and Precipitation Data',
    height=800,  # Increase overall height of the figure
    showlegend=True,
    hovermode='x unified'
)

# Update y-axes labels
fig.update_yaxes(title_text="Precipitation [mm]", row=1, col=1)
fig.update_yaxes(title_text="Water Level [m]", row=2, col=1)

# Create a temporary HTML file and open it in the browser
temp = NamedTemporaryFile(delete=False, suffix='.html')
fig.write_html(temp.name)
webbrowser.open('file://' + temp.name)

