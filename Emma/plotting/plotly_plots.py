import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

def plot_data_overview(df, edt_df=None, rain_df=None, vinge_df=None, 
                      value_column='Value', timestamp_column='Date'):
    """Plot raw data, edited data, VINGE data, and rainfall using Plotly."""
    # Create figure with two rows, one column and shared x-axes
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.2, 0.8],  # Rainfall plot takes 20% of height
        vertical_spacing=0.02,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}]],
        shared_xaxes=True
    )
    
    # Create a FigureResampler object
    fig = FigureResampler(fig)
    
    # Add rainfall data in top subplot
    if rain_df is not None:
        daily_rain = rain_df.set_index('datetime')\
            .resample('D')['precipitation (mm)']\
            .sum()\
            .reset_index()
        
        fig.add_trace(
            go.Bar(
                name='Daily Rainfall',
                x=daily_rain['datetime'],
                y=daily_rain['precipitation (mm)'],
                marker_color='blue',
                opacity=0.3,
                width=24*60*60*1000,
                showlegend=True,
                hovertemplate='Date: %{x}<br>Daily Rainfall: %{y:.1f} mm<extra></extra>',
            ),
            row=1, col=1
        )
    
    # Add VINGE data (in background)
    if vinge_df is not None:
        fig.add_trace(
            go.Scattergl(
                name='VINGE Data',
                mode='markers',
                marker=dict(
                    color='orange',
                    size=4,
                    symbol='circle'
                ),
                showlegend=True,
                hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            ),
            hf_x=vinge_df['Date'],
            hf_y=vinge_df['W.L [cm]'],
            row=2, col=1
        )
    
    # Add edited data
    if edt_df is not None:
        fig.add_trace(
            go.Scattergl(
                name='Edited Data',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=True,
                hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            ),
            hf_x=edt_df[timestamp_column],
            hf_y=edt_df[value_column],
            row=2, col=1
        )
    
    # Add original data
    fig.add_trace(
        go.Scattergl(
            name='Original Data',
            showlegend=True,
            line=dict(color='black', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
        ),
        hf_x=df[timestamp_column],
        hf_y=df[value_column],
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1600,
        title_text="Water Level and Rainfall Data Overview",
        showlegend=True,
        hovermode='closest',
        margin=dict(t=100, b=50, l=50, r=50),
        font=dict(size=14),
        title_font=dict(size=24)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", title_font=dict(size=16), row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="Rainfall (mm)", title_font=dict(size=16), row=1, col=1)
    fig.update_yaxes(title_text="Water Level (mm)", title_font=dict(size=16), row=2, col=1)
    
    # Show the figure
    fig.show_dash(mode='inline')

def plot_vst_raw_overview(all_data):
    """Create overview plots showing raw VST data using plotly-resampler."""
    for folder, data in all_data.items():
        if data['vst_raw'] is None:
            continue
            
        fig = FigureResampler(go.Figure())
        
        fig.add_trace(
            go.Scattergl(
                name='VST_RAW',
                showlegend=True,
                line=dict(color='#1f77b4', width=1),
                hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
            ),
            hf_x=data['vst_raw']['Date'],
            hf_y=data['vst_raw']['Value']
        )
        
        if data['vst_edt'] is not None:
            fig.add_trace(
                go.Scattergl(
                    name='VST_EDT',
                    showlegend=True,
                    line=dict(color='red', width=1),
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>'
                ),
                hf_x=data['vst_edt']['Date'],
                hf_y=data['vst_edt']['Value']
            )
        
        if data['vinge'] is not None:
            fig.add_trace(
                go.Scatter(
                    name='Manual Measurements',
                    mode='markers',
                    marker=dict(color='black', size=6),
                    showlegend=True,
                    hovertemplate='Date: %{x}<br>Value: %{y:.2f} mm<extra></extra>',
                    x=data['vinge']['Date'],
                    y=data['vinge']['W.L [cm]']
                )
            )

        fig.update_layout(
            title=f'Water Level Data Overview - Station {folder}',
            yaxis_title='Water level (mm)',
            xaxis_title='Date',
            template='plotly_white',
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.show_dash(mode='inline', port=8050 + int(folder[-1])) 