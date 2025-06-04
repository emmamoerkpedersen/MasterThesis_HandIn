"""
Visualization utilities specifically for the alternating forecaster model.
These visualizations focus on analyzing model behavior, hidden states, and 
the impact of alternating between original and predicted data inputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

def set_thesis_plot_style():
    """Set a thesis-friendly plot style with larger fonts and a professional appearance."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font sizes - larger for thesis
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    # Set colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    # Set grid style
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.linestyle'] = '--'
    
    # Set figure background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Set spine colors
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Set figure size
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Improve readability
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8


def plot_alternating_pattern_performance(test_data, predictions, week_steps, output_dir=None, station_id='main'):
    """
    Visualize the performance differences between weeks using original data vs. predicted data.
    
    Args:
        test_data: DataFrame containing actual water level values
        predictions: Series or array containing predicted water level values
        week_steps: Number of time steps in a week (for alternating pattern)
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    set_thesis_plot_style()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/visualizations")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract actual values
    actual = test_data['vst_raw']
    
    # Make sure predictions is aligned with actual data
    if isinstance(predictions, np.ndarray):
        # Convert to pandas Series for easier alignment
        predictions = pd.Series(predictions, index=actual.index[:len(predictions)])
    
    # Get common indices
    common_idx = actual.index.intersection(predictions.index)
    actual = actual.loc[common_idx]
    predictions = predictions.loc[common_idx]
    
    # Calculate errors
    errors = (actual - predictions).abs()
    
    # Create week index for each data point (0 for first week, 1 for second week, etc.)
    week_index = np.zeros(len(actual))
    for i in range(len(actual)):
        week_index[i] = (i // week_steps) % 2  # 0 for original weeks, 1 for prediction weeks
    
    # Create DataFrame with week information
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predictions,
        'error': errors,
        'week_type': ['Original' if w == 0 else 'Prediction-based' for w in week_index],
        'timestamp': actual.index
    })
    
    # Calculate error statistics by week type
    error_stats = df.groupby('week_type')['error'].agg(['mean', 'median', 'std']).reset_index()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot 1: Time series with alternating background
    ax1 = fig.add_subplot(gs[0])
    
    # First plot the entire time series
    ax1.plot(actual.index, actual.values, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(predictions.index, predictions.values, 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
    
    # Shade the background differently for original vs prediction weeks
    week_change_idx = [i * week_steps for i in range(1, (len(actual) // week_steps) + 1)]
    week_start_idx = 0
    
    for i, change_idx in enumerate(week_change_idx):
        if change_idx <= len(actual):
            # Determine if this is an original or prediction-based week
            is_original = (i % 2 == 0)
            color = 'lightblue' if is_original else 'lightsalmon'
            alpha = 0.2
            label = 'Original Data Period' if is_original else 'Prediction-based Period'
            
            # Only add label for the first occurrence of each type
            if week_start_idx == 0 or (is_original and 'Original Data Period' not in [l.get_label() for l in ax1.get_lines()]) or \
               (not is_original and 'Prediction-based Period' not in [l.get_label() for l in ax1.get_lines()]):
                ax1.axvspan(actual.index[week_start_idx], actual.index[min(change_idx-1, len(actual)-1)], 
                           alpha=alpha, color=color, label=label)
            else:
                ax1.axvspan(actual.index[week_start_idx], actual.index[min(change_idx-1, len(actual)-1)], 
                           alpha=alpha, color=color)
            
            week_start_idx = change_idx
    
    # Format the first plot
    ax1.set_title(f'Alternating Forecasting Pattern Performance - Station {station_id}', fontweight='bold')
    ax1.set_ylabel('Water Level [mm]', fontweight='bold')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Error by week type as boxplots
    ax2 = fig.add_subplot(gs[1])
    sns.boxplot(x='week_type', y='error', data=df, ax=ax2, hue='week_type', 
               palette={'Original': 'lightblue', 'Prediction-based': 'lightsalmon'}, legend=False)
    
    # Add error statistics as text
    for i, row in error_stats.iterrows():
        stats_text = f"Mean: {row['mean']:.2f}\nMedian: {row['median']:.2f}\nStd: {row['std']:.2f}"
        ax2.text(i, df['error'].max() * 0.9, stats_text, ha='center', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax2.set_title('Error Distribution by Week Type', fontweight='bold')
    ax2.set_xlabel('Week Type', fontweight='bold')
    ax2.set_ylabel('Absolute Error [mm]', fontweight='bold')
    
    # Plot 3: Cumulative error over time with week transitions marked
    ax3 = fig.add_subplot(gs[2])
    
    # Calculate cumulative mean error - fixed to avoid adding to DatetimeIndex
    # Instead of using index + 1, we'll create a range index for the denominator
    cumulative_errors = df['error'].cumsum()
    count_values = np.arange(1, len(df) + 1)
    df['cumulative_error'] = cumulative_errors / count_values
    
    # Plot cumulative error
    ax3.plot(df['timestamp'], df['cumulative_error'], 'k-', linewidth=2)
    
    # Highlight transitions between week types
    transitions = []
    for i in range(1, len(df)):
        if df['week_type'].iloc[i] != df['week_type'].iloc[i-1]:
            transitions.append(i)
    
    for t in transitions:
        if t < len(df):
            ax3.axvline(x=df['timestamp'].iloc[t], color='purple', linestyle='--', alpha=0.7,
                       label='Transition' if t == transitions[0] else None)
    
    ax3.set_title('Cumulative Mean Error Over Time', fontweight='bold')
    ax3.set_ylabel('Cumulative Mean Error [mm]', fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    if transitions:
        ax3.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'alternating_performance_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved alternating pattern performance plot to: {output_path}")
    return output_path


def plot_error_accumulation(test_data, predictions, week_steps, output_dir=None, station_id='main'):
    """
    Visualize how errors accumulate during prediction-based weeks and get corrected
    in original data weeks.
    
    Args:
        test_data: DataFrame containing actual water level values
        predictions: Series or array containing predicted water level values
        week_steps: Number of time steps in a week (for alternating pattern)
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    set_thesis_plot_style()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/visualizations")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract actual values
    actual = test_data['vst_raw']
    
    # Make sure predictions is aligned with actual data
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=actual.index[:len(predictions)])
    
    # Get common indices
    common_idx = actual.index.intersection(predictions.index)
    actual = actual.loc[common_idx]
    predictions = predictions.loc[common_idx]
    
    # Calculate errors
    abs_errors = (actual - predictions).abs()
    
    # Create week index for each data point
    total_weeks = (len(actual) // week_steps) + (1 if len(actual) % week_steps > 0 else 0)
    week_data = []
    
    # Process errors by week
    for week in range(total_weeks):
        start_idx = week * week_steps
        end_idx = min((week + 1) * week_steps, len(actual))
        
        if start_idx < len(actual):
            week_type = 'Original' if week % 2 == 0 else 'Prediction-based'
            
            # Calculate position within week (normalized to 0-1)
            for pos in range(start_idx, end_idx):
                position_in_week = (pos - start_idx) / week_steps
                week_data.append({
                    'week': week,
                    'week_type': week_type,
                    'pos_in_week': position_in_week,
                    'step_in_week': pos - start_idx,
                    'error': abs_errors.iloc[pos],
                    'timestamp': actual.index[pos]
                })
    
    # Convert to DataFrame
    week_df = pd.DataFrame(week_data)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(2, 1)
    
    # Plot 1: Error patterns within weeks
    ax1 = fig.add_subplot(gs[0])
    
    # Group by position in week and week type, then calculate mean error
    position_errors = week_df.groupby(['step_in_week', 'week_type'])['error'].mean().reset_index()
    
    # Pivot for easier plotting
    pivot_errors = position_errors.pivot(index='step_in_week', columns='week_type', values='error')
    
    # Plot error pattern by position in week
    if 'Original' in pivot_errors.columns:
        ax1.plot(pivot_errors.index, pivot_errors['Original'], 'b-', 
               label='Original Weeks', linewidth=2.5)
    
    if 'Prediction-based' in pivot_errors.columns:
        ax1.plot(pivot_errors.index, pivot_errors['Prediction-based'], 'r-', 
               label='Prediction-based Weeks', linewidth=2.5)
    
    # Format the plot
    ax1.set_title('Error Accumulation Within Weeks', fontweight='bold')
    ax1.set_xlabel('Time Step Within Week', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error [mm]', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot 2: Heatmap of errors across all weeks
    ax2 = fig.add_subplot(gs[1])
    
    # Create pivot table for heatmap - weeks as rows, position as columns
    max_step = week_df['step_in_week'].max()
    
    # Prepare data for heatmap with consistent dimensions
    heatmap_data = np.zeros((total_weeks, int(max_step) + 1))
    heatmap_data.fill(np.nan)  # Fill with NaN for missing values
    
    for w in range(total_weeks):
        week_subset = week_df[week_df['week'] == w]
        for s in range(int(max_step) + 1):
            step_data = week_subset[week_subset['step_in_week'] == s]
            if not step_data.empty:
                heatmap_data[w, s] = step_data['error'].values[0]
    
    # Create a custom colormap that goes from blue to red
    colors = ['#d4f1f9', '#9ed9ea', '#6abcdb', '#3e9ec2', '#1177a5', '#084d91']
    custom_cmap = LinearSegmentedColormap.from_list('blue_to_red', colors)
    
    # Plot heatmap
    im = ax2.imshow(heatmap_data, aspect='auto', cmap=custom_cmap)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Absolute Error [mm]', fontweight='bold')
    
    # Format the plot
    ax2.set_title('Error Heatmap Across All Weeks', fontweight='bold')
    ax2.set_xlabel('Time Step Within Week', fontweight='bold')
    ax2.set_ylabel('Week Number', fontweight='bold')
    
    # Mark alternating weeks
    for w in range(total_weeks):
        label = 'Original' if w % 2 == 0 else 'Prediction'
        ax2.text(-10, w, label, va='center', ha='right', 
               color='blue' if w % 2 == 0 else 'red',
               fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'error_accumulation_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error accumulation plot to: {output_path}")
    return output_path


def plot_recovery_after_transition(test_data, predictions, week_steps, output_dir=None, station_id='main', transition_window=48):
    """
    Analyze and visualize how the model recovers after transitions between original and
    prediction-based inputs.
    
    Args:
        test_data: DataFrame containing actual water level values
        predictions: Series or array containing predicted water level values
        week_steps: Number of time steps in a week (for alternating pattern)
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        transition_window: Number of time steps to analyze before/after transitions
        
    Returns:
        Path to the saved PNG file
    """
    set_thesis_plot_style()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/visualizations")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract actual values
    actual = test_data['vst_raw']
    
    # Make sure predictions is aligned with actual data
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=actual.index[:len(predictions)])
    
    # Get common indices
    common_idx = actual.index.intersection(predictions.index)
    actual = actual.loc[common_idx]
    predictions = predictions.loc[common_idx]
    
    # Calculate errors
    errors = (actual - predictions).abs()
    
    # Create week type markers
    week_index = np.zeros(len(actual))
    for i in range(len(actual)):
        week_index[i] = (i // week_steps) % 2  # 0 for original weeks, 1 for prediction weeks
    
    # Find transition points (from original to prediction-based and vice versa)
    transitions = []
    for i in range(1, len(week_index)):
        if week_index[i] != week_index[i-1]:
            transitions.append({
                'index': i,
                'from': 'Original' if week_index[i-1] == 0 else 'Prediction-based',
                'to': 'Original' if week_index[i] == 0 else 'Prediction-based',
                'timestamp': actual.index[i]
            })
    
    # Collect data around each transition
    transition_data = []
    
    for t in transitions:
        # Get window before and after transition
        start_idx = max(0, t['index'] - transition_window)
        end_idx = min(len(actual), t['index'] + transition_window)
        
        # Calculate relative positions
        for i in range(start_idx, end_idx):
            rel_pos = i - t['index']  # Negative for before, positive for after
            transition_data.append({
                'relative_pos': rel_pos,
                'error': errors.iloc[i],
                'actual': actual.iloc[i],
                'predicted': predictions.iloc[i],
                'transition_type': f"{t['from']} to {t['to']}",
                'timestamp': actual.index[i]
            })
    
    # Convert to DataFrame
    trans_df = pd.DataFrame(transition_data)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 1, height_ratios=[1.5, 1, 1])
    
    # Plot 1: Error patterns around transitions
    ax1 = fig.add_subplot(gs[0])
    
    # Group by transition type and relative position
    transition_errors = trans_df.groupby(['transition_type', 'relative_pos'])['error'].mean().reset_index()
    
    # Pivot for easier plotting
    pivot_errors = transition_errors.pivot(index='relative_pos', columns='transition_type', values='error')
    
    # Plot error pattern around transitions
    for col in pivot_errors.columns:
        linestyle = '-' if 'Original to Prediction' in col else '--'
        color = 'r' if 'Original to Prediction' in col else 'b'
        ax1.plot(pivot_errors.index, pivot_errors[col], linestyle=linestyle, color=color,
                label=col, linewidth=2.5)
    
    # Add vertical line at transition point
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, label='Transition Point')
    
    # Format the plot
    ax1.set_title('Error Patterns Around Transitions', fontweight='bold')
    ax1.set_xlabel('Time Steps Relative to Transition', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error [mm]', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Add text annotations explaining the transitions
    annotation_text = (
        "Negative time steps: Before transition\n"
        "Positive time steps: After transition\n"
        "Solid red line: Transition from Original to Prediction-based inputs\n"
        "Dashed blue line: Transition from Prediction-based to Original inputs"
    )
    ax1.text(0.02, 0.98, annotation_text, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Plot 2: Error distribution before and after transition (boxplots)
    ax2 = fig.add_subplot(gs[1])
    
    # Create categorical variable for boxplot
    trans_df['position_category'] = 'During'
    trans_df.loc[trans_df['relative_pos'] < -24, 'position_category'] = 'Well Before'
    trans_df.loc[(trans_df['relative_pos'] >= -24) & (trans_df['relative_pos'] < 0), 'position_category'] = 'Just Before'
    trans_df.loc[(trans_df['relative_pos'] > 0) & (trans_df['relative_pos'] <= 24), 'position_category'] = 'Just After'
    trans_df.loc[trans_df['relative_pos'] > 24, 'position_category'] = 'Well After'
    
    # Order categories
    order = ['Well Before', 'Just Before', 'During', 'Just After', 'Well After']
    
    # Separate by transition type
    orig_to_pred = trans_df[trans_df['transition_type'] == 'Original to Prediction-based']
    pred_to_orig = trans_df[trans_df['transition_type'] == 'Prediction-based to Original']
    
    if not orig_to_pred.empty:
        sns.boxplot(x='position_category', y='error', data=orig_to_pred, 
                  ax=ax2, order=order, color='lightcoral', showfliers=False)
    
    # Format the plot
    ax2.set_title('Error Distribution Around Transitions (Original to Prediction-based)', fontweight='bold')
    ax2.set_xlabel('Position Relative to Transition', fontweight='bold')
    ax2.set_ylabel('Absolute Error [mm]', fontweight='bold')
    
    # Plot 3: Error distribution for the other transition type
    ax3 = fig.add_subplot(gs[2])
    
    if not pred_to_orig.empty:
        sns.boxplot(x='position_category', y='error', data=pred_to_orig, 
                  ax=ax3, order=order, color='lightblue', showfliers=False)
    
    # Format the plot
    ax3.set_title('Error Distribution Around Transitions (Prediction-based to Original)', fontweight='bold')
    ax3.set_xlabel('Position Relative to Transition', fontweight='bold')
    ax3.set_ylabel('Absolute Error [mm]', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'recovery_pattern_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved recovery pattern plot to: {output_path}")
    return output_path


def plot_hidden_state_evolution(model, test_data, week_steps, output_dir=None, station_id='main', num_states_to_show=5):
    """
    Visualize how LSTM hidden states evolve during the alternating forecasting pattern.
    
    Args:
        model: Trained AlternatingForecastModel instance
        test_data: DataFrame containing test data with features
        week_steps: Number of time steps in a week (for alternating pattern)
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        num_states_to_show: Number of hidden state dimensions to visualize (top by variance)
        
    Returns:
        Path to the saved PNG file
    """
    set_thesis_plot_style()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/visualizations")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure we're in evaluation mode
    model.eval()
    
    # Get device
    device = next(model.parameters()).device
    
    # Extract features from test data - need to match with what the model expects
    feature_cols = [col for col in test_data.columns if col != 'vst_raw']
    
    # Only use the first N columns where N is model.input_size
    if len(feature_cols) > model.input_size:
        feature_cols = feature_cols[:model.input_size]
    
    # Prepare input features
    features = test_data[feature_cols].values
    
    # Convert to tensor
    x = torch.FloatTensor(features).unsqueeze(0).to(device)  # [1, seq_len, input_size]
    
    # Variables to store hidden states
    hidden_states = []
    week_indices = []
    
    # Process through model, collecting hidden states
    with torch.no_grad():
        hidden_state, cell_state = None, None
        
        # Create a week index marker
        for t in range(x.shape[1]):
            week_idx = (t // week_steps) % 2  # 0 for original weeks, 1 for prediction weeks
            week_indices.append(week_idx)
            
            # Create binary flag based on week
            binary_flag = torch.ones(1, 1, device=device) if week_idx == 1 else torch.zeros(1, 1, device=device)
            
            # Prepare input with flag
            current_input = torch.cat([x[:, t, :], binary_flag], dim=1)
            
            # Process through LSTM cell
            hidden_state, cell_state = model.lstm_cell(current_input, 
                                                     (hidden_state, cell_state) if hidden_state is not None else None)
            
            # Store the hidden state
            hidden_states.append(hidden_state.cpu().numpy())
    
    # Convert to numpy array
    hidden_states = np.array(hidden_states).squeeze()  # [seq_len, hidden_size]
    
    # Select the hidden state dimensions with highest variance to visualize
    if hidden_states.shape[1] > num_states_to_show:
        # Calculate variance of each dimension
        variances = np.var(hidden_states, axis=0)
        # Get indices of top dimensions by variance
        top_dims = np.argsort(variances)[-num_states_to_show:]
        # Select only these dimensions
        hidden_states_selected = hidden_states[:, top_dims]
    else:
        hidden_states_selected = hidden_states
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot 1: Hidden state dimensions over time
    ax1 = fig.add_subplot(gs[0])
    
    # Plot each selected hidden state dimension
    for i in range(hidden_states_selected.shape[1]):
        ax1.plot(hidden_states_selected[:, i], label=f'Dim {i+1}', linewidth=1.5)
    
    # Add vertical lines at week transitions
    for t in range(week_steps, len(hidden_states), week_steps):
        if t < len(hidden_states):
            # Determine if transition is to original or prediction week
            is_to_original = (t // week_steps) % 2 == 0
            label = 'To Original' if is_to_original else 'To Prediction'
            color = 'blue' if is_to_original else 'red'
            
            # Only add label for the first occurrence of each type
            if t == week_steps or (is_to_original and 'To Original' not in [l.get_label() for l in ax1.get_lines()]) or \
               (not is_to_original and 'To Prediction' not in [l.get_label() for l in ax1.get_lines()]):
                ax1.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
            else:
                ax1.axvline(x=t, color=color, linestyle='--', alpha=0.7)
    
    # Format the plot
    ax1.set_title(f'LSTM Hidden State Evolution - Station {station_id}', fontweight='bold')
    ax1.set_xlabel('Time Step', fontweight='bold')
    ax1.set_ylabel('Hidden State Value', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Heatmap of hidden states
    ax2 = fig.add_subplot(gs[1])
    
    # Create heatmap
    im = ax2.imshow(hidden_states_selected.T, aspect='auto', cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Hidden State Value', fontweight='bold')
    
    # Format the plot
    ax2.set_title('Hidden State Heatmap', fontweight='bold')
    ax2.set_xlabel('Time Step', fontweight='bold')
    ax2.set_ylabel('Hidden State Dimension', fontweight='bold')
    
    # Add week type indicators
    ax2.set_yticks(np.arange(hidden_states_selected.shape[1]))
    ax2.set_yticklabels([f'Dim {i+1}' for i in range(hidden_states_selected.shape[1])])
    
    # Plot 3: Hidden state statistics by week type
    ax3 = fig.add_subplot(gs[2])
    
    # Separate hidden states by week type
    orig_weeks = [hidden_states_selected[i] for i in range(len(hidden_states_selected)) if week_indices[i] == 0]
    pred_weeks = [hidden_states_selected[i] for i in range(len(hidden_states_selected)) if week_indices[i] == 1]
    
    # Convert to numpy arrays
    orig_weeks = np.array(orig_weeks) if orig_weeks else np.array([]).reshape(0, hidden_states_selected.shape[1])
    pred_weeks = np.array(pred_weeks) if pred_weeks else np.array([]).reshape(0, hidden_states_selected.shape[1])
    
    # Calculate statistics
    orig_mean = np.mean(orig_weeks, axis=0) if orig_weeks.size > 0 else np.zeros(hidden_states_selected.shape[1])
    pred_mean = np.mean(pred_weeks, axis=0) if pred_weeks.size > 0 else np.zeros(hidden_states_selected.shape[1])
    orig_std = np.std(orig_weeks, axis=0) if orig_weeks.size > 0 else np.zeros(hidden_states_selected.shape[1])
    pred_std = np.std(pred_weeks, axis=0) if pred_weeks.size > 0 else np.zeros(hidden_states_selected.shape[1])
    
    # Set up dimensions for plotting
    dims = np.arange(hidden_states_selected.shape[1])
    width = 0.35
    
    # Create grouped bar chart
    ax3.bar(dims - width/2, orig_mean, width, yerr=orig_std, capsize=5, label='Original Weeks')
    ax3.bar(dims + width/2, pred_mean, width, yerr=pred_std, capsize=5, label='Prediction Weeks')
    
    # Format the plot
    ax3.set_title('Hidden State Statistics by Week Type', fontweight='bold')
    ax3.set_xlabel('Hidden State Dimension', fontweight='bold')
    ax3.set_ylabel('Mean Value (with std error)', fontweight='bold')
    ax3.set_xticks(dims)
    ax3.set_xticklabels([f'Dim {i+1}' for i in range(hidden_states_selected.shape[1])])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'hidden_state_evolution_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved hidden state evolution plot to: {output_path}")
    return output_path


def plot_input_flag_impact(test_data, predictions, week_steps, output_dir=None, station_id='main'):
    """
    Analyze and visualize how the binary input flag (indicating original vs predicted inputs)
    affects prediction performance.
    
    Args:
        test_data: DataFrame containing actual water level values
        predictions: Series or array containing predicted water level values
        week_steps: Number of time steps in a week (for alternating pattern)
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        
    Returns:
        Path to the saved PNG file
    """
    set_thesis_plot_style()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/visualizations")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract actual values
    actual = test_data['vst_raw']
    
    # Make sure predictions is aligned with actual data
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=actual.index[:len(predictions)])
    
    # Get common indices
    common_idx = actual.index.intersection(predictions.index)
    actual = actual.loc[common_idx]
    predictions = predictions.loc[common_idx]
    
    # Calculate errors and performance metrics
    abs_errors = (actual - predictions).abs()
    sq_errors = (actual - predictions) ** 2
    
    # Create binary flag indicators
    flag_values = np.zeros(len(actual))
    for i in range(len(actual)):
        # 0 for original data weeks, 1 for prediction-based weeks
        flag_values[i] = (i // week_steps) % 2
    
    # Create DataFrame with flag information
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predictions,
        'abs_error': abs_errors,
        'sq_error': sq_errors,
        'binary_flag': flag_values,
        'timestamp': actual.index
    })
    
    # Add some additional features if available in test_data
    for col in ['temperature', 'rainfall']:
        if col in test_data.columns:
            df[col] = test_data[col].loc[common_idx].values
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(2, 2)
    
    # Plot 1: Actual vs Predicted scatter plot colored by flag
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Scatter plot colored by flag value
    scatter = ax1.scatter(df['actual'], df['predicted'], 
                        c=df['binary_flag'], cmap='coolwarm', 
                        alpha=0.5, s=20)
    
    # Add perfect prediction line (y=x)
    min_val = min(df['actual'].min(), df['predicted'].min())
    max_val = max(df['actual'].max(), df['predicted'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect prediction')
    
    # Format the plot
    ax1.set_title('Actual vs Predicted by Input Flag', fontweight='bold')
    ax1.set_xlabel('Actual Water Level [mm]', fontweight='bold')
    ax1.set_ylabel('Predicted Water Level [mm]', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for flag values
    legend1 = ax1.legend(*scatter.legend_elements(),
                        loc="lower right", title="Input Flag")
    ax1.add_artist(legend1)
    ax1.legend()
    
    # Plot 2: Error distribution by flag value
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for violin plot
    flag_0_errors = df[df['binary_flag'] == 0]['abs_error']
    flag_1_errors = df[df['binary_flag'] == 1]['abs_error']
    
    # Create violin plot
    violin_data = [flag_0_errors, flag_1_errors]
    ax2.violinplot(violin_data, showmeans=True, showmedians=True)
    
    # Add boxplot inside violin
    box_data = [flag_0_errors, flag_1_errors]
    ax2.boxplot(box_data, positions=[1, 2], widths=0.3, patch_artist=True,
              boxprops=dict(facecolor='lightblue', alpha=0.5),
              showfliers=False)
    
    # Format the plot
    ax2.set_title('Error Distribution by Input Flag', fontweight='bold')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Original (0)', 'Prediction-based (1)'])
    ax2.set_xlabel('Input Flag Value', fontweight='bold')
    ax2.set_ylabel('Absolute Error [mm]', fontweight='bold')
    
    # Add statistical summary
    for i, flag in enumerate([0, 1]):
        flag_data = df[df['binary_flag'] == flag]['abs_error']
        stats_text = (
            f"Mean: {flag_data.mean():.2f}\n"
            f"Median: {flag_data.median():.2f}\n"
            f"Std: {flag_data.std():.2f}\n"
            f"Min: {flag_data.min():.2f}\n"
            f"Max: {flag_data.max():.2f}"
        )
        ax2.text(i+1, flag_data.max() * 0.95, stats_text, ha='center', 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Plot 3: Time series with flag indicators
    ax3 = fig.add_subplot(gs[1, :])
    
    # Plot actual and predicted values
    ax3.plot(df['timestamp'], df['actual'], 'b-', label='Actual', linewidth=1.2, alpha=0.7)
    ax3.plot(df['timestamp'], df['predicted'], 'r-', label='Predicted', linewidth=1.2, alpha=0.7)
    
    # Shade the background by flag value
    week_change_idx = [i * week_steps for i in range(1, (len(actual) // week_steps) + 1)]
    week_start_idx = 0
    
    for i, change_idx in enumerate(week_change_idx):
        if change_idx <= len(actual):
            # Determine if this is original or prediction-based input
            is_original = (i % 2 == 0)
            color = 'lightblue' if is_original else 'lightsalmon'
            alpha = 0.2
            label = 'Flag=0 (Original)' if is_original else 'Flag=1 (Prediction)'
            
            # Only add label for the first occurrence of each type
            if week_start_idx == 0 or (is_original and 'Flag=0' not in [l.get_label() for l in ax3.get_lines()]) or \
               (not is_original and 'Flag=1' not in [l.get_label() for l in ax3.get_lines()]):
                ax3.axvspan(actual.index[week_start_idx], actual.index[min(change_idx-1, len(actual)-1)], 
                          alpha=alpha, color=color, label=label)
            else:
                ax3.axvspan(actual.index[week_start_idx], actual.index[min(change_idx-1, len(actual)-1)], 
                          alpha=alpha, color=color)
            
            week_start_idx = change_idx
    
    # Format the plot
    ax3.set_title('Prediction Performance with Binary Flag Highlighting', fontweight='bold')
    ax3.set_xlabel('Time', fontweight='bold')
    ax3.set_ylabel('Water Level [mm]', fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'input_flag_impact_{station_id}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved input flag impact plot to: {output_path}")
    return output_path


def plot_zoom_comparison(actual_data, predictions, output_dir=None, station_id='main', 
                        corrupted_data=None, zoom_start=None, zoom_end=None, 
                        title_suffix="", show_alternating_periods=False, week_steps=672):
    """
    Create a zoom plot comparing actual data, predictions, and optionally corrupted data
    for a specified time period.
    
    Args:
        actual_data: Series or DataFrame containing actual water level values (vst_raw)
        predictions: Series or array containing predicted water level values
        output_dir: Optional output directory path
        station_id: Station identifier for plot title
        corrupted_data: Optional Series containing corrupted data (if synthetic errors injected)
        zoom_start: Start date/time for zoom period (if None, uses first 30 days)
        zoom_end: End date/time for zoom period (if None, uses zoom_start + 30 days)
        title_suffix: Additional text to add to plot title
        show_alternating_periods: Whether to highlight alternating periods (original vs prediction) - default False
        week_steps: Number of time steps in a week (for alternating pattern highlighting)
        
    Returns:
        Path to the saved PNG file
    """
    set_thesis_plot_style()
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("results/visualizations")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract actual values (handle both Series and DataFrame)
    if isinstance(actual_data, pd.DataFrame):
        if 'vst_raw' in actual_data.columns:
            actual = actual_data['vst_raw']
        else:
            # Use the first column if vst_raw not found
            actual = actual_data.iloc[:, 0]
    else:
        actual = actual_data
    
    # Handle corrupted data if provided
    corrupted = None
    if corrupted_data is not None:
        if isinstance(corrupted_data, pd.DataFrame):
            if 'vst_raw' in corrupted_data.columns:
                corrupted = corrupted_data['vst_raw']
            else:
                corrupted = corrupted_data.iloc[:, 0]
        else:
            corrupted = corrupted_data
    
    # Make sure predictions is aligned with actual data
    if isinstance(predictions, np.ndarray):
        predictions = pd.Series(predictions, index=actual.index[:len(predictions)])
    
    # Set default zoom period if not provided
    if zoom_start is None:
        zoom_start = actual.index[0]
    if zoom_end is None:
        zoom_end = zoom_start + pd.Timedelta(days=30)
    
    # Convert string dates to timestamps if needed
    if isinstance(zoom_start, str):
        zoom_start = pd.Timestamp(zoom_start)
    if isinstance(zoom_end, str):
        zoom_end = pd.Timestamp(zoom_end)
    
    # Filter data to zoom period
    zoom_mask = (actual.index >= zoom_start) & (actual.index <= zoom_end)
    actual_zoom = actual.loc[zoom_mask]
    
    # Get common indices for the zoom period
    common_idx = actual_zoom.index.intersection(predictions.index)
    if len(common_idx) == 0:
        print(f"Warning: No overlapping data found for zoom period {zoom_start} to {zoom_end}")
        return None
    
    # Filter all data to common indices
    actual_zoom = actual_zoom.loc[common_idx]
    predictions_zoom = predictions.loc[common_idx]
    
    if corrupted is not None:
        corrupted_zoom = corrupted.loc[corrupted.index.intersection(common_idx)]
    else:
        corrupted_zoom = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot the data series with updated colors
    ax.plot(actual_zoom.index, actual_zoom.values, 'b-', label='Actual Data', 
            linewidth=0.8, alpha=0.8)
    ax.plot(predictions_zoom.index, predictions_zoom.values, 'g-', label='Predictions', 
            linewidth=0.8, alpha=0.8)
    
    # Plot corrupted data if available - red stippled line
    if corrupted_zoom is not None and len(corrupted_zoom) > 0:
        ax.plot(corrupted_zoom.index, corrupted_zoom.values, 'r:', 
                label='VST Raw with Errors', 
                linewidth=2.0, alpha=0.8)
    
    # Highlight alternating periods if requested
    if show_alternating_periods and week_steps is not None:
        # Calculate which periods are original vs prediction-based
        start_time = common_idx[0]
        full_actual_index = actual.index
        
        # Find the position of our zoom start in the full dataset
        if start_time in full_actual_index:
            start_position = full_actual_index.get_loc(start_time)
        else:
            # Find the closest position
            start_position = full_actual_index.searchsorted(start_time)
        
        # Calculate week transitions within the zoom period
        for i, timestamp in enumerate(common_idx):
            current_position = start_position + i
            week_num = (current_position // week_steps) % 2
            
            # Only mark transitions, not every point
            if i > 0:
                prev_position = start_position + i - 1
                prev_week_num = (prev_position // week_steps) % 2
                
                if week_num != prev_week_num:
                    # This is a transition point
                    color = 'blue' if week_num == 0 else 'red'
                    label = 'Original Data Period' if week_num == 0 else 'Prediction-based Period'
                    
                    ax.axvline(x=timestamp, color=color, linestyle=':', alpha=0.6, linewidth=1)
        
        # Add background shading for the periods
        current_start = common_idx[0]
        current_position = start_position
        current_week_type = (current_position // week_steps) % 2
        
        for i, timestamp in enumerate(common_idx[1:], 1):
            position = start_position + i
            week_type = (position // week_steps) % 2
            
            if week_type != current_week_type or i == len(common_idx) - 1:
                # End of current period or end of data
                end_time = timestamp if week_type != current_week_type else common_idx[-1]
                
                # Add shading
                color = 'lightblue' if current_week_type == 0 else 'lightsalmon'
                alpha = 0.15
                ax.axvspan(current_start, end_time, alpha=alpha, color=color)
                
                # Update for next period
                current_start = timestamp
                current_week_type = week_type
    
    # Format the plot
    title = f'Water Level Comparison - Station {station_id}'
    if title_suffix:
        title += f' {title_suffix}'
    title += f'\nZoom Period: {zoom_start.strftime("%Y-%m-%d")} to {zoom_end.strftime("%Y-%m-%d")}'
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Water Level [mm]', fontweight='bold')
    
    # Format x-axis to show only year and month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    zoom_period_str = f"{zoom_start.strftime('%Y%m%d')}_{zoom_end.strftime('%Y%m%d')}"
    output_path = output_dir / f'zoom_comparison_{station_id}_{zoom_period_str}_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved zoom comparison plot to: {output_path}")
    return output_path
