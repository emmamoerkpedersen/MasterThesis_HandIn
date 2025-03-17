"""
Diagnostic tools for hyperparameter tuning results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats

def convert_to_serializable(obj):
    """Convert objects to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_hyperparameter_results(results: List[Dict], best_config: Dict, output_dir: Path):
    """Save hyperparameter tuning results to JSON."""
    # Convert results to serializable format
    serializable_results = convert_to_serializable(results)
    serializable_config = convert_to_serializable(best_config)
    
    # Create output dictionary
    output = {
        'results': serializable_results,
        'best_config': serializable_config
    }
    
    # Save to file
    output_path = output_dir / 'hyperparameter_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def plot_best_model_results(
    station_name: str,
    results: Dict,
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    output_dir: Path,
    trial_number: int
):
    """Plot results for the best model found during hyperparameter tuning."""
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Implement our own plotting logic instead of calling plot_reconstruction_results
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot original vs modified data
    axes[0].plot(original_data['timestamp'], original_data['vst_raw'], 
                 label='Original', color='blue', alpha=0.7)
    axes[0].plot(modified_data['timestamp'], modified_data['vst_raw'], 
                 label='Modified', color='red', alpha=0.7)
    axes[0].set_title(f'Original vs Modified Data - Station {station_name} - Trial {trial_number}')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot prediction if available
    if 'vst_pred' in modified_data.columns:
        axes[1].plot(modified_data['timestamp'], modified_data['vst_raw'], 
                     label='Actual', color='blue', alpha=0.7)
        axes[1].plot(modified_data['timestamp'], modified_data['vst_pred'], 
                     label='Predicted', color='green', alpha=0.7)
        axes[1].set_title(f'Actual vs Predicted - Trial {trial_number}')
        axes[1].set_ylabel('Value')
        axes[1].set_xlabel('Timestamp')
        axes[1].legend()
        axes[1].grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(plots_dir / f'best_model_results_trial_{trial_number}.png')
    plt.close()
    
    # Create an interactive plot with Plotly
    fig = make_subplots(rows=2, cols=1, 
                         shared_xaxes=True,
                         subplot_titles=(f'Original vs Modified Data - Station {station_name}',
                                         'Actual vs Predicted'))
    
    # Add traces for original vs modified
    fig.add_trace(
        go.Scatter(x=original_data['timestamp'], y=original_data['vst_raw'],
                   mode='lines', name='Original', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=modified_data['timestamp'], y=modified_data['vst_raw'],
                   mode='lines', name='Modified', line=dict(color='red')),
        row=1, col=1
    )
    
    # Add traces for actual vs predicted
    if 'vst_pred' in modified_data.columns:
        fig.add_trace(
            go.Scatter(x=modified_data['timestamp'], y=modified_data['vst_raw'],
                       mode='lines', name='Actual', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=modified_data['timestamp'], y=modified_data['vst_pred'],
                       mode='lines', name='Predicted', line=dict(color='green')),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Model Results - Trial {trial_number}",
        showlegend=True
    )
    
    # Save as interactive HTML
    fig.write_html(plots_dir / f'best_model_results_trial_{trial_number}.html')

def generate_hyperparameter_report(
    results: List[Dict],
    best_config: Dict,
    output_dir: Path,
    evaluation_metric: str
):
    """Generate a comprehensive report of hyperparameter tuning results."""
    try:
        # Create report directory
        report_dir = output_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
    
        print("\nGenerating hyperparameter tuning visualizations...")
        print(f"Number of trials in results: {len(results)}")
        print(f"Output directory: {output_dir}")
        
        # Generate all available visualizations if we have enough data
        if len(results) >= 3:
            print("Generating pairwise interaction plots...")
            try:
                plot_hyperparameter_pairwise_interactions(results, output_dir)
                print("✓ Pairwise interaction plots generated")
            except Exception as e:
                print(f"✗ Error generating pairwise interaction plots: {e}")
            
            print("Generating hyperparameter importance analysis...")
            try:
                analyze_hyperparameter_importance(results, output_dir)
                print("✓ Hyperparameter importance analysis generated")
            except Exception as e:
                print(f"✗ Error generating hyperparameter importance analysis: {e}")
            
            print("Generating parameter sensitivity plots...")
            try:
                plot_parameter_sensitivity(results, output_dir)
                print("✓ Parameter sensitivity plots generated")
            except Exception as e:
                print(f"✗ Error generating parameter sensitivity plots: {e}")
            
            print("Generating training dynamics plots...")
            try:
                plot_training_dynamics(results, output_dir)
                print("✓ Training dynamics plots generated")
            except Exception as e:
                print(f"✗ Error generating training dynamics plots: {e}")
            
            print("Generating training time analysis...")
            try:
                plot_training_time_analysis(results, output_dir)
                print("✓ Training time analysis generated")
            except Exception as e:
                print(f"✗ Error generating training time analysis: {e}")
            
            print("Generating parameter evolution plots...")
            try:
                plot_parameter_evolution(results, output_dir)
                print("✓ Parameter evolution plots generated")
            except Exception as e:
                print(f"✗ Error generating parameter evolution plots: {e}")
            
            print("Generating parallel coordinates plot...")
            try:
                plot_parallel_coordinates_clustered(results, output_dir)
                print("✓ Parallel coordinates plot generated")
            except Exception as e:
                print(f"✗ Error generating parallel coordinates plot: {e}")
                
            print("Generating 3D surface plots...")
            try:
                create_3d_surface_plots(results, output_dir)
                print("✓ 3D surface plots generated")
            except Exception as e:
                print(f"✗ Error generating 3D surface plots: {e}")
            
            print("Generating learning curve clusters...")
            try:
                plot_learning_curve_clusters(results, output_dir)
                print("✓ Learning curve clusters generated")
            except Exception as e:
                print(f"✗ Error generating learning curve clusters: {e}")
                
            print("Generating learning rate landscape...")
            try:
                plot_learning_rate_landscape(results, output_dir)
                print("✓ Learning rate landscape generated")
            except Exception as e:
                print(f"✗ Error generating learning rate landscape: {e}")
                
            print("Generating network architecture performance...")
            try:
                plot_network_architecture_performance(results, output_dir)
                print("✓ Network architecture performance generated")
            except Exception as e:
                print(f"✗ Error generating network architecture performance: {e}")
                
            print("Generating top N distributions...")
            try:
                plot_top_n_distributions(results, output_dir)
                print("✓ Top N distributions generated")
            except Exception as e:
                print(f"✗ Error generating top N distributions: {e}")
                
            print("Generating architecture impact plot...")
            try:
                generate_architecture_impact_plot(results, output_dir)
                print("✓ Architecture impact plot generated")
            except Exception as e:
                print(f"✗ Error generating architecture impact plot: {e}")
        else:
            print(f"Not enough trials ({len(results)}) for detailed visualizations. Need at least 3 trials.")
        
        # Generate HTML report
        report_path = report_dir / "hyperparameter_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("""
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .section { margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>Hyperparameter Tuning Report</h1>
            """)
            
            # Best Configuration
            f.write("<div class='section'><h2>Best Configuration</h2>")
            f.write("<table><tr><th>Parameter</th><th>Value</th></tr>")
            for param, value in best_config.items():
                f.write(f"<tr><td>{param.replace('_', ' ').title()}</td><td>{value}</td></tr>")
            f.write("</table></div>")
            
            # Trial Statistics
            f.write("<div class='section'><h2>Trial Statistics</h2>")
            f.write("<table><tr><th>Metric</th><th>Value</th></tr>")
            
            # Calculate statistics
            scores = []
            for r in results:
                if 'score' in r:
                    scores.append(r['score'])
                elif 'value' in r:
                    scores.append(r['value'])
                    
            if scores:
                f.write(f"<tr><td>Number of Trials</td><td>{len(scores)}</td></tr>")
                f.write(f"<tr><td>Best Score</td><td>{min(scores):.6f}</td></tr>")
                f.write(f"<tr><td>Mean Score</td><td>{np.mean(scores):.6f}</td></tr>")
                f.write(f"<tr><td>Score Std Dev</td><td>{np.std(scores):.6f}</td></tr>")
            
            f.write("</table></div>")
            
            # Add links to generated plots
            f.write("<div class='section'><h2>Generated Visualizations</h2>")
            f.write("<ul>")
            
            # Check for generated plots in the output directory
            plot_files = list(output_dir.glob("*.png"))
            plot_files.extend(list((output_dir / "plots").glob("*.png")) if (output_dir / "plots").exists() else [])
            
            if plot_files:
                for plot_file in plot_files:
                    plot_name = plot_file.name
                    relative_path = plot_file.relative_to(output_dir) if output_dir in plot_file.parents else plot_file.name
                    f.write(f"<li><a href='../{relative_path}'>{plot_name.replace('_', ' ').replace('.png', '').title()}</a></li>")
            else:
                f.write("<li>No visualization files found</li>")
                
            f.write("</ul></div>")
            
            f.write("</body></html>")
        
        # Save results as JSON for future reference
        save_hyperparameter_results(results, best_config, output_dir)
        
        print(f"Generated hyperparameter report at {report_path}")
        
    except Exception as e:
        print(f"Error generating hyperparameter report: {e}")
        import traceback
        print(traceback.format_exc())

def plot_hyperparameter_pairwise_interactions(results, output_dir):
    """Create pairwise interaction plots between hyperparameters."""
    # Extract data from results, focusing on core parameters
    core_params = [
        'sequence_length', 'dropout_rate', 'learning_rate', 
        'batch_size', 'epochs'
    ]
    
    data = []
    for result in results:
        if 'config' in result and 'score' in result and not np.isinf(result['score']):
            row = {k: v for k, v in result['config'].items() 
                  if k in core_params and not isinstance(v, list)}
            
            # Handle lstm_units specially
            if 'lstm_units' in result['config']:
                row['lstm_layer_1'] = result['config']['lstm_units'][0]
                if len(result['config']['lstm_units']) > 1:
                    row['lstm_layer_2'] = result['config']['lstm_units'][1]
            
            row['score'] = result['score']
            data.append(row)
    
    if not data:
        print("No valid data for pairwise interaction plot")
        return
        
    df = pd.DataFrame(data)
    
    # Check if we have enough data points and variables
    if len(df) < 5 or len(df.columns) < 3:
        print(f"Not enough data for pairwise plot: {len(df)} rows, {len(df.columns)} columns")
        return
    
    try:
        # Format column names for better readability
        df.columns = [col.replace('_', ' ').title() if col != 'score' else 'Score' for col in df.columns]
        
        # Create pairwise plots with custom styling
        plt.figure(figsize=(20, 20))
        g = sns.PairGrid(df, diag_sharey=False, corner=True)
        g.map_lower(sns.scatterplot, s=80, alpha=0.6, palette='viridis')
        g.map_diag(sns.kdeplot, fill=True, alpha=0.6, linewidth=2)
        
        # Add correlation coefficients more safely with better positioning
        for i in range(len(g.axes)):
            for j in range(len(g.axes[i])):
                if i > j and g.axes[i, j] is not None:  # Lower triangle and not None
                    try:
                        corr = df.iloc[:, j].corr(df.iloc[:, i])
                        # Color code correlation text based on strength
                        text_color = 'red' if abs(corr) > 0.7 else ('blue' if abs(corr) > 0.4 else 'black')
                        g.axes[i, j].annotate(
                            f'ρ = {corr:.2f}',
                            xy=(.2, .95), xycoords='axes fraction',  # Position in top-left to avoid data points
                            ha='left', va='top', fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                            color=text_color
                        )
                    except Exception as e:
                        print(f"Error adding correlation text: {e}")
        
        # Format axis labels to be more readable
        for ax in g.axes.flat:
            if ax is not None:
                ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='bold')
                ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='bold')
                ax.tick_params(labelsize=10)
        
        plt.suptitle('Hyperparameter Pairwise Interactions', fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to accommodate title
        plt.savefig(output_dir / 'hyperparameter_pairwise.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating pairwise plot: {e}")

def analyze_hyperparameter_importance(results: List[Dict], output_dir: Path):
    """Analyze the importance of different hyperparameters using random forest regression."""
    # Prepare data
    data = []
    
    # Map for display names - updated to focus on core parameters
    display_names = {
        'sequence_length': 'Sequence Length',
        'hidden_size': 'Hidden Size',
        'num_layers': 'Number of Layers',
        'dropout': 'Dropout Rate',
        'learning_rate': 'Learning Rate',
        'batch_size': 'Batch Size',
        'epochs': 'Training Epochs'
    }
    
    # Extract features and target
    for result in results:
        if 'params' not in result or 'value' not in result:
            continue
            
        params = result['params']
        value = result['value']
        
        # Extract numerical hyperparameters
        features = {}
        
        # Handle scalar parameters
        for param in ['sequence_length', 'hidden_size', 'num_layers', 'dropout', 
                     'learning_rate', 'batch_size', 'epochs']:
            if param in params:
                try:
                    features[param] = float(params[param])
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
        
        # Only add if we have features
        if features:
            features['score'] = value
            data.append(features)
    
    # Check if we have enough data
    if not data or len(data) < 5:
        print("Not enough data for hyperparameter importance analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Separate features and target
    X = df.drop('score', axis=1)
    y = df['score']
    
    # Check if we have enough features
    if X.shape[1] < 2:
        print(f"Not enough features for importance analysis: {X.shape[1]} features")
        return
    
    try:
        # Train a Random Forest Regressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from scipy import stats
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Sort feature importances in descending order
        sorted_idx = importances.argsort()[::-1]
        
        # Create a DataFrame for visualization
        feature_importances = pd.DataFrame(
            {'Feature': X.columns[sorted_idx], 'Importance': importances[sorted_idx]}
        )
        
        # Add statistical significance analysis
        print("\nStatistical Significance Analysis:")
        print("-" * 50)
        
        # Calculate mean and std of scores for each parameter value
        significance_results = {}
        for param in X.columns:
            unique_values = sorted(X[param].unique())
            if len(unique_values) >= 2:  # Need at least 2 values to compare
                # Group scores by parameter value
                value_groups = [y[X[param] == val] for val in unique_values]
                
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*value_groups)
                
                # Calculate effect size (Eta-squared)
                groups_mean = [group.mean() for group in value_groups]
                groups_var = [group.var() for group in value_groups]
                eta_squared = (np.var(groups_mean) * len(value_groups)) / (np.var(groups_mean) * len(value_groups) + np.mean(groups_var))
                
                significance_results[param] = {
                    'p_value': p_value,
                    'f_statistic': f_stat,
                    'effect_size': eta_squared,
                    'mean_range': max(groups_mean) - min(groups_mean)
                }
                
                print(f"\nParameter: {param}")
                print(f"  F-statistic: {f_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Effect size (η²): {eta_squared:.4f}")
                print(f"  Range of means: {significance_results[param]['mean_range']:.4f}")
                
                # Post-hoc analysis if significant
                if p_value < 0.05:
                    # Perform Tukey's HSD test
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey = pairwise_tukeyhsd(y, X[param])
                    print("\n  Tukey's HSD test results:")
                    print(tukey)
        
        # Create significance analysis plot
        plt.figure(figsize=(12, 6))
        significance_df = pd.DataFrame(significance_results).T
        
        # Plot p-values and effect sizes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot bars for effect size
        bars = ax1.bar(np.arange(len(significance_results)), 
                      [d['effect_size'] for d in significance_results.values()],
                      alpha=0.3, color='blue', label='Effect Size (η²)')
        
        # Plot line for p-values
        line = ax2.plot(np.arange(len(significance_results)), 
                       [d['p_value'] for d in significance_results.values()],
                       'r-o', label='p-value')
        
        # Add significance threshold line
        ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Significance Threshold (p=0.05)')
        
        # Customize plot
        ax1.set_xticks(np.arange(len(significance_results)))
        ax1.set_xticklabels(significance_results.keys(), rotation=45)
        ax1.set_ylabel('Effect Size (η²)', color='blue')
        ax2.set_ylabel('p-value', color='red')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Statistical Significance Analysis of Hyperparameters')
        plt.tight_layout()
        plt.savefig(output_dir / 'hyperparameter_significance.png', dpi=300)
        plt.close()
        
        # Map feature names to display names
        feature_importances['Display'] = feature_importances['Feature'].map(
            lambda x: display_names.get(x, x)
        )
        
        # Plot feature importances
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importances['Display'], feature_importances['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('')
        plt.title('Hyperparameter Importance Analysis')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / 'hyperparameter_importance.png', dpi=300)
        plt.close()
        
        # Create interactive Plotly version
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=feature_importances['Display'],
            x=feature_importances['Importance'],
            orientation='h',
            marker=dict(color='royalblue')
        ))
        
        fig.update_layout(
            title='Hyperparameter Importance Analysis',
            xaxis_title='Importance',
            yaxis_title='',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        fig.write_html(output_dir / 'hyperparameter_importance.html')
        
        # Save feature importances to CSV
        feature_importances.to_csv(output_dir / 'hyperparameter_importance.csv', index=False)
        
        print(f"Hyperparameter importance analysis completed")
        
    except Exception as e:
        print(f"Error in hyperparameter importance analysis: {e}")
        import traceback
        traceback.print_exc()

def create_3d_surface_plots(results, output_dir):
    """Create 3D surface plots for the most important parameter pairs."""
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract data from results
    data = []
    for result in results:
        if ('config' in result and 'score' in result and not np.isinf(result['score'])):
            row = {k: v for k, v in result['config'].items() 
                  if not isinstance(v, list)}  # Skip list parameters
            row['score'] = result['score']
            data.append(row)
        elif ('params' in result and 'value' in result and not np.isinf(result['value'])):
            row = {k: v for k, v in result['params'].items() 
                  if not isinstance(v, list)}  # Skip list parameters
            row['score'] = result['value']
            data.append(row)
    
    # Check if we have enough data
    if not data or len(data) < 4:  # Need at least 4 points for a meaningful surface
        print("Not enough data for 3D surface plot (need at least 4 trials)")
        
        # Create a simple placeholder image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                "Not enough data for 3D surface plot.\n"
                "Need at least 4 trials with varying parameters.",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "3d_surface_insufficient.png", dpi=300)
        plt.close()
        return
    
    df = pd.DataFrame(data)
    
    # Find parameters with most variance
    numeric_cols = [col for col in df.columns if col != 'score' and pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        print("Not enough numeric parameters for 3D surface plot")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                "Not enough numeric parameters for 3D surface plot.\n"
                "Need at least 2 numeric parameters.",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "3d_surface_insufficient.png", dpi=300)
        plt.close()
        return
    
    param_vars = {col: df[col].var() for col in numeric_cols}
    top_params = sorted(param_vars.items(), key=lambda x: x[1], reverse=True)[:2]
    param1, param2 = top_params[0][0], top_params[1][0]
    
    # Format parameter names for display
    param1_display = param1.replace('_', ' ').title()
    param2_display = param2.replace('_', ' ').title()
    
    # Create grid for surface plot
    x = df[param1].values
    y = df[param2].values
    z = df['score'].values
    
    # Create a grid to interpolate on
    xi = np.linspace(min(x), max(x), min(50, len(x)))  # Limit grid size
    yi = np.linspace(min(y), max(y), min(50, len(y)))  # Limit grid size
    xi, yi = np.meshgrid(xi, yi)
    
    try:
        # Interpolate
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        
        # Create 3D plot with enhanced styling
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface with enhanced appearance
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Plot the actual data points with better visibility
        sc = ax.scatter(x, y, z, c=z, cmap='plasma', s=70, alpha=1.0, edgecolor='black')
        
        # Find the minimum point safely
        try:
            # Find minimum in the interpolated surface
            min_idx = np.nanargmin(zi)
            min_point_i, min_point_j = np.unravel_index(min_idx, zi.shape)
            min_x, min_y = xi[min_point_i, min_point_j], yi[min_point_i, min_point_j]
            min_z = zi[min_point_i, min_point_j]
            
            # Highlight the minimum point
            ax.scatter([min_x], [min_y], [min_z], color='red', s=200, edgecolor='black', 
                      marker='*', label='Minimum')
            
            # Add text annotation for minimum
            ax.text(min_x, min_y, min_z, 
                   f"Min: ({min_x:.2f}, {min_y:.2f})\nScore: {min_z:.4f}",
                   color='red', fontsize=12)
        except Exception as e:
            print(f"Could not highlight minimum point: {e}")
        
        # Enhance the plot with better labels and styling
        ax.set_xlabel(param1_display, fontsize=14, labelpad=10)
        ax.set_ylabel(param2_display, fontsize=14, labelpad=10)
        ax.set_zlabel('Score', fontsize=14, labelpad=10)
        ax.set_title(f'Parameter Surface: {param1_display} vs {param2_display}', fontsize=16, pad=20)
        
        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=20, pad=0.1)
        cbar.set_label('Score', fontsize=12)
        
        # Rotate the plot for better visibility
        ax.view_init(elev=30, azim=45)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_dir / f"3d_surface_{param1}_{param2}.png", dpi=300)
        
        # Create an alternative view
        ax.view_init(elev=20, azim=135)
        plt.savefig(output_dir / f"3d_surface_{param1}_{param2}_alt_view.png", dpi=300)
        plt.close()
        
        # Create a 2D contour plot as well for better interpretability
        plt.figure(figsize=(12, 10))
        
        # Create contour plot
        contour = plt.contourf(xi, yi, zi, 15, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, label='Score')
        
        # Add data points
        plt.scatter(x, y, c=z, cmap='plasma', s=100, edgecolor='black')
        
        # Add minimum point
        try:
            plt.scatter([min_x], [min_y], color='red', s=200, edgecolor='black', 
                       marker='*', label='Minimum')
            plt.annotate(f"Min: ({min_x:.2f}, {min_y:.2f})\nScore: {min_z:.4f}",
                        xy=(min_x, min_y), xytext=(30, 30),
                        textcoords='offset points', color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
        except:
            pass
        
        plt.xlabel(param1_display, fontsize=14)
        plt.ylabel(param2_display, fontsize=14)
        plt.title(f'Contour Plot: {param1_display} vs {param2_display}', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"contour_{param1}_{param2}.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating 3D surface plot: {e}")
        # Create a simple error image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                f"Error creating 3D surface plot:\n{str(e)}",
                ha='center', va='center', fontsize=14, color='red')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "3d_surface_error.png", dpi=300)
        plt.close()

def plot_convergence_analysis(results, output_dir):
    """Analyze and visualize convergence rates of different hyperparameter configurations."""
    # Group results by key parameters
    convergence_data = {}
    
    for result in results:
        if 'training_results' not in result or 'history' not in result['training_results']:
            continue
            
        history = result['training_results']['history']
        if 'val_loss' not in history or len(history['val_loss']) < 2:
            continue
            
        # Create a key based on important parameters
        config = result['config']
        key_params = ['sequence_length', 'batch_size', 'learning_rate']
        key_values = {p: config.get(p, 'NA') for p in key_params if p in config}
        
        # Create a more concise key
        key = ", ".join([
            f"{p.replace('_', ' ').title()}={v:.1e}" if p == 'learning_rate' 
            else f"{p.replace('_', ' ').title()}={v}" 
            for p, v in key_values.items()
        ])
        
        # Store convergence data with score for sorting
        convergence_data[key] = {
            'val_loss': history['val_loss'],
            'train_loss': history.get('loss', history['val_loss']),  # Fallback if no train loss
            'epochs': len(history['val_loss']),
            'final_loss': history['val_loss'][-1],
            'score': result.get('score', float('inf')),
            'config': config
        }
    
    if not convergence_data:
        print("No convergence data available")
        return
        
    # Sort configurations by score and limit to top 6 for clarity
    sorted_configs = sorted(convergence_data.items(), key=lambda x: x[1]['score'])[:6]
    
    # Create plot with enhanced styling
    plt.figure(figsize=(14, 10))
    
    # Define a good color palette 
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_configs)))
    markers = ['o', 's', '^', 'd', 'v', 'p']
    
    # Plot both training and validation loss with markers only at intervals for clarity
    for idx, (key, data) in enumerate(sorted_configs):
        x_vals = range(1, data['epochs'] + 1)
        marker_interval = max(1, data['epochs'] // 10)  # Show 10 markers at most
        
        # Plot validation loss with line and markers
        plt.plot(x_vals, data['val_loss'], 
                marker=markers[idx % len(markers)], 
                markersize=8, 
                markevery=marker_interval,
                markeredgecolor='black', 
                markeredgewidth=1,
                color=colors[idx], 
                linewidth=2.5,
                label=f"{key} (Final: {data['final_loss']:.6f})")
        
        # Plot training loss with thinner line and no markers
        plt.plot(x_vals, data['train_loss'], 
                color=colors[idx], 
                linewidth=1,
                linestyle='--',
                alpha=0.6)
    
    # Style the plot
    plt.title('Convergence Analysis of Top Model Configurations', fontsize=18, pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value (log scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')  # Log scale often better visualizes convergence
    
    # Customize grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations to explain the lines
    plt.annotate('Solid: Validation Loss\nDashed: Training Loss', 
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Create a custom legend OUTSIDE the plot area
    plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              framealpha=0.95, frameon=True, fancybox=True, shadow=True, ncol=2)
    
    # Add minor grid lines for log scale
    plt.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    # Add more bottom margin to accommodate the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend below the plot
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300)
    plt.close()
    
    # Create a second plot showing convergence speed
    plt.figure(figsize=(12, 8))
    
    # Calculate relative improvement per epoch
    for idx, (key, data) in enumerate(sorted_configs):
        if len(data['val_loss']) > 5:  # Only for configs with enough epochs
            # Calculate improvement percentage relative to initial loss
            initial_loss = data['val_loss'][0]
            improvements = [(initial_loss - loss) / initial_loss * 100 for loss in data['val_loss']]
            
            plt.plot(range(1, len(improvements) + 1), improvements,
                    marker=markers[idx % len(markers)], 
                    markersize=8,
                    markevery=marker_interval,
                    color=colors[idx],
                    linewidth=2.5,
                    label=key)
    
    plt.title('Convergence Speed: Relative Improvement by Epoch', fontsize=18, pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Improvement (% of Initial Loss)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Move this legend outside as well
    plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              framealpha=0.95, frameon=True, ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend below
    plt.savefig(output_dir / 'convergence_speed.png', dpi=300)
    plt.close() 

def plot_learning_curve_clusters(results: List[Dict], output_dir: Path, n_clusters: int = 4):
    """
    Group similar learning curves to identify hyperparameter patterns.
    
    Args:
        results: List of hyperparameter tuning results
        output_dir: Directory to save the plot
        n_clusters: Number of clusters to form
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Extract learning curves
    learning_curves = []
    configs = []
    
    for result in results:
        if 'training_results' in result and 'history' in result['training_results']:
            if 'val_loss' in result['training_results']['history']:
                curve = result['training_results']['history']['val_loss']
                # Normalize and pad curves to same length
                max_len = 50  # Maximum epochs to consider
                padded_curve = np.zeros(max_len)
                padded_curve[:min(len(curve), max_len)] = curve[:min(len(curve), max_len)]
                learning_curves.append(padded_curve)
                configs.append(result['config'])
    
    if len(learning_curves) < n_clusters:
        print(f"Not enough learning curves for clustering (need at least {n_clusters})")
        return
        
    try:
        # Standardize curves
        scaler = StandardScaler()
        scaled_curves = scaler.fit_transform(learning_curves)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_curves)
        
        # Create color palette
        colors = plt.cm.viridis(np.linspace(0, 0.8, n_clusters))
        
        # Plot clusters
        plt.figure(figsize=(15, 10))
        
        for cluster_id in range(n_clusters):
            plt.subplot(2, n_clusters//2 + n_clusters%2, cluster_id+1)
            
            # Get curves in this cluster
            cluster_indices = [i for i in range(len(clusters)) if clusters[i] == cluster_id]
            cluster_curves = [learning_curves[i] for i in cluster_indices]
            
            # Plot individual curves with transparency
            for curve in cluster_curves:
                plt.plot(curve, alpha=0.2, color=colors[cluster_id])
            
            # Plot centroid with thicker line
            centroid = np.mean(cluster_curves, axis=0)
            plt.plot(centroid, linewidth=2.5, color=colors[cluster_id], 
                   label=f'Centroid (n={len(cluster_curves)})')
            
            # Add horizontal grid lines
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Analyze key parameters for this cluster
            cluster_configs = [configs[i] for i in cluster_indices]
            param_stats = {}
            key_params = ['batch_size', 'learning_rate', 'sequence_length', 'dropout_rate', 'epochs']
            
            for param in key_params:
                values = [c.get(param, None) for c in cluster_configs if param in c]
                if values:
                    if len(set(values)) <= 5:  # Categorical or few values
                        value_counts = {}
                        for v in values:
                            value_counts[v] = value_counts.get(v, 0) + 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])
                        param_stats[param] = f"{param.replace('_', ' ').title()}: {most_common[0]} ({most_common[1]}/{len(values)})"
                    else:  # Continuous parameter
                        param_stats[param] = f"{param.replace('_', ' ').title()}: {np.mean(values):.3g}±{np.std(values):.2g}"
            
            # Get average score for this cluster
            scores = [results[i].get('score', float('inf')) for i in cluster_indices]
            avg_score = np.mean(scores)
            
            # Create title with parameter insights
            plt.title(f"Cluster {cluster_id+1}: {len(cluster_curves)} configs\n" + 
                     "\n".join(list(param_stats.values())[:3]) + f"\nAvg Score: {avg_score:.5f}",
                     fontsize=11)
            
            plt.ylabel('Loss', fontweight='bold')
            plt.xlabel('Epoch', fontweight='bold')
            
            # Format y-axis to be readable
            plt.ylim(0, min(1.0, max(centroid)*1.5))
            plt.legend(loc='upper right')
        
        plt.suptitle('Learning Curve Clusters', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / 'learning_curve_clusters.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating learning curve clusters: {e}")

def plot_parameter_sensitivity(results: List[Dict], output_dir: Path):
    """
    Calculate and visualize how sensitive model performance is to each parameter.
    
    Args:
        results: List of hyperparameter tuning results
        output_dir: Directory to save the plot
    """
    import scipy.stats as stats
    
    # Extract parameters and scores
    param_data = {}
    
    for result in results:
        if 'config' in result and 'score' in result and not np.isinf(result['score']):
            for param, value in result['config'].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if param not in param_data:
                        param_data[param] = {'values': [], 'scores': []}
                    param_data[param]['values'].append(value)
                    param_data[param]['scores'].append(result['score'])
    
    # Check if we have enough data
    if not param_data:
        print("Not enough data for sensitivity analysis")
        return
    
    try:
        # Calculate sensitivity (standard deviation of scores grouped by parameter value)
        sensitivities = {}
        confidence_intervals = {}
        
        for param, data in param_data.items():
            if len(set(data['values'])) >= 3:  # Need at least 3 different values
                # Convert to numpy arrays
                values = np.array(data['values'])
                scores = np.array(data['scores'])
                
                # Sort by values
                idx = np.argsort(values)
                sorted_values = values[idx]
                sorted_scores = scores[idx]
                
                # Calculate local sensitivity using rolling window
                window = min(5, len(sorted_values)//3 + 1)
                window = max(window, 2)  # Ensure window is at least 2
                
                # Calculate sensitivity for each window
                windows = [sorted_scores[i:i+window] for i in range(0, len(sorted_scores)-window+1, max(1, window//2))]
                if windows:
                    local_stds = np.array([np.std(w) for w in windows])
                    sensitivities[param] = np.mean(local_stds)
                    
                    # Bootstrap for confidence intervals
                    n_bootstrap = 1000
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(local_stds, size=len(local_stds), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    confidence_intervals[param] = (
                        np.percentile(bootstrap_means, 5),
                        np.percentile(bootstrap_means, 95)
                    )
        
        if not sensitivities:
            print("Not enough varied parameters for sensitivity analysis")
            return
            
        # Plot sensitivities
        plt.figure(figsize=(12, 7))
        params = list(sensitivities.keys())
        sens_values = [sensitivities[p] for p in params]
        
        # Sort by sensitivity
        idx = np.argsort(sens_values)
        params = [params[i] for i in reversed(idx)]
        sens_values = [sens_values[i] for i in reversed(idx)]
        
        # Format parameter names for display
        display_names = [p.replace('_', ' ').title() for p in params]
        
        # Plot as horizontal bars with error bars
        y_pos = np.arange(len(params))
        bars = plt.barh(y_pos, sens_values, height=0.6, alpha=0.7,
                      color=plt.cm.viridis(np.linspace(0.1, 0.8, len(params))))
        
        # Add error bars if available
        for i, param in enumerate(params):
            if param in confidence_intervals:
                ci_low, ci_high = confidence_intervals[param]
                plt.errorbar(sensitivities[param], i, xerr=[[sensitivities[param]-ci_low], [ci_high-sensitivities[param]]], 
                           fmt='none', ecolor='black', capsize=5)
        
        plt.yticks(y_pos, display_names, fontsize=12)
        plt.xlabel('Sensitivity (Higher = More Impact on Performance Variance)', 
                 fontsize=12, fontweight='bold')
        plt.title('Hyperparameter Sensitivity Analysis', fontsize=16, pad=20)
        
        # Add grid for readability
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add annotations with values
        max_sens = max(sens_values)
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + max_sens*0.02, bar.get_y() + bar.get_height()/2, 
                    f"{sens_values[i]:.4f}", va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add explanatory note
        plt.figtext(0.01, 0.01, 
                   "Higher values indicate parameters that cause more performance variability when changed.",
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating sensitivity analysis: {e}")

def plot_top_n_distributions(results: List[Dict], output_dir: Path, top_n: int = 10):
    """
    Plot distributions of parameters in top-performing models compared to all models.
    
    Args:
        results: List of hyperparameter tuning results
        output_dir: Directory to save the plot
        top_n: Number of top models to include in the analysis
    """
    if len(results) < top_n:
        top_n = max(1, len(results) // 2)
        print(f"Not enough results for requested top_n, using {top_n} instead")
        
    # Filter out any results with invalid scores
    valid_results = [r for r in results if 'score' in r and not np.isinf(r['score'])]
    
    if len(valid_results) < 2:
        print("Not enough valid results for distribution analysis")
        return
        
    try:
        # Sort by score (assuming lower is better)
        sorted_results = sorted(valid_results, key=lambda x: x.get('score', float('inf')))
        top_results = sorted_results[:top_n]
        
        # Extract parameters to analyze
        params_to_plot = []
        for result in top_results:
            if 'config' in result:
                for param, value in result['config'].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if param not in params_to_plot:
                            params_to_plot.append(param)
        
        if not params_to_plot:
            print("No numeric parameters found in top results")
            return
            
        # Create a grid of violin plots
        n_params = len(params_to_plot)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        plt.figure(figsize=(4*n_cols, 3*n_rows))
        
        for i, param in enumerate(params_to_plot):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Extract values from top models vs all models
            top_values = [r['config'][param] for r in top_results if param in r['config']]
            all_values = [r['config'][param] for r in valid_results if 'config' in r and param in r['config']]
            
            # Skip if not enough data
            if len(top_values) < 2 or len(all_values) < 3:
                plt.text(0.5, 0.5, f"Insufficient data\nfor {param}", 
                       ha='center', va='center', transform=plt.gca().transAxes)
                continue
            
            # Create violin plots with custom styling
            positions = [1, 2]
            violin_parts = plt.violinplot([all_values, top_values], positions, 
                                        showmeans=True, showmedians=True, showextrema=True)
            
            # Style the violin plots
            for pc in violin_parts['bodies']:
                pc.set_alpha(0.7)
                pc.set_facecolor(plt.cm.viridis(0.6))
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Style the lines
            for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if partname in violin_parts:
                    violin_parts[partname].set_edgecolor('black')
                    violin_parts[partname].set_linewidth(1.5)
            
            # Add individual data points for top models
            plt.scatter([2] * len(top_values), top_values, color='red', alpha=0.6, s=30, zorder=3)
            
            # Add labels
            plt.xticks(positions, ['All Models', f'Top {top_n}'], fontsize=11)
            plt.title(param.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            
            # Add key statistics
            plt.text(0.97, 0.03, 
                   f"All: μ={np.mean(all_values):.3g}, σ={np.std(all_values):.3g}\n"
                   f"Top: μ={np.mean(top_values):.3g}, σ={np.std(top_values):.3g}",
                   ha='right', va='bottom', transform=plt.gca().transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   fontsize=9)
            
            # Add grid for readability
            plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.suptitle('Parameter Distributions: All Models vs. Top Performers', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / 'top_parameter_distributions.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating distribution analysis: {e}")

def generate_architecture_impact_plot(trials_data, output_dir):
    """
    Analyze the impact of architectural features on model performance.
    """
    # Create directory if it doesn't exist
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we have enough data
    if len(trials_data) < 5:
        print("Not enough data for architecture impact analysis")
        return
    
    # Extract architectural features from trials
    features = {
        'num_layers': [],
        'hidden_size': [],
        'dropout': [],
        'sequence_length': []
    }
    
    scores = []
    
    for trial in trials_data:
        # Get architecture features
        params = trial.get('params', {})
        
        # Skip trials without params or value
        if not params or 'value' not in trial:
            continue
            
        # Extract features
        for feature in features.keys():
            if feature in params:
                try:
                    features[feature].append(float(params[feature]))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    features[feature].append(None)
            else:
                features[feature].append(None)
        
        # Get score (value)
        scores.append(trial['value'])
    
    # Check if we have enough data after filtering
    if len(scores) < 3:
        print("Not enough valid data for architecture impact analysis")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot impact of each feature
    for i, (feature, values) in enumerate(features.items()):
        row, col = i // 2, i % 2
        ax = axs[row, col]
        
        # Filter out None values
        filtered_data = [(v, s) for v, s in zip(values, scores) if v is not None]
        if not filtered_data:
            ax.text(0.5, 0.5, f"No data for {feature}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
            
        feature_values, feature_scores = zip(*filtered_data)
        
        # For categorical features
        if feature == 'num_layers':
            # Convert to categorical for boxplot
            categories = sorted(set(feature_values))
            
            # Create boxplot data
            boxplot_data = []
            for category in categories:
                category_scores = [score for val, score in zip(feature_values, feature_scores) if val == category]
                boxplot_data.append(category_scores)
            
            # Create boxplot
            ax.boxplot(boxplot_data, labels=[str(int(c)) for c in categories])
            ax.set_title(f'Impact of {feature.replace("_", " ").title()}')
            ax.set_ylabel('Loss Value')
            
        # For continuous features
        else:
            # Create scatter plot
            ax.scatter(feature_values, feature_scores, alpha=0.7)
            
            # Try to fit a trend line
            try:
                z = np.polyfit(feature_values, feature_scores, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(feature_values), max(feature_values), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.7)
            except:
                pass
                
            ax.set_title(f'Impact of {feature.replace("_", " ").title()}')
            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel('Loss Value')
            
            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Architecture Feature Impact Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_dir / "architecture_impact.png", dpi=300)
    
    # Create interactive version with Plotly
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=[f.replace("_", " ").title() for f in features.keys()])
    
    for i, (feature, values) in enumerate(features.items()):
        row, col = i // 2 + 1, i % 2 + 1
        
        # Filter out None values
        filtered_data = [(v, s) for v, s in zip(values, scores) if v is not None]
        if not filtered_data:
            continue
            
        feature_values, feature_scores = zip(*filtered_data)
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=feature_values, 
                y=feature_scores,
                mode='markers',
                marker=dict(
                    size=10,
                    color=feature_values,
                    colorscale='Viridis',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title="Value") if i == 0 else None
                ),
                name=feature
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Architecture Feature Impact Analysis",
        height=800,
        showlegend=False
    )
    
    fig.write_html(plot_dir / "architecture_impact.html")
    plt.close()

def plot_learning_rate_landscape(results: List[Dict], output_dir: Path):
    """Create a visualization of the learning rate landscape and its impact on training."""
    # Extract learning rate data
    lr_data = []
    for result in results:
        if ('config' in result and 'score' in result and 
            'training_results' in result and 'history' in result['training_results']):
            lr = result['config'].get('learning_rate')
            history = result['training_results']['history']
            if lr and 'val_loss' in history:
                lr_data.append({
                    'lr': lr,
                    'final_loss': history['val_loss'][-1],
                    'convergence_speed': len(history['val_loss']),
                    'loss_curve': history['val_loss']
                })
    
    if not lr_data:
        return
        
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1.5])
    
    # Plot 1: Learning Rate vs Final Loss
    lr_df = pd.DataFrame(lr_data)
    
    # Sort by learning rate for better visualization
    lr_df = lr_df.sort_values('lr')
    
    # Plot with consistent styling
    ax1.scatter(lr_df['lr'], lr_df['final_loss'], 
               s=100, color='royalblue', alpha=0.7,
               edgecolor='black', linewidth=1)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Rate Impact on Model Performance', 
                  fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add minor grid lines
    ax1.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    # Plot 2: Loss Curves for Different Learning Rates
    colors = plt.cm.viridis(np.linspace(0, 1, len(lr_data)))
    markers = ['o', 's', '^', 'd', 'v', 'p']
    
    for idx, data in enumerate(lr_data):
        epochs = range(1, len(data['loss_curve']) + 1)
        ax2.plot(epochs, data['loss_curve'],
                marker=markers[idx % len(markers)],
                markersize=6,
                markevery=max(1, len(epochs)//10),
                color=colors[idx],
                linewidth=2,
                label=f'LR: {data["lr"]:.1e}')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Patterns by Learning Rate', 
                  fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    # Create a custom legend with fewer columns and better placement
    ax2.legend(fontsize=10, 
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=3,
              framealpha=0.95,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    plt.savefig(output_dir / 'learning_rate_landscape.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_network_architecture_performance(results: List[Dict], output_dir: Path):
    """Visualize performance across different network architectures."""
    # Extract architecture data
    arch_data = []
    for result in results:
        # Try different data formats
        if 'params' in result and 'value' in result:
            params = result['params']
            score = result['value']
            
            # Extract architecture parameters
            arch_data.append({
                'n_layers': params.get('num_layers', 1),
                'hidden_size': params.get('hidden_size', 0),
                'total_params': params.get('hidden_size', 0) * params.get('num_layers', 1),
                'layer_sizes': f"{params.get('hidden_size', 0)} x {params.get('num_layers', 1)}",
                'score': score
            })
        elif 'config' in result and 'score' in result:
            config = result['config']
            if 'lstm_units' in config:
                arch_data.append({
                    'n_layers': len(config['lstm_units']),
                    'total_params': sum(config['lstm_units']),
                    'layer_sizes': ' → '.join(map(str, config['lstm_units'])),
                    'score': result['score']
                })
    
    if not arch_data:
        print("No architecture data available")
        # Create a placeholder image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                "No architecture data available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "network_architecture_performance.png", dpi=300)
        plt.close()
        return
        
    df = pd.DataFrame(arch_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Number of layers vs Performance
    sns.boxplot(data=df, x='n_layers', y='score', ax=ax1)
    ax1.set_xlabel('Number of LSTM Layers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Model Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance by Network Depth', fontsize=14)
    
    # Plot 2: Model Size vs Performance
    scatter = ax2.scatter(df['total_params'], df['score'], 
                         c=df['n_layers'], cmap='viridis', 
                         s=100, alpha=0.6)
    ax2.set_xlabel('Total LSTM Units', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Model Score', fontsize=12, fontweight='bold')
    ax2.set_title('Performance by Network Size', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Number of Layers', fontsize=10)
    
    # Add annotations for best architectures
    top_3 = df.nsmallest(3, 'score')
    for _, row in top_3.iterrows():
        ax2.annotate(f"Architecture: {row['layer_sizes']}", 
                    xy=(row['total_params'], row['score']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->'))
    
    plt.suptitle('Network Architecture Performance Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'network_architecture_performance.png', dpi=300)
    plt.close()
    
    # Create interactive version with Plotly
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=['Performance by Network Depth', 
                                      'Performance by Network Size'])
    
    # Add boxplot for network depth
    for layer in sorted(df['n_layers'].unique()):
        layer_scores = df[df['n_layers'] == layer]['score']
        fig.add_trace(
            go.Box(y=layer_scores, name=f"{layer} Layers"),
            row=1, col=1
        )
    
    # Add scatter plot for network size
    fig.add_trace(
        go.Scatter(
            x=df['total_params'],
            y=df['score'],
            mode='markers',
            marker=dict(
                size=12,
                color=df['n_layers'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Number of Layers")
            ),
            text=df['layer_sizes'],
            hovertemplate="<b>Architecture:</b> %{text}<br>" +
                         "<b>Total Units:</b> %{x}<br>" +
                         "<b>Score:</b> %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Network Architecture Performance Analysis",
        height=600,
        showlegend=False
    )
    
    fig.write_html(output_dir / "network_architecture_performance.html")

def plot_training_dynamics(results: List[Dict], output_dir: Path):
    """Visualize training dynamics and stability across different configurations."""
    # Extract training dynamics data
    dynamics_data = []
    for result in results:
        if ('config' in result and 'training_results' in result and 
            'history' in result['training_results']):
            config = result['config']
            history = result['training_results']['history']
            if 'val_loss' in history:
                # Calculate training stability metrics
                val_losses = history['val_loss']
                loss_volatility = np.std(np.diff(val_losses)) if len(val_losses) > 1 else 0
                
                dynamics_data.append({
                    'batch_size': config.get('batch_size', 0),
                    'sequence_length': config.get('sequence_length', 0),
                    'loss_volatility': loss_volatility,
                    'final_loss': val_losses[-1],
                    'convergence_epoch': len(val_losses)
                })
    
    if not dynamics_data:
        print("No training dynamics data available")
        # Create a placeholder image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                "No training dynamics data available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "training_dynamics.png", dpi=300)
        plt.close()
        return
        
    df = pd.DataFrame(dynamics_data)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Batch Size vs Loss Volatility
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df, x='batch_size', y='loss_volatility', 
                   size='final_loss', sizes=(50, 400), alpha=0.6, ax=ax1)
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss Volatility', fontsize=12, fontweight='bold')
    ax1.set_title('Training Stability by Batch Size', fontsize=14)
    
    # Plot 2: Sequence Length vs Loss Volatility
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(data=df, x='sequence_length', y='loss_volatility',
                   size='final_loss', sizes=(50, 400), alpha=0.6, ax=ax2)
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss Volatility', fontsize=12, fontweight='bold')
    ax2.set_title('Training Stability by Sequence Length', fontsize=14)
    
    # Plot 3: Heatmap of Training Stability
    ax3 = fig.add_subplot(gs[1, :])
    
    try:
        # Try to create pivot table with qcut, handling duplicate values
        pivot_data = df.pivot_table(
            values='loss_volatility', 
            index=pd.qcut(df['batch_size'], min(4, len(df['batch_size'].unique())), duplicates='drop'), 
            columns=pd.qcut(df['sequence_length'], min(4, len(df['sequence_length'].unique())), duplicates='drop'),
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=ax3)
        ax3.set_title('Training Stability Heatmap (Batch Size vs Sequence Length)', fontsize=14)
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # Create a simpler alternative plot
        ax3.text(0.5, 0.5, 
                f"Could not create heatmap: {str(e)}",
                ha='center', va='center', transform=ax3.transAxes)
    
    plt.suptitle('Training Dynamics Analysis', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "training_dynamics.png", dpi=300)
    plt.close()
    
    # Create interactive version with Plotly
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=['Training Stability by Batch Size', 
                                      'Training Stability by Sequence Length'])
    
    # Add scatter plots
    fig.add_trace(
        go.Scatter(
            x=df['batch_size'],
            y=df['loss_volatility'],
            mode='markers',
            marker=dict(
                size=df['final_loss'] * 100,  # Scale for visibility
                color=df['final_loss'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Final Loss")
            ),
            name='Batch Size'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['sequence_length'],
            y=df['loss_volatility'],
            mode='markers',
            marker=dict(
                size=df['final_loss'] * 100,  # Scale for visibility
                color=df['final_loss'],
                colorscale='Viridis',
                showscale=False
            ),
            name='Sequence Length'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Training Dynamics Analysis",
        height=600,
        showlegend=False
    )
    
    fig.write_html(output_dir / "training_dynamics.html")

def plot_training_time_analysis(results: List[Dict], output_dir: Path):
    """Analyze how different parameters affect training time."""
    time_data = []
    for result in results:
        # Try different data formats
        if 'params' in result and 'metrics' in result and 'training_time' in result['metrics']:
            params = result['params']
            time_data.append({
                'batch_size': params.get('batch_size', 0),
                'sequence_length': params.get('sequence_length', 0),
                'n_layers': params.get('num_layers', 1),
                'hidden_size': params.get('hidden_size', 0),
                'total_units': params.get('hidden_size', 0) * params.get('num_layers', 1),
                'training_time': result['metrics']['training_time'],
                'score': result.get('value', float('inf'))
            })
        elif 'config' in result and 'trial_time' in result:
            config = result['config']
            time_data.append({
                'batch_size': config.get('batch_size', 0),
                'sequence_length': config.get('sequence_length', 0),
                'n_layers': len(config.get('lstm_units', [])),
                'total_units': sum(config.get('lstm_units', [])),
                'training_time': result['trial_time'],
                'score': result.get('score', float('inf'))
            })
        elif 'params' in result and 'duration' in result:
            params = result['params']
            time_data.append({
                'batch_size': params.get('batch_size', 0),
                'sequence_length': params.get('sequence_length', 0),
                'n_layers': params.get('num_layers', 1),
                'hidden_size': params.get('hidden_size', 0),
                'total_units': params.get('hidden_size', 0) * params.get('num_layers', 1),
                'training_time': result['duration'],
                'score': result.get('value', float('inf'))
            })
    
    if not time_data:
        print("No training time data available")
        # Create a placeholder image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                "No training time data available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "training_time_analysis.png", dpi=300)
        plt.close()
        return
        
    df = pd.DataFrame(time_data)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Plot 1: Model Size vs Training Time
    ax1 = fig.add_subplot(gs[0, 0])
    if 'total_units' in df.columns and df['total_units'].sum() > 0:
        scatter1 = ax1.scatter(df['total_units'], df['training_time'], 
                              c=df['score'], cmap='viridis', 
                              s=100, alpha=0.6)
        ax1.set_xlabel('Total Units', fontsize=12)
        ax1.set_ylabel('Training Time (s)', fontsize=12)
        plt.colorbar(scatter1, ax=ax1, label='Score')
    else:
        ax1.text(0.5, 0.5, "No model size data available", 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Sequence Length vs Training Time
    ax2 = fig.add_subplot(gs[0, 1])
    if 'sequence_length' in df.columns and df['sequence_length'].sum() > 0:
        scatter2 = ax2.scatter(df['sequence_length'], df['training_time'],
                              c=df['batch_size'], cmap='plasma',
                              s=100, alpha=0.6)
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('Training Time (s)', fontsize=12)
        plt.colorbar(scatter2, ax=ax2, label='Batch Size')
    else:
        ax2.text(0.5, 0.5, "No sequence length data available", 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Training Time Distribution by Layers
    ax3 = fig.add_subplot(gs[1, :])
    if 'n_layers' in df.columns and df['n_layers'].nunique() > 1:
        sns.boxplot(data=df, x='n_layers', y='training_time', ax=ax3)
        ax3.set_xlabel('Number of Layers', fontsize=12)
        ax3.set_ylabel('Training Time (s)', fontsize=12)
    else:
        # Alternative: Batch Size vs Training Time
        sns.boxplot(data=df, x='batch_size', y='training_time', ax=ax3)
        ax3.set_xlabel('Batch Size', fontsize=12)
        ax3.set_ylabel('Training Time (s)', fontsize=12)
    
    plt.suptitle('Training Time Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'training_time_analysis.png', dpi=300)
    plt.close()
    
    # Create interactive version with Plotly
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=['Model Size vs Training Time', 
                                      'Sequence Length vs Training Time',
                                      'Training Time by Number of Layers',
                                      'Training Time by Batch Size'])
    
    # Add scatter plot for model size
    if 'total_units' in df.columns and df['total_units'].sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=df['total_units'],
                y=df['training_time'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                name='Model Size'
            ),
            row=1, col=1
        )
    
    # Add scatter plot for sequence length
    if 'sequence_length' in df.columns and df['sequence_length'].sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=df['sequence_length'],
                y=df['training_time'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['batch_size'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Batch Size")
                ),
                name='Sequence Length'
            ),
            row=1, col=2
        )
    
    # Add boxplot for number of layers
    if 'n_layers' in df.columns and df['n_layers'].nunique() > 1:
        for layer in sorted(df['n_layers'].unique()):
            layer_times = df[df['n_layers'] == layer]['training_time']
            fig.add_trace(
                go.Box(y=layer_times, name=f"{layer} Layers"),
                row=2, col=1
            )
    
    # Add boxplot for batch size
    for batch in sorted(df['batch_size'].unique()):
        batch_times = df[df['batch_size'] == batch]['training_time']
        fig.add_trace(
            go.Box(y=batch_times, name=f"Batch {batch}"),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Training Time Analysis",
        height=800,
        showlegend=False
    )
    
    fig.write_html(output_dir / "training_time_analysis.html")

def plot_parameter_evolution(results: List[Dict], output_dir: Path):
    """Visualize how parameter choices evolved during the tuning process."""
    evolution_data = []
    for i, result in enumerate(results):
        if 'config' in result and 'score' in result:
            data = {
                'trial': i,
                'score': result['score'] if not np.isinf(result['score']) else None
            }
            data.update(result['config'])
            evolution_data.append(data)
    
    if not evolution_data:
        return
        
    df = pd.DataFrame(evolution_data)
    
    # Create multi-panel evolution plot
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # Plot 1: Parameter trajectories
    ax1 = fig.add_subplot(gs[0])
    for param in ['sequence_length', 'batch_size', 'learning_rate']:
        if param in df.columns:
            normalized_values = (df[param] - df[param].min()) / (df[param].max() - df[param].min())
            ax1.plot(df['trial'], normalized_values, 'o-', label=param, alpha=0.7)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Normalized Parameter Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Score evolution
    ax2 = fig.add_subplot(gs[1])
    valid_scores = df[df['score'].notna()]
    ax2.plot(valid_scores['trial'], valid_scores['score'], 'o-', color='green', alpha=0.7)
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Running best score
    ax3 = fig.add_subplot(gs[2])
    running_best = valid_scores['score'].cummin()
    ax3.plot(valid_scores['trial'], running_best, 'r-', label='Best Score', linewidth=2)
    ax3.fill_between(valid_scores['trial'], running_best, alpha=0.2, color='red')
    ax3.set_xlabel('Trial Number')
    ax3.set_ylabel('Best Score')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Evolution During Tuning', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_evolution.png', dpi=300, bbox_inches='tight')
    plt.close() 

def plot_parallel_coordinates_clustered(results: List[Dict], output_dir: Path):
    """Create a parallel coordinates plot with clustering to visualize parameter relationships."""
    # Extract parameter data
    param_data = []
    for result in results:
        if 'params' in result and 'value' in result:
            params = result['params']
            row = {}
            
            # Extract numerical parameters
            for param in ['sequence_length', 'hidden_size', 'num_layers', 'dropout', 
                         'learning_rate', 'batch_size']:
                if param in params:
                    try:
                        row[param] = float(params[param])
                    except (ValueError, TypeError):
                        continue
            
            # Add performance metric
            row['score'] = result['value']
            
            # Only add if we have enough parameters
            if len(row) >= 3:  # At least 2 parameters + score
                param_data.append(row)
    
    if not param_data or len(param_data) < 3:
        print("Not enough data for parallel coordinates plot")
        # Create a placeholder image
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                "Not enough data for parallel coordinates plot",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "parallel_coordinates_clustered.png", dpi=300)
        plt.close()
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(param_data)
    
    # Normalize data for better visualization
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    
    # Perform clustering
    from sklearn.cluster import KMeans
    
    # Determine optimal number of clusters (2-5)
    n_clusters = min(5, len(df) // 2)
    n_clusters = max(2, n_clusters)  # At least 2 clusters
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled.drop('score', axis=1))
    
    # Create parallel coordinates plot with clusters
    fig = go.Figure(data=[
        go.Parcoords(
            line=dict(color=clusters, 
                     colorscale='Viridis'),
            dimensions=[
                dict(range=[df[col].min(), df[col].max()],
                     label=col,
                     values=df[col])
                for col in df.columns
            ]
        )
    ])
    
    fig.update_layout(
        title="Clustered Parameter Configurations",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save both HTML and PNG versions
    fig.write_html(output_dir / 'parallel_coordinates_clustered.html')
    
    # Create a static version with matplotlib for PNG output
    plt.figure(figsize=(12, 8))
    
    # Create a parallel coordinates plot using pandas plotting
    pd.plotting.parallel_coordinates(
        df.assign(cluster=clusters), 
        'cluster',
        colormap='viridis',
        alpha=0.7
    )
    
    plt.title('Clustered Parameter Configurations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the PNG version
    plt.savefig(output_dir / 'parallel_coordinates_clustered.png', dpi=300)
    plt.close()

def plot_test_vs_validation_performance(results: List[Dict], output_dir: Path):
    """
    Create a scatter plot comparing validation loss vs test RMSE to identify models
    that generalize well to unseen data.
    
    Args:
        results: List of trial results
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    # Create plot directory
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Extract validation loss and test RMSE
    val_losses = []
    test_rmses = []
    sequence_lengths = []
    hidden_sizes = []
    trial_ids = []
    
    for result in results:
        if 'value' in result and result.get('test_metrics_avg', {}).get('test_rmse') is not None:
            val_losses.append(result['value'])
            test_rmses.append(result['test_metrics_avg']['test_rmse'])
            sequence_lengths.append(result['params'].get('sequence_length', 0))
            hidden_sizes.append(result['params'].get('hidden_size', 0))
            trial_ids.append(result.get('trial_id', 0))
    
    if not val_losses or not test_rmses:
        print("No validation loss or test RMSE data available for comparison plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color based on sequence length
    norm = Normalize(vmin=min(sequence_lengths), vmax=max(sequence_lengths))
    scatter = plt.scatter(val_losses, test_rmses, c=sequence_lengths, cmap='viridis', 
                         alpha=0.7, s=100, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sequence Length')
    
    # Add trial IDs as annotations
    for i, trial_id in enumerate(trial_ids):
        plt.annotate(f"Trial {trial_id}", 
                    (val_losses[i], test_rmses[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Find the best models
    best_val_idx = np.argmin(val_losses)
    best_test_idx = np.argmin(test_rmses)
    
    # Highlight the best models
    plt.scatter(val_losses[best_val_idx], test_rmses[best_val_idx], 
               s=200, facecolors='none', edgecolors='red', linewidth=2,
               label=f'Best Validation (Trial {trial_ids[best_val_idx]})')
    
    plt.scatter(val_losses[best_test_idx], test_rmses[best_test_idx], 
               s=200, facecolors='none', edgecolors='blue', linewidth=2,
               label=f'Best Test RMSE (Trial {trial_ids[best_test_idx]})')
    
    # Add labels and title
    plt.xlabel('Validation Loss')
    plt.ylabel('Test RMSE')
    plt.title('Validation Loss vs Test RMSE: Identifying Models that Generalize Well')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(plot_dir / 'validation_vs_test_performance.png', dpi=300)
    plt.close()
    
    # Create a second plot showing the relationship with hidden size
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color based on hidden size
    norm = Normalize(vmin=min(hidden_sizes), vmax=max(hidden_sizes))
    scatter = plt.scatter(val_losses, test_rmses, c=hidden_sizes, cmap='plasma', 
                         alpha=0.7, s=100, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hidden Size')
    
    # Add trial IDs as annotations
    for i, trial_id in enumerate(trial_ids):
        plt.annotate(f"Trial {trial_id}", 
                    (val_losses[i], test_rmses[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Highlight the best models
    plt.scatter(val_losses[best_val_idx], test_rmses[best_val_idx], 
               s=200, facecolors='none', edgecolors='red', linewidth=2,
               label=f'Best Validation (Trial {trial_ids[best_val_idx]})')
    
    plt.scatter(val_losses[best_test_idx], test_rmses[best_test_idx], 
               s=200, facecolors='none', edgecolors='blue', linewidth=2,
               label=f'Best Test RMSE (Trial {trial_ids[best_test_idx]})')
    
    # Add labels and title
    plt.xlabel('Validation Loss')
    plt.ylabel('Test RMSE')
    plt.title('Validation Loss vs Test RMSE: Effect of Hidden Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(plot_dir / 'validation_vs_test_performance_hidden_size.png', dpi=300)
    plt.close()
    
    # Create a table of the top 10 models by test RMSE
    top_indices = np.argsort(test_rmses)[:10]
    
    top_data = {
        'Trial ID': [trial_ids[i] for i in top_indices],
        'Test RMSE': [test_rmses[i] for i in top_indices],
        'Validation Loss': [val_losses[i] for i in top_indices],
        'Sequence Length': [sequence_lengths[i] for i in top_indices],
        'Hidden Size': [hidden_sizes[i] for i in top_indices]
    }
    
    top_df = pd.DataFrame(top_data)
    
    # Save to CSV
    top_df.to_csv(plot_dir / 'top_models_by_test_rmse.csv', index=False)
    
    # Print top models
    print("\nTop 10 Models by Test RMSE:")
    print(top_df)
    
    return top_df


def plot_sudden_change_performance(results: List[Dict], output_dir: Path):
    """
    Create visualizations to analyze how well models handle sudden changes in the data.
    
    Args:
        results: List of trial results
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Create plot directory
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Extract metrics
    data = []
    
    for result in results:
        test_metrics = result.get('test_metrics_avg', {})
        if test_metrics.get('sudden_change_mse') is not None:
            data.append({
                'trial_id': result.get('trial_id', 0),
                'sequence_length': result['params'].get('sequence_length', 0),
                'hidden_size': result['params'].get('hidden_size', 0),
                'num_layers': result['params'].get('num_layers', 0),
                'learning_rate': result['params'].get('learning_rate', 0),
                'test_rmse': test_metrics.get('test_rmse', float('inf')),
                'sudden_change_rmse': np.sqrt(test_metrics.get('sudden_change_mse', float('inf'))),
                'test_mae': test_metrics.get('test_mae', float('inf')),
                'sudden_change_mae': test_metrics.get('sudden_change_mae', float('inf'))
            })
    
    if not data:
        print("No sudden change performance data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate the ratio of sudden change error to overall error
    df['rmse_ratio'] = df['sudden_change_rmse'] / df['test_rmse']
    df['mae_ratio'] = df['sudden_change_mae'] / df['test_mae']
    
    # Create figure for RMSE comparison
    plt.figure(figsize=(12, 8))
    
    # Sort by test RMSE
    df_sorted = df.sort_values('test_rmse')
    
    # Plot bars for overall RMSE and sudden change RMSE
    x = np.arange(len(df_sorted))
    width = 0.35
    
    plt.bar(x - width/2, df_sorted['test_rmse'], width, label='Overall RMSE')
    plt.bar(x + width/2, df_sorted['sudden_change_rmse'], width, label='Sudden Change RMSE')
    
    # Add trial IDs as x-tick labels
    plt.xticks(x, [f"Trial {tid}" for tid in df_sorted['trial_id']], rotation=45)
    
    # Add labels and title
    plt.xlabel('Trial')
    plt.ylabel('RMSE')
    plt.title('Overall RMSE vs Sudden Change RMSE by Trial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(plot_dir / 'overall_vs_sudden_change_rmse.png', dpi=300)
    plt.close()
    
    # Create figure for ratio analysis
    plt.figure(figsize=(12, 8))
    
    # Sort by RMSE ratio
    df_sorted = df.sort_values('rmse_ratio')
    
    # Plot bars for RMSE ratio
    plt.bar(np.arange(len(df_sorted)), df_sorted['rmse_ratio'], color='skyblue')
    
    # Add horizontal line at ratio = 1
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
               label='Equal Performance (Ratio = 1)')
    
    # Add trial IDs as x-tick labels
    plt.xticks(np.arange(len(df_sorted)), 
              [f"Trial {tid}" for tid in df_sorted['trial_id']], 
              rotation=45)
    
    # Add labels and title
    plt.xlabel('Trial')
    plt.ylabel('Sudden Change RMSE / Overall RMSE')
    plt.title('Ratio of Sudden Change RMSE to Overall RMSE by Trial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(plot_dir / 'sudden_change_rmse_ratio.png', dpi=300)
    plt.close()
    
    # Create scatter plot of sequence length vs RMSE ratio
    plt.figure(figsize=(12, 8))
    
    plt.scatter(df['sequence_length'], df['rmse_ratio'], 
               c=df['hidden_size'], cmap='viridis', 
               alpha=0.7, s=100)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Hidden Size')
    
    # Add trial IDs as annotations
    for i, row in df.iterrows():
        plt.annotate(f"Trial {row['trial_id']}", 
                    (row['sequence_length'], row['rmse_ratio']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Add labels and title
    plt.xlabel('Sequence Length')
    plt.ylabel('Sudden Change RMSE / Overall RMSE')
    plt.title('Effect of Sequence Length on Handling Sudden Changes')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(plot_dir / 'sequence_length_vs_sudden_change_ratio.png', dpi=300)
    plt.close()
    
    # Create a table of the top 10 models by sudden change handling
    top_indices = np.argsort(df['rmse_ratio'].values)[:10]
    
    top_data = {
        'Trial ID': df.iloc[top_indices]['trial_id'].values,
        'Sudden Change Ratio': df.iloc[top_indices]['rmse_ratio'].values,
        'Overall RMSE': df.iloc[top_indices]['test_rmse'].values,
        'Sudden Change RMSE': df.iloc[top_indices]['sudden_change_rmse'].values,
        'Sequence Length': df.iloc[top_indices]['sequence_length'].values,
        'Hidden Size': df.iloc[top_indices]['hidden_size'].values,
        'Num Layers': df.iloc[top_indices]['num_layers'].values
    }
    
    top_df = pd.DataFrame(top_data)
    
    # Save to CSV
    top_df.to_csv(plot_dir / 'top_models_by_sudden_change_handling.csv', index=False)
    
    # Print top models
    print("\nTop 10 Models by Sudden Change Handling:")
    print(top_df)
    
    return top_df


def plot_test_predictions_comparison(results: List[Dict], output_dir: Path, num_models: int = 3):
    """
    Create a plot comparing the test predictions of the top models.
    
    Args:
        results: List of trial results
        output_dir: Output directory
        num_models: Number of top models to compare
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Create plot directory
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Extract test RMSE and predictions
    models_with_predictions = []
    
    for result in results:
        test_metrics = result.get('test_metrics_avg', {})
        if test_metrics.get('test_rmse') is not None:
            # Get the first window result with predictions
            for window_result in result.get('window_results', []):
                if window_result.get('test_predictions') is not None:
                    models_with_predictions.append({
                        'trial_id': result.get('trial_id', 0),
                        'test_rmse': test_metrics.get('test_rmse', float('inf')),
                        'sequence_length': result['params'].get('sequence_length', 0),
                        'hidden_size': result['params'].get('hidden_size', 0),
                        'num_layers': result['params'].get('num_layers', 0),
                        'predictions': window_result.get('test_predictions'),
                        'window_idx': window_result.get('window_idx', 0)
                    })
                    break
    
    if not models_with_predictions:
        print("No models with test predictions available")
        return
    
    # Sort by test RMSE
    models_with_predictions.sort(key=lambda x: x['test_rmse'])
    
    # Select top N models
    top_models = models_with_predictions[:num_models]
    
    # We need actual test data to compare against
    # This would typically come from the dataset, but for now we'll just use what we have
    # In a real implementation, you would load the actual test data here
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot predictions for each model
    for i, model in enumerate(top_models):
        predictions = model['predictions']
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # Plot a subset of predictions for clarity (e.g., 200 points)
        subset_size = min(200, len(predictions))
        x = np.arange(subset_size)
        
        plt.plot(x, predictions[:subset_size], 
                label=f"Trial {model['trial_id']} (RMSE: {model['test_rmse']:.6f}, SL: {model['sequence_length']})")
    
    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Predicted Value')
    plt.title('Comparison of Test Predictions from Top Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(plot_dir / 'top_models_predictions_comparison.png', dpi=300)
    plt.close()
    
    # Create a table of the top models
    top_data = {
        'Trial ID': [model['trial_id'] for model in top_models],
        'Test RMSE': [model['test_rmse'] for model in top_models],
        'Sequence Length': [model['sequence_length'] for model in top_models],
        'Hidden Size': [model['hidden_size'] for model in top_models],
        'Num Layers': [model['num_layers'] for model in top_models],
        'Window': [model['window_idx'] for model in top_models]
    }
    
    top_df = pd.DataFrame(top_data)
    
    # Save to CSV
    top_df.to_csv(plot_dir / 'top_models_with_predictions.csv', index=False)
    
    # Print top models
    print("\nTop Models with Predictions:")
    print(top_df)
    
    return top_df

if __name__ == "__main__":
    """
    Test the hyperparameter diagnostics functions with sample data.
    
    This allows running this file directly to verify that all visualization
    functions are working properly.
    
    Usage:
        python -m diagnostics.hyperparameter_diagnostics
    """
    import os
    import random
    from pathlib import Path
    
    print("Testing hyperparameter diagnostics functions...")
    
    # Create a test output directory
    output_dir = Path("./test_diagnostics_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Generate sample hyperparameter tuning results
    num_trials = 50  # Increased for better visualizations
    sample_results = []
    
    # Parameters to vary
    param_ranges = {
        'sequence_length': [24, 48, 72, 96, 120],
        'hidden_size': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64, 128],
        'epochs': [50, 100, 150]
    }
    
    # Generate sample results
    for i in range(num_trials):
        # Sample parameters
        config = {}
        for param, values in param_ranges.items():
            config[param] = random.choice(values)
        
        # Generate a score (lower is better)
        # Make it somewhat correlated with parameters to create realistic patterns
        score = 0.5
        score += 0.1 * (config['num_layers'] - 2) ** 2  # Optimal at 2 layers
        score += 0.001 * (config['hidden_size'] - 128) ** 2 / 128  # Optimal at 128
        score += 0.2 * (config['dropout'] - 0.2) ** 2  # Optimal at 0.2
        score += 0.1 * (config['learning_rate'] - 0.005) ** 2 / 0.005  # Optimal at 0.005
        
        # Add some randomness
        score += random.uniform(-0.1, 0.1)
        score = max(0.1, score)  # Ensure positive score
        
        # Create a result dictionary
        result = {
            'trial_id': i,
            'config': config,
            'score': score,
            'metrics': {
                'training_time': random.uniform(10, 100),
                'val_improvement': random.uniform(0.1, 0.5),
                'val_stability': random.uniform(0.01, 0.1)
            },
            'history': {
                'train_loss': [random.uniform(0.5, 1.0) * (0.9 ** j) for j in range(10)],
                'val_loss': [random.uniform(0.6, 1.2) * (0.9 ** j) for j in range(10)]
            }
        }
        
        sample_results.append(result)
    
    # Sort by score
    sample_results.sort(key=lambda x: x['score'])
    
    # Get best config
    best_config = sample_results[0]['config'].copy()
    best_config['feature_cols'] = ['vst_raw']
    
    # Generate hyperparameter report
    print("\nGenerating hyperparameter report...")
    generate_hyperparameter_report(
        results=sample_results,
        best_config=best_config,
        output_dir=output_dir,
        evaluation_metric='score'
    )
    
    def ensure_png_output(func_name, output_dir):
        """Check if a PNG file was created for the function, create a placeholder if not."""
        # Generate possible file names based on function name
        possible_names = [
            f"{func_name}.png",
            f"{func_name.replace('plot_', '')}.png",
            f"{func_name.replace('plot_', '').replace('_', '')}.png",
            f"{func_name.replace('plot_', '').replace('_', '-')}.png",
            f"hyperparameter_{func_name.replace('plot_', '')}.png",
            f"{func_name.replace('plot_', '')}_analysis.png",
            f"{func_name.split('_')[0]}.png"
        ]
        
        # Special cases for known functions
        if func_name == "plot_hyperparameter_pairwise_interactions":
            possible_names.append("hyperparameter_pairwise.png")
        elif func_name == "analyze_hyperparameter_importance":
            possible_names.append("hyperparameter_importance.png")
        elif func_name == "create_3d_surface_plots":
            possible_names.extend(["3d_surface_insufficient.png", "3d_surface_hidden_size_batch_size.png"])
        elif func_name == "plot_convergence_analysis":
            possible_names.extend(["convergence_analysis.png", "convergence_speed.png"])
        elif func_name == "plot_learning_curve_clusters":
            possible_names.append("learning_curve_clusters.png")
        elif func_name == "plot_parameter_sensitivity":
            possible_names.append("parameter_sensitivity.png")
        elif func_name == "plot_top_n_distributions":
            possible_names.append("top_parameter_distributions.png")
        elif func_name == "generate_architecture_impact_plot":
            possible_names.append("architecture_impact.png")
        elif func_name == "plot_learning_rate_landscape":
            possible_names.append("learning_rate_landscape.png")
        elif func_name == "plot_network_architecture_performance":
            possible_names.append("network_architecture_performance.png")
        elif func_name == "plot_training_dynamics":
            possible_names.append("training_dynamics.png")
        elif func_name == "plot_training_time_analysis":
            possible_names.append("training_time_analysis.png")
        elif func_name == "plot_parameter_evolution":
            possible_names.append("parameter_evolution.png")
        elif func_name == "plot_parallel_coordinates_clustered":
            possible_names.append("parallel_coordinates_clustered.png")
        
        # Debug: print all possible names we're checking
        print(f"  Checking for PNG files for {func_name}:")
        
        # Check both in main directory and plots subdirectory
        found = False
        found_file = None
        
        for name in possible_names:
            print(f"    ✗ {name} not found", end="")
            
            # Check in main directory
            main_path = output_dir / name
            if main_path.exists():
                found = True
                found_file = main_path
                print("\r    ✓ Found", name, "in main directory")
                break
                
            # Check in plots subdirectory
            plots_path = output_dir / "plots" / name
            if plots_path.exists():
                found = True
                found_file = plots_path
                print("\r    ✓ Found", name, "in plots directory")
                break
                
            print()  # Complete the line
        
        if found:
            print(f"  Using existing PNG: {found_file}")
            return
            
        # If no PNG was found, create a placeholder
        print(f"  Creating placeholder PNG for {func_name}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                f"No output generated for {func_name}\n"
                f"This is a placeholder image.",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Save with a distinctive name
        placeholder_path = output_dir / f"{func_name}_placeholder.png"
        plt.savefig(placeholder_path, dpi=300)
        plt.close()
    
    print("\nTesting individual plotting functions...")
    
    functions_to_test = [
        plot_hyperparameter_pairwise_interactions,
        analyze_hyperparameter_importance,
        create_3d_surface_plots,
        plot_convergence_analysis,
        plot_learning_curve_clusters,
        plot_parameter_sensitivity,
        plot_top_n_distributions,
        generate_architecture_impact_plot,
        plot_learning_rate_landscape,
        plot_network_architecture_performance,
        plot_training_dynamics,
        plot_training_time_analysis,
        plot_parameter_evolution,
        plot_parallel_coordinates_clustered
    ]
    
    for func in functions_to_test:
        try:
            print(f"Testing {func.__name__}...")
            if func.__name__ == "plot_learning_curve_clusters":
                func(sample_results, output_dir, n_clusters=3)
            elif func.__name__ == "plot_top_n_distributions":
                func(sample_results, output_dir, top_n=5)
            else:
                func(sample_results, output_dir)
            print(f"✓ {func.__name__} completed successfully")
            
            # Ensure PNG output
            ensure_png_output(func.__name__, output_dir)
            
        except Exception as e:
            print(f"✗ Error in {func.__name__}: {str(e)}")
            
            # Create error PNG
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 
                    f"Error in {func.__name__}\n"
                    f"Error: {str(e)}",
                    ha='center', va='center', fontsize=14, color='red')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"{func.__name__}_error.png", dpi=300)
            plt.close()
    
    print("\nTesting plot_best_model_results...")
    try:
        # Create sample data for plot_best_model_results
        station_name = "test_station"
        original_data = pd.DataFrame({
            'vst_raw': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        }, index=pd.date_range(start='2020-01-01', periods=100, freq='D'))
        
        modified_data = original_data.copy()
        modified_data['vst_raw'] += np.random.normal(0, 0.2, 100)
        
        results = {
            'forecast': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.15, 100),
            'truth': original_data['vst_raw'].values
        }
        
        plot_best_model_results(
            station_name=station_name,
            results=results,
            original_data=original_data,
            modified_data=modified_data,
            output_dir=output_dir,
            trial_number=0
        )
        print("✓ plot_best_model_results completed successfully")
    except Exception as e:
        print(f"✗ Error in plot_best_model_results: {str(e)}")
    
    print("\nAll tests completed. Check the output directory for results.")
    print(f"Output directory: {os.path.abspath(output_dir)}")