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

from diagnostics.lstm_diagnostics import plot_reconstruction_results

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
    
    # Call plot_reconstruction_results but override its output directory
    # to prevent it from creating additional nested directories
    plot_reconstruction_results(
        original_data=original_data,
        modified_data=modified_data,
        reconstruction_errors=results['reconstruction_errors'],
        anomaly_flags=results['anomaly_flags'],
        timestamps=results['timestamps'],
        threshold=results['threshold'],
        station_name=f"{station_name}_best_trial_{trial_number}",
        output_dir=plots_dir,  # Use the plots_dir directly
        reconstructed_values=results.get('reconstructions'),
        create_subdirs=False,  # Add this parameter to lstm_diagnostics.py
        figsize=(14, 10),
        dpi=300
    )

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
        
        # Generate all available visualizations if we have enough data
        if len(results) >= 5:
            print("Generating pairwise interaction plots...")
            plot_hyperparameter_pairwise_interactions(results, output_dir)
            
            print("Generating hyperparameter importance analysis...")
            analyze_hyperparameter_importance(results, output_dir)
            
            print("Generating parameter sensitivity plots...")
            plot_parameter_sensitivity(results, output_dir)
            
            print("Generating training dynamics plots...")
            plot_training_dynamics(results, output_dir)
            
            print("Generating training time analysis...")
            plot_training_time_analysis(results, output_dir)
            
            print("Generating parameter evolution plots...")
            plot_parameter_evolution(results, output_dir)
            
            print("Generating parallel coordinates plot...")
            plot_parallel_coordinates_clustered(results, output_dir)
        else:
            print(f"Not enough trials ({len(results)}) for detailed visualizations. Need at least 5 trials.")
        
        if len(results) >= 4:
            print("Generating 3D surface plots...")
            create_3d_surface_plots(results, output_dir)
            
            print("Generating learning curve clusters...")
            plot_learning_curve_clusters(results, output_dir)
        
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
            scores = [r['score'] for r in results if 'score' in r]
            if scores:
                f.write(f"<tr><td>Number of Trials</td><td>{len(scores)}</td></tr>")
                f.write(f"<tr><td>Best Score</td><td>{min(scores):.6f}</td></tr>")
                f.write(f"<tr><td>Mean Score</td><td>{np.mean(scores):.6f}</td></tr>")
                f.write(f"<tr><td>Score Std Dev</td><td>{np.std(scores):.6f}</td></tr>")
            
            f.write("</table></div>")
            
            # Add links to generated plots
            f.write("<div class='section'><h2>Generated Visualizations</h2>")
            f.write("<ul>")
            plot_types = [
                "pairwise_interactions",
                "hyperparameter_importance",
                "parameter_sensitivity",
                "training_dynamics",
                "training_time",
                "parameter_evolution",
                "parallel_coordinates",
                "3d_surface",
                "learning_curves"
            ]
            for plot_type in plot_types:
                plot_path = output_dir / "plots" / f"{plot_type}.png"
                if plot_path.exists():
                    f.write(f"<li><a href='../plots/{plot_type}.png'>{plot_type.replace('_', ' ').title()}</a></li>")
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

def analyze_hyperparameter_importance(results, output_dir):
    """Analyze the importance of different hyperparameters using random forest regression."""
    # Prepare data
    X = []
    y = []
    param_names = []
    
    # Map for display names - updated to focus on core parameters
    display_names = {
        'sequence_length': 'Sequence Length',
        'lstm_units_0': 'LSTM Layer 1 Units',
        'lstm_units_1': 'LSTM Layer 2 Units',
        'lstm_units_2': 'LSTM Layer 3 Units',
        'lstm_units_3': 'LSTM Layer 4 Units',
        'dropout_rate': 'Dropout Rate',
        'learning_rate': 'Learning Rate',
        'batch_size': 'Batch Size',
        'epochs': 'Training Epochs'
    }
    
    # Extract features and target
    for result in results:
        if 'config' not in result or 'score' not in result:
            continue
            
        config = result['config']
        score = result['score']
        
        # Extract numerical hyperparameters
        features = {}
        
        # Handle scalar parameters
        for param in ['sequence_length', 'dropout_rate', 'learning_rate', 
                     'batch_size', 'epochs']:
            if param in config:
                    features[param] = config[param]
    
    # Convert features to a DataFrame
    X = pd.DataFrame([features])
    y = [score]
    
    # Train a Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
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
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_importance_rf.png', dpi=300)
    plt.close()
    
    # Return feature importances
    return dict(zip(X.columns[sorted_idx], importances[sorted_idx]))

def create_3d_surface_plots(results, output_dir):
    """Create 3D surface plots for the most important parameter pairs."""
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract data from results
    data = []
    for result in results:
        if 'config' in result and 'score' in result and not np.isinf(result['score']):
            row = {k: v for k, v in result['config'].items() 
                  if not isinstance(v, list)}  # Skip list parameters
            row['score'] = result['score']
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
    param_vars = {col: df[col].var() for col in df.columns if col != 'score'}
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
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
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
    
    # Add contour plot at the bottom for additional perspective
    offset = min(z) - (max(z) - min(z)) * 0.1  # Offset below the minimum z
    cset = ax.contourf(xi, yi, zi, zdir='z', offset=offset, cmap='viridis', alpha=0.5)
    
    # Add labels and title with improved styling
    ax.set_xlabel(param1_display, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(param2_display, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('Score', fontsize=14, fontweight='bold', labelpad=10)
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set z-axis limits to include the contour plot
    ax.set_zlim(offset, max(z) * 1.1)
    
    # Add informative title
    plt.title(f'Parameter Surface: Impact of {param1_display} and {param2_display} on Performance', 
              fontsize=16, pad=20)
    
    # Add colorbar with label
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Score Value', fontsize=12, fontweight='bold')
    
    # Add a text box with key insights
    min_point = np.unravel_index(np.argmin(zi) if np.min(z) == np.min(zi) else np.argmin(z), z.shape if np.min(z) == np.min(zi) else (len(z),))
    best_x = xi.flatten()[min_point] if np.min(z) == np.min(zi) else x[np.argmin(z)]
    best_y = yi.flatten()[min_point] if np.min(z) == np.min(zi) else y[np.argmin(z)]
    
    ax.text2D(0.02, 0.01, f"Best combination:\n{param1_display}: {best_x:.2f}\n{param2_display}: {best_y:.2f}\nScore: {np.min(z):.6f}", 
             transform=ax.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Set the view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'3d_surface_{param1}_{param2}.png', dpi=300)
    plt.close()
    
    # Create a second view from a different angle
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8, linewidth=0)
    ax.scatter(x, y, z, c=z, cmap='plasma', s=70, alpha=1.0, edgecolor='black')
    
    # Set labels and title
    ax.set_xlabel(param1_display, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(param2_display, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('Score', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set a different viewing angle
    ax.view_init(elev=10, azim=120)
    
    plt.title(f'Alternate View: {param1_display} and {param2_display} Impact', fontsize=16, pad=20)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'3d_surface_{param1}_{param2}_alt_view.png', dpi=300)
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
        key = ", ".join([f"{p.replace('_', ' ').title()}={v}" for p, v in key_values.items()])
        
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
    
    # Extract architectural features from trials
    features = {
        'use_attention': [],
        'use_residual': [],
        'use_layer_norm': [],
        'num_layers': []  # Inferred from lstm_units length
    }
    
    scores = []
    
    for trial in trials_data:
        # Get architecture features
        params = trial['params']
        
        features['use_attention'].append(params.get('use_attention', False))
        features['use_residual'].append(params.get('use_residual', False))
        features['use_layer_norm'].append(params.get('use_layer_norm', False))
        
        # Infer number of layers from lstm_units
        lstm_units = params.get('lstm_units', [64, 32])
        features['num_layers'].append(len(lstm_units))
        
        # Get score
        scores.append(trial['score'])
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot impact of each feature
    for i, (feature, values) in enumerate(features.items()):
        row, col = i // 2, i % 2
        ax = axs[row, col]
        
        # Convert to categorical for boxplot if needed
        if feature in ['use_attention', 'use_residual', 'use_layer_norm']:
            # Convert boolean to string for better labels
            categories = ['No', 'Yes']
            cat_values = ['Yes' if v else 'No' for v in values]
        else:
            # For num_layers, use unique values
            categories = sorted(set(values))
            cat_values = values
        
        # Create boxplot data
        boxplot_data = []
        for category in categories:
            category_scores = [score for val, score in zip(cat_values, scores) if val == category]
            boxplot_data.append(category_scores)
        
        # Create boxplot
        ax.boxplot(boxplot_data, labels=categories)
        ax.set_title(f'Impact of {feature.replace("_", " ").title()}')
        ax.set_ylabel('Score')
        
    plt.tight_layout()
    plt.savefig(plot_dir / "architecture_impact.png")
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
    df = pd.DataFrame(lr_data)
    sns.scatterplot(data=df, x='lr', y='final_loss', size='convergence_speed', 
                   sizes=(100, 400), alpha=0.6, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Rate Impact on Model Performance', fontsize=14, pad=20)
    
    # Plot 2: Loss Curves for Different Learning Rates
    for data in lr_data:
        epochs = range(1, len(data['loss_curve']) + 1)
        ax2.plot(epochs, data['loss_curve'], 
                label=f"LR: {data['lr']:.1e}", alpha=0.7)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Patterns by Learning Rate', fontsize=14, pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_landscape.png', dpi=300, bbox_inches='tight')
    plt.close() 

def plot_network_architecture_performance(results: List[Dict], output_dir: Path):
    """Visualize performance across different network architectures."""
    # Extract architecture data
    arch_data = []
    for result in results:
        if ('config' in result and 'score' in result and 
            not np.isinf(result.get('score', float('inf')))):
            config = result['config']
            if 'lstm_units' in config:
                arch_data.append({
                    'n_layers': len(config['lstm_units']),
                    'total_params': sum(config['lstm_units']),
                    'layer_sizes': ' → '.join(map(str, config['lstm_units'])),
                    'score': result['score']
                })
    
    if not arch_data:
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
    
    plt.tight_layout()
    plt.savefig(output_dir / 'network_architecture_performance.png', dpi=300)
    plt.close() 

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
    pivot_data = df.pivot_table(
        values='loss_volatility', 
        index=pd.qcut(df['batch_size'], 4), 
        columns=pd.qcut(df['sequence_length'], 4),
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax3)
    ax3.set_xlabel('Sequence Length (Quartiles)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Batch Size (Quartiles)', fontsize=12, fontweight='bold')
    ax3.set_title('Training Stability Heatmap', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dynamics.png', dpi=300)
    plt.close() 

def plot_training_time_analysis(results: List[Dict], output_dir: Path):
    """Analyze how different parameters affect training time."""
    time_data = []
    for result in results:
        if 'config' in result and 'trial_time' in result:
            config = result['config']
            time_data.append({
                'batch_size': config.get('batch_size', 0),
                'sequence_length': config.get('sequence_length', 0),
                'n_layers': len(config.get('lstm_units', [])),
                'total_units': sum(config.get('lstm_units', [])),
                'training_time': result['trial_time'],
                'score': result.get('score', float('inf'))
            })
    
    if not time_data:
        return
        
    df = pd.DataFrame(time_data)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Plot 1: Model Size vs Training Time
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(df['total_units'], df['training_time'], 
                          c=df['score'], cmap='viridis', 
                          s=100, alpha=0.6)
    ax1.set_xlabel('Total LSTM Units', fontsize=12)
    ax1.set_ylabel('Training Time (s)', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Score')
    
    # Plot 2: Sequence Length vs Training Time
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(df['sequence_length'], df['training_time'],
                          c=df['batch_size'], cmap='plasma',
                          s=100, alpha=0.6)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Training Time (s)', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Batch Size')
    
    # Plot 3: Training Time Distribution
    ax3 = fig.add_subplot(gs[1, :])
    sns.boxenplot(data=df, x='n_layers', y='training_time', ax=ax3)
    ax3.set_xlabel('Number of LSTM Layers', fontsize=12)
    ax3.set_ylabel('Training Time (s)', fontsize=12)
    
    plt.suptitle('Training Time Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close() 

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
    """Create a parallel coordinates plot with clustering of similar configurations."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Extract successful trials
    valid_results = [r for r in results if not np.isinf(r.get('score', float('inf')))]
    if len(valid_results) < 5:  # Need enough data for meaningful clustering
        return
        
    # Prepare data
    data = []
    for result in valid_results:
        if 'config' in result:
            row = {k: v for k, v in result['config'].items() 
                  if not isinstance(v, list)}
            row['score'] = result['score']
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Normalize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )
    
    # Perform clustering
    n_clusters = min(3, len(df) // 2)  # Adjust number of clusters based on data size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    
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
    
    fig.write_html(output_dir / 'parallel_coordinates_clustered.html') 