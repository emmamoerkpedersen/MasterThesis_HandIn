import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

def generate_hyperparameter_report(study_path, output_dir, top_n=5):
    """
    Generate a comprehensive report on hyperparameter tuning results
    
    Args:
        study_path: Path to the Optuna study database or results JSON
        output_dir: Directory to save the report
        top_n: Number of top models to include in the report
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load study or trials data
    try:
        # First try loading from JSON file with all trials
        if Path(study_path).is_file() and str(study_path).endswith('.json'):
            with open(study_path, 'r') as f:
                trials_data = json.load(f)
                
            # Create a DataFrame from the trials data
            df = pd.DataFrame([
                {**{'trial': t['number'], 'score': t['value']}, **t['params']} 
                for t in trials_data
            ])
        
        # If not a JSON file, try loading as an Optuna study
        else:
            study = optuna.load_study(
                study_name=None,
                storage=f"sqlite:///{study_path}"
            )
            
            # Extract data from completed trials
            trials_data = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    trial_info = {
                        'trial': trial.number,
                        'score': trial.value,
                        **trial.params
                    }
                    trials_data.append(trial_info)
            
            df = pd.DataFrame(trials_data)
    
    except Exception as e:
        print(f"Error loading study data: {e}")
        return
    
    if df.empty:
        print("No valid trials found.")
        return
    
    # Sort by score (assuming lower is better)
    df = df.sort_values('score')
    
    # Create the report
    create_hyperparameter_overview_plot(df, output_dir, top_n)
    create_top_models_table(df, output_dir, top_n)
    create_correlation_plot(df, output_dir)
    create_parallel_coordinates_plot(df, output_dir, top_n)
    create_parameter_importance_plot(df, output_dir)
    
    print(f"Hyperparameter tuning report generated in {output_dir}")

def create_hyperparameter_overview_plot(df, output_dir, top_n=5):
    """
    Create an overview plot of hyperparameter tuning performance
    """
    plt.figure(figsize=(15, 10))
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    gs = GridSpec(2, 2, figure=plt.gcf(), height_ratios=[1, 1], width_ratios=[2, 1])
    
    # Plot 1: Performance trend over trials
    ax1 = plt.subplot(gs[0, 0])
    best_scores = df['score'].cummin()
    
    ax1.plot(df['trial'], df['score'], 'o-', alpha=0.5, markersize=4, label='Trial Score')
    ax1.plot(df['trial'], best_scores, 'r-', linewidth=2, label='Best Score')
    
    # Highlight top N models
    top_n_df = df.head(top_n)
    ax1.scatter(top_n_df['trial'], top_n_df['score'], color='red', s=80, 
               edgecolor='black', zorder=10, label=f'Top {top_n} Models')
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Hyperparameter Optimization Progress', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Top N models performance comparison
    ax2 = plt.subplot(gs[0, 1])
    
    model_names = [f"Model {i+1}\n(Trial {trial})" for i, trial in enumerate(top_n_df['trial'])]
    ax2.barh(model_names, top_n_df['score'], color=sns.color_palette("viridis", top_n))
    
    # Add score values to the end of each bar
    for i, score in enumerate(top_n_df['score']):
        ax2.text(score + 0.0001, i, f"{score:.4f}", va='center')
    
    ax2.set_xlabel('Validation Loss')
    ax2.set_title(f'Top {top_n} Models', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Plot 3: Parameter distribution by performance
    ax3 = plt.subplot(gs[1, :])
    
    # Get numeric parameters that have multiple values
    numeric_params = []
    for col in df.columns:
        if col not in ['trial', 'score']:
            unique_values = df[col].nunique()
            if unique_values > 1 and pd.api.types.is_numeric_dtype(df[col]):
                numeric_params.append(col)
    
    if not numeric_params:
        ax3.text(0.5, 0.5, "No suitable parameters for distribution plot", 
                ha='center', va='center', fontsize=14)
    else:
        # Select 3-4 most important parameters for visualization
        if len(numeric_params) > 4:
            # Choose most important params (those with highest correlation to score)
            correlations = [abs(df[param].corr(df['score'])) for param in numeric_params]
            sorted_params = [x for _, x in sorted(zip(correlations, numeric_params), reverse=True)]
            selected_params = sorted_params[:4]
        else:
            selected_params = numeric_params
        
        # Create violin plots for each parameter, colored by performance
        for i, param in enumerate(selected_params):
            # Create a sequential color mapping based on performance
            norm = plt.Normalize(df['score'].min(), df['score'].max())
            colors = plt.cm.viridis(norm(df['score']))
            
            position = i + 1
            parts = ax3.violinplot(dataset=[df[param]], positions=[position], 
                               showmeans=False, showmedians=True)
            
            # Customize violin appearance
            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)
            
            # Add individual points
            y = df[param]
            x = np.random.normal(position, 0.05, size=len(y))
            scatter = ax3.scatter(x, y, c=df['score'], cmap='viridis', 
                             s=30, alpha=0.8, edgecolors='gray')
            
            # Add best model points
            top_values = top_n_df[param]
            top_x = [position] * len(top_values)
            ax3.scatter(top_x, top_values, color='red', s=80, 
                       edgecolor='black', zorder=10, marker='*')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, orientation='vertical', pad=0.01)
        cbar.set_label('Validation Loss')
        
        # Configure axis
        ax3.set_xticks(range(1, len(selected_params) + 1))
        ax3.set_xticklabels([p.replace('_', ' ').title() for p in selected_params])
        ax3.set_title('Parameter Distribution vs Performance', fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "hyperparameter_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hyperparameter overview plot to {output_path}")

def create_top_models_table(df, output_dir, top_n=5):
    """
    Create a visual table of top model hyperparameters
    """
    # Get top N models
    top_models = df.head(top_n).copy()
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Drop the 'score' column
    param_cols = [col for col in top_models.columns if col not in ['score']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(param_cols) * 0.5 + 2))
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Create table data
    table_data = []
    header = ['Parameter']
    for i, (_, row) in enumerate(top_models.iterrows()):
        header.append(f"Model {i+1}\n(Trial {int(row['trial'])})")
    
    # Format each parameter value
    for param in param_cols:
        if param == 'trial':
            continue
            
        param_row = [param.replace('_', ' ').title()]
        for _, row in top_models.iterrows():
            value = row[param]
            if isinstance(value, float):
                # Format based on value range
                if abs(value) < 0.01 or abs(value) >= 1000:
                    formatted_value = f"{value:.2e}"
                elif param == 'learning_rate':
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            param_row.append(formatted_value)
        table_data.append(param_row)
    
    # Add score as last row
    score_row = ['Validation Loss']
    for _, row in top_models.iterrows():
        score_row.append(f"{row['score']:.6f}")
    table_data.append(score_row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=header, loc='center',
                    cellLoc='center', colLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color the header and score row
    for i, key in enumerate(table._cells):
        cell = table._cells[key]
        if key[0] == 0:  # Header row
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[1] == 0:  # Parameter names column
            cell.set_facecolor('#D9E1F2')
            cell.set_text_props(fontweight='bold')
        elif table_data[key[0]-1][0] == 'Validation Loss':  # Score row
            cell.set_facecolor('#E2EFDA')
        
        # Add border
        cell.set_edgecolor('black')
    
    plt.title('Top Models Hyperparameter Comparison', fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save as image
    output_path = output_dir / "top_models_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV for reference
    csv_path = output_dir / "top_models.csv"
    top_models.to_csv(csv_path, index=False)
    
    print(f"Saved top models table to {output_path}")
    print(f"Saved top models data to {csv_path}")

def create_correlation_plot(df, output_dir):
    """
    Create a correlation matrix plot of hyperparameters vs performance
    """
    # Only include numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if 'trial' in numeric_cols:
        numeric_cols.remove('trial')
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation plot")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Draw the heatmap with the mask
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
    
    plt.title('Hyperparameter Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "hyperparameter_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation plot to {output_path}")

def create_parallel_coordinates_plot(df, output_dir, top_n=5):
    """
    Create a parallel coordinates plot to visualize the hyperparameter space
    """
    # Get columns to include (exclude trial number)
    param_cols = [col for col in df.columns if col != 'trial']
    
    if len(param_cols) < 3:
        print("Not enough parameters for parallel coordinates plot")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Normalize the data for consistent scale
    normalized_df = df[param_cols].copy()
    for col in param_cols:
        if col != 'score' and pd.api.types.is_numeric_dtype(df[col]):
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            # Avoid division by zero
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    # Sort by performance (best first)
    normalized_df = normalized_df.sort_values('score')
    
    # Color by score (best = darkest)
    norm = plt.Normalize(normalized_df['score'].min(), normalized_df['score'].max())
    colors = plt.cm.viridis_r(norm(normalized_df['score']))
    
    # Plot all trials with alpha proportional to performance rank
    for i, (idx, row) in enumerate(normalized_df.iterrows()):
        # Alpha decreases as rank increases (worst models are more transparent)
        alpha = 0.3 + 0.7 * (1 - i / len(normalized_df))
        
        # Plot line
        xs = list(range(len(param_cols)))
        ys = [row[col] for col in param_cols]
        plt.plot(xs, ys, color=colors[i], alpha=alpha, linewidth=1.5)
    
    # Highlight top N models
    top_n_df = normalized_df.head(top_n)
    for i, (idx, row) in enumerate(top_n_df.iterrows()):
        xs = list(range(len(param_cols)))
        ys = [row[col] for col in param_cols]
        plt.plot(xs, ys, color='red', alpha=0.8, linewidth=2.5, 
                label=f"Model {i+1} (Loss: {df.loc[idx, 'score']:.4f})")
    
    # Set axis labels and ticks
    plt.xticks(range(len(param_cols)), [p.replace('_', ' ').title() for p in param_cols], rotation=30)
    plt.yticks([])  # Hide y-ticks for cleaner look
    
    # Add y-axis labels for each parameter
    for i, param in enumerate(param_cols):
        if param != 'score' and pd.api.types.is_numeric_dtype(df[param]):
            min_val = df[param].min()
            max_val = df[param].max()
            plt.text(i, 1.02, f"{max_val:.4g}", ha='center', va='bottom', fontsize=9)
            plt.text(i, -0.02, f"{min_val:.4g}", ha='center', va='top', fontsize=9)
        elif param == 'score':
            min_val = df[param].min()
            max_val = df[param].max()
            plt.text(i, 1.02, f"{min_val:.4g}", ha='center', va='bottom', fontsize=9)
            plt.text(i, -0.02, f"{max_val:.4g}", ha='center', va='top', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.title('Parallel Coordinates Plot of Hyperparameter Space', fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    
    plt.tight_layout()
    output_path = output_dir / "hyperparameter_parallel_coordinates.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parallel coordinates plot to {output_path}")

def create_parameter_importance_plot(df, output_dir):
    """
    Create a plot showing the importance of each parameter
    """
    # Only include parameters, not trial and score
    param_cols = [col for col in df.columns if col not in ['trial', 'score']]
    
    if len(param_cols) < 2:
        print("Not enough parameters for importance plot")
        return
    
    # Calculate feature importance using correlation or custom method
    importance_dict = {}
    
    for param in param_cols:
        if pd.api.types.is_numeric_dtype(df[param]) and df[param].nunique() > 1:
            # Calculate correlation with score
            corr = abs(df[param].corr(df['score']))
            importance_dict[param] = corr
        elif df[param].nunique() > 1:
            # For categorical parameters, calculate importance based on variance of group means
            grouped = df.groupby(param)['score'].mean()
            if len(grouped) > 1:
                # Importance is the variance of group means normalized by overall variance
                importance = grouped.var() / df['score'].var()
                importance_dict[param] = min(importance, 1.0)  # Cap at 1.0
    
    if not importance_dict:
        print("No valid importance scores calculated")
        return
    
    # Sort by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    params = [item[0].replace('_', ' ').title() for item in sorted_importance]
    scores = [item[1] for item in sorted_importance]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Create horizontal bar chart
    bars = plt.barh(params, scores, color=sns.color_palette("viridis", len(params)))
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{width:.2f}", va='center')
    
    plt.xlabel('Relative Importance')
    plt.title('Hyperparameter Importance Analysis', fontweight='bold')
    plt.xlim(0, max(scores) * 1.1)  # Add some margin for labels
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "hyperparameter_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter importance plot to {output_path}")

def save_hyperparameter_results(tuning_results, output_dir):
    """
    Save hyperparameter tuning results and generate visualizations
    
    Args:
        tuning_results: Dictionary containing tuning results
        output_dir: Path to save the results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_path = output_dir / "hyperparameter_results.json"
    with open(results_path, 'w') as f:
        json.dump(tuning_results, f, indent=4)
    
    # Generate report if there are trials
    if 'trials' in tuning_results and tuning_results['trials']:
        generate_hyperparameter_report(results_path, output_dir)
    
    print(f"Saved hyperparameter results to {results_path}")
