"""
Grid search for LSTM hyperparameters.

This script performs a systematic grid search across multiple hyperparameters:
- Hidden sizes: 32, 64, 128, 256
- Layers: 1, 2
- Sequence lengths: 2000, 4000, 5000, 6000, 10000, 20000
- Learning rates: 0.01, 0.001, 0.0001, 0.00001
- Loss functions: peak_weighted_loss, dynamic_weighted_loss, smoothL1_loss, mse_loss

The results are saved to disk and plots are generated for each model.
"""

import os
import json
import itertools
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Import from project modules
sys.path.append(str(Path(__file__).parent.parent))

from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence
from _3_lstm_model.objective_functions import get_objective_function
from experiments.Improved_model_structure.train_model import LSTM_Trainer
from config import LSTM_CONFIG

def setup_grid_search():
    """
    Set up the grid search parameters.
    
    Returns:
        dict: Grid search parameters
    """
    # Set up grid search parameters
    grid_params = {
        'hidden_size': [64, 128, 256, 284, 320],
        'num_layers': [1, 2],
        'sequence_length': [3000, 4000, 5000, 10000],
        'learning_rate': [0.0001,],
        'objective_function': ['peak_weighted_loss', 'dynamic_weighted_loss', 'smoothL1_loss', 'smoothL1_dynamic_weighted_loss']
    }
    
    return grid_params

def generate_config_combinations(base_config, grid_params):
    """
    Generate all combinations of hyperparameters.
    
    Args:
        base_config: Base configuration dictionary
        grid_params: Grid search parameters dictionary
    
    Returns:
        list: List of configuration dictionaries
    """
    # Extract the keys and values from the grid params
    keys = grid_params.keys()
    values = grid_params.values()
    
    # Generate all combinations
    combinations = list(itertools.product(*values))
    
    # Create configurations for each combination
    configs = []
    for combo in combinations:
        config = base_config.copy()
        for i, key in enumerate(keys):
            config[key] = combo[i]
        configs.append(config)
    
    return configs

def get_config_name(config):
    """
    Generate a unique name for a configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        str: Unique configuration name
    """
    return (f"h{config['hidden_size']}_"
            f"l{config['num_layers']}_"
            f"s{config['sequence_length']}_"
            f"lr{config['learning_rate']}_"
            f"{config['objective_function']}")

def run_single_model(config, train_data, val_data, output_dir, station_id, preprocessor):
    """
    Train a single model with the given configuration and save results.
    
    Args:
        config: Configuration dictionary
        train_data: Training data
        val_data: Validation data
        output_dir: Output directory for results
        station_id: Station ID
        preprocessor: DataPreprocessor object
    
    Returns:
        tuple: (Best validation loss, history dictionary, model object)
    """
    print("\n" + "="*80)
    print(f"Training model with configuration: {get_config_name(config)}")
    print("="*80)
    
    # Generate a unique model name
    model_name = get_config_name(config)
    
    # Set up model directory - organize by loss function first, then other parameters
    loss_function = config['objective_function']
    model_subdir = model_name.replace(f"_{loss_function}", "")  # Remove loss function from name for subdirectory
    model_dir = output_dir / "models_by_loss_function" / loss_function / model_subdir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create a flat plots directory
    all_plots_dir = output_dir / "all_plots"
    all_plots_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(model_dir / "config.json", "w") as f:
        # Convert config values to JSON serializable types
        json_config = {k: v if not isinstance(v, (np.int64, np.float64, np.bool_)) 
                      else v.item() for k, v in config.items()}
        json.dump(json_config, f, indent=4)
    
    # Initialize trainer
    trainer = LSTM_Trainer(config, preprocessor=preprocessor)
    
    # Limit epochs for quick results during grid search
    epochs = 1000  # Reduced epochs for grid search
    
    # Print model configuration
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Layers: {config['num_layers']}")
    print(f"Sequence Length: {config['sequence_length']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Loss Function: {config['objective_function']}")
    
    try:
        # Suppress detailed output during training
        import sys, os
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Train the model
        start_time = time.time()
        history, val_predictions, val_targets = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=config['batch_size'],
            patience=config['patience']
        )
        training_time = time.time() - start_time
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Get best validation loss
        best_val_loss = min(history['val_loss'])
        
        # Convert validation predictions to numpy and reshape
        val_predictions = val_predictions.cpu().numpy()
        predictions_reshaped = val_predictions.reshape(-1, 1)
        predictions_original = preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped)
        predictions_flattened = predictions_original.flatten()
        
        # Create DataFrame with aligned predictions and targets
        val_predictions_df = pd.DataFrame(
            predictions_flattened[:len(val_data)],
            index=val_data.index[:len(predictions_flattened)],
            columns=['vst_raw']
        )
        
        # Get performance metrics - use the metrics from history if available
        if 'metrics' in history:
            performance_metrics = history['metrics']
        else:
            # Fallback: Calculate metrics using the validation data
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            actual_vals = val_data['vst_raw'].dropna().values
            pred_vals = val_predictions_df.loc[val_data['vst_raw'].dropna().index, 'vst_raw'].values
            
            if len(actual_vals) > 0 and len(pred_vals) > 0:
                rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
                mae = mean_absolute_error(actual_vals, pred_vals) 
                r2 = r2_score(actual_vals, pred_vals)
            else:
                rmse = float('nan')
                mae = float('nan')
                r2 = float('nan')
                
            performance_metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_error': float('nan'),
                'std_error': float('nan'),
                'peak_mae': float('nan'),
                'peak_rmse': float('nan')
            }
            
        # Create plots
        print("Generating plots...")
        
        # Create full prediction plot - save in model directory
        # Only generate PNG plots (no HTML)
        plot_path = create_full_plot(
            val_data, 
            val_predictions_df, 
            str(station_id), 
            config,
            best_val_loss,
            create_html=False,
            open_browser=False,
            metrics=performance_metrics  # Pass metrics to the plotting function
        )
        
        # Copy the PNG file to all_plots directory with model name prefix
        import shutil
        if plot_path and plot_path.exists():
            all_plots_png_path = all_plots_dir / f"{model_name}_{plot_path.name}"
            shutil.copy(plot_path, all_plots_png_path)
            print(f"Copied PNG plot to: {all_plots_png_path}")
        
        # Create convergence plot
        convergence_path = model_dir / f"convergence_plot_{station_id}.png"
        plot_convergence(
            history, 
            str(station_id), 
            title=f"Training and Validation Loss - {model_subdir}"
        )
        
        # Copy convergence plot to all_plots directory
        if convergence_path.exists():
            all_plots_convergence_path = all_plots_dir / f"{model_name}_convergence_plot_{station_id}.png"
            shutil.copy(convergence_path, all_plots_convergence_path)
        
        # Save model state
        torch.save(trainer.model.state_dict(), model_dir / "model.pth")
        
        # Save history
        with open(model_dir / "history.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if key == 'metrics':
                    # Convert metrics dictionary to json-serializable
                    json_history[key] = {k: float(v) for k, v in value.items()}
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.float32, np.float64)):
                    json_history[key] = [float(v) for v in value]
                else:
                    json_history[key] = value
            json.dump(json_history, f, indent=4)
        
        # Save metrics
        metrics = {
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(history['val_loss'][-1]),
            "training_time_seconds": training_time,
            "epochs_trained": len(history['train_loss']),
            "early_stopping": len(history['train_loss']) < epochs,
            "rmse": float(performance_metrics['rmse']),
            "mae": float(performance_metrics['mae']),
            "r2": float(performance_metrics['r2']),
            "peak_rmse": float(performance_metrics['peak_rmse']),
            "peak_mae": float(performance_metrics['peak_mae'])
        }
        
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Completed training in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        if not np.isnan(metrics['peak_rmse']):
            print(f"Peak RMSE: {metrics['peak_rmse']:.4f}")
            print(f"Peak MAE: {metrics['peak_mae']:.4f}")
        
        # Return results with metrics included
        result = {
            "best_val_loss": best_val_loss,
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "r2": metrics['r2'],
            "peak_rmse": metrics['peak_rmse'],
            "peak_mae": metrics['peak_mae']
        }
        
        return result, history, trainer.model
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "best_val_loss": float('inf'),
            "rmse": float('nan'),
            "mae": float('nan'),
            "r2": float('nan'),
            "peak_rmse": float('nan'),
            "peak_mae": float('nan')
        }, None, None

def create_best_models_notebook(results_df, output_dir, station_id):
    """
    Create a Jupyter notebook showcasing the best model for each loss function.
    
    Args:
        results_df: DataFrame with grid search results
        output_dir: Output directory for notebook
        station_id: Station ID
    """
    try:
        import nbformat as nbf
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
        
        notebook = new_notebook()
        
        # Create title
        notebook.cells.append(new_markdown_cell(
            f"# LSTM Grid Search Results - Station {station_id}\n\n"
            f"This notebook presents the best models from the grid search, "
            f"organized by loss function."
        ))
        
        # Add overview section
        notebook.cells.append(new_markdown_cell(
            "## Overview\n\n"
            "This grid search evaluated LSTM models with various hyperparameters:\n"
            "- Hidden sizes: " + ", ".join(map(str, sorted(results_df['hidden_size'].unique()))) + "\n"
            "- Layers: " + ", ".join(map(str, sorted(results_df['num_layers'].unique()))) + "\n"
            "- Sequence lengths: " + ", ".join(map(str, sorted(results_df['sequence_length'].unique()))) + "\n"
            "- Learning rates: " + ", ".join(map(str, sorted(results_df['learning_rate'].unique()))) + "\n"
            "- Loss functions: " + ", ".join(sorted(results_df['objective_function'].unique())) + "\n\n"
            f"Total models evaluated: {len(results_df)}"
        ))
        
        # Add best overall model section
        best_overall = results_df.loc[results_df['best_val_loss'].idxmin()]
        notebook.cells.append(new_markdown_cell(
            "## Best Overall Model\n\n"
            f"**Configuration:** {best_overall['config_name']}\n\n"
            f"**Best Validation Loss:** {best_overall['best_val_loss']:.6f}\n\n"
            f"- Hidden Size: {best_overall['hidden_size']}\n"
            f"- Layers: {best_overall['num_layers']}\n"
            f"- Sequence Length: {best_overall['sequence_length']}\n"
            f"- Learning Rate: {best_overall['learning_rate']}\n"
            f"- Loss Function: {best_overall['objective_function']}\n"
        ))
        
        # Add code to display best overall model plots
        notebook.cells.append(new_code_cell(
            f"# Display best overall model plots\n"
            f"from IPython.display import IFrame, display, HTML\n"
            f"import matplotlib.pyplot as plt\n"
            f"import pandas as pd\n"
            f"import os\n\n"
            f"# Define paths to best model files\n"
            f"loss_function = '{best_overall['objective_function']}'\n"
            f"model_subdir = '{best_overall['config_name']}'.replace(f'_{loss_function}', '')\n"
            f"model_dir = os.path.join('results', 'grid_search', 'models_by_loss_function', loss_function, model_subdir)\n\n"
            f"# Display prediction plot\n"
            f"prediction_plot = os.path.join(model_dir, 'prediction_plot_{station_id}.html')\n"
            f"if os.path.exists(prediction_plot):\n"
            f"    display(IFrame(prediction_plot, width=1000, height=600))\n"
            f"else:\n"
            f"    print(f'Prediction plot not found at: {{prediction_plot}}')\n\n"
            f"# Display convergence plot\n"
            f"convergence_plot = os.path.join(model_dir, 'convergence_plot_{station_id}.png')\n"
            f"if os.path.exists(convergence_plot):\n"
            f"    plt.figure(figsize=(10, 6))\n"
            f"    img = plt.imread(convergence_plot)\n"
            f"    plt.imshow(img)\n"
            f"    plt.axis('off')\n"
            f"    plt.show()\n"
            f"else:\n"
            f"    print(f'Convergence plot not found at: {{convergence_plot}}')\n"
        ))
        
        # Add sections for each loss function
        for loss_func in sorted(results_df['objective_function'].unique()):
            df_loss = results_df[results_df['objective_function'] == loss_func]
            best_for_loss = df_loss.loc[df_loss['best_val_loss'].idxmin()]
            
            notebook.cells.append(new_markdown_cell(
                f"## Best Model for {loss_func}\n\n"
                f"**Configuration:** {best_for_loss['config_name']}\n\n"
                f"**Best Validation Loss:** {best_for_loss['best_val_loss']:.6f}\n\n"
                f"- Hidden Size: {best_for_loss['hidden_size']}\n"
                f"- Layers: {best_for_loss['num_layers']}\n"
                f"- Sequence Length: {best_for_loss['sequence_length']}\n"
                f"- Learning Rate: {best_for_loss['learning_rate']}\n"
            ))
            
            # Add code to display best model plots for this loss function
            notebook.cells.append(new_code_cell(
                f"# Display plots for best model with {loss_func}\n"
                f"loss_function = '{loss_func}'\n"
                f"model_subdir = '{best_for_loss['config_name']}'.replace(f'_{loss_function}', '')\n"
                f"model_dir = os.path.join('results', 'grid_search', 'models_by_loss_function', loss_function, model_subdir)\n\n"
                f"# Display prediction plot\n"
                f"prediction_plot = os.path.join(model_dir, 'prediction_plot_{station_id}.html')\n"
                f"if os.path.exists(prediction_plot):\n"
                f"    display(IFrame(prediction_plot, width=1000, height=600))\n"
                f"else:\n"
                f"    print(f'Prediction plot not found at: {{prediction_plot}}')\n\n"
                f"# Display convergence plot\n"
                f"convergence_plot = os.path.join(model_dir, 'convergence_plot_{station_id}.png')\n"
                f"if os.path.exists(convergence_plot):\n"
                f"    plt.figure(figsize=(10, 6))\n"
                f"    img = plt.imread(convergence_plot)\n"
                f"    plt.imshow(img)\n"
                f"    plt.axis('off')\n"
                f"    plt.show()\n"
                f"else:\n"
                f"    print(f'Convergence plot not found at: {{convergence_plot}}')\n"
            ))
            
            # Add hyperparameter effect plots for this loss function
            notebook.cells.append(new_markdown_cell(
                f"### Hyperparameter Effects for {loss_func}\n\n"
                f"These plots show how different hyperparameters affect model performance "
                f"for the {loss_func} loss function."
            ))
            
            notebook.cells.append(new_code_cell(
                f"# Display hyperparameter effect plots for {loss_func}\n"
                f"loss_summary_dir = os.path.join('results', 'grid_search', 'summary_plots', '{loss_func}')\n"
                f"param_plots = ['hidden_size_effect.png', 'sequence_length_effect.png', \n"
                f"                'learning_rate_effect.png', 'num_layers_effect.png']\n\n"
                f"for plot in param_plots:\n"
                f"    plot_path = os.path.join(loss_summary_dir, plot)\n"
                f"    if os.path.exists(plot_path):\n"
                f"        plt.figure(figsize=(8, 5))\n"
                f"        img = plt.imread(plot_path)\n"
                f"        plt.imshow(img)\n"
                f"        plt.axis('off')\n"
                f"        plt.title(plot.replace('_effect.png', '').replace('_', ' ').title())\n"
                f"        plt.show()\n"
            ))
            
            # Add summary stats for this loss function
            notebook.cells.append(new_markdown_cell(
                f"### Summary Statistics for {loss_func}\n\n"
                f"The following table shows all models trained with {loss_func}, ordered by validation loss."
            ))
            
            notebook.cells.append(new_code_cell(
                f"# Display summary table for {loss_func}\n"
                f"loss_summary_file = os.path.join('results', 'grid_search', 'summary_plots', '{loss_func}', '{loss_func}_results.csv')\n"
                f"if os.path.exists(loss_summary_file):\n"
                f"    df = pd.read_csv(loss_summary_file)\n"
                f"    display(df.head(10))\n"
                f"else:\n"
                f"    print(f'Summary file not found at: {{loss_summary_file}}')\n"
            ))
        
        # Add comparison section
        notebook.cells.append(new_markdown_cell(
            "## Loss Function Comparison\n\n"
            "This section compares the performance across different loss functions."
        ))
        
        notebook.cells.append(new_code_cell(
            "# Display loss function comparison plots\n"
            "comparison_plots = ['hidden_size_vs_loss.png', 'sequence_length_vs_loss.png', 'learning_rate_vs_loss.png']\n"
            "summary_dir = os.path.join('results', 'grid_search', 'summary_plots')\n\n"
            "for plot in comparison_plots:\n"
            "    plot_path = os.path.join(summary_dir, plot)\n"
            "    if os.path.exists(plot_path):\n"
            "        plt.figure(figsize=(10, 6))\n"
            "        img = plt.imread(plot_path)\n"
            "        plt.imshow(img)\n"
            "        plt.axis('off')\n"
            "        plt.title(plot.replace('.png', '').replace('_', ' ').title())\n"
            "        plt.show()\n"
        ))
        
        # Add final comparison table
        notebook.cells.append(new_markdown_cell(
            "## Final Comparison Table\n\n"
            "The following table shows the best model for each loss function."
        ))
        
        notebook.cells.append(new_code_cell(
            "# Display best models by loss function\n"
            "best_models_file = os.path.join('results', 'grid_search', 'summary_plots', 'best_models_by_loss_function.csv')\n"
            "if os.path.exists(best_models_file):\n"
            "    best_models = pd.read_csv(best_models_file)\n"
            "    display(best_models)\n"
            "else:\n"
            "    print(f'Best models file not found at: {best_models_file}')\n"
        ))
        
        # Save the notebook
        notebook_path = output_dir / "best_models_analysis.ipynb"
        with open(notebook_path, 'w') as f:
            nbf.write(notebook, f)
        
        print(f"Created analysis notebook at: {notebook_path}")
        return notebook_path
        
    except ImportError:
        print("Could not create notebook - nbformat package not installed.")
        print("To create the notebook, install with: pip install nbformat")
        return None
    except Exception as e:
        print(f"Error creating notebook: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_grid_search(project_root, output_path, station_id):
    """
    Run the grid search for all hyperparameter combinations.
    
    Args:
        project_root: Root directory of the project
        output_path: Output directory for results
        station_id: Station ID
    
    Returns:
        pd.DataFrame: Results summary
    """
    # Start with base configuration from config.py
    base_config = LSTM_CONFIG.copy()
    
    # Ensure time and cumulative features are enabled
    base_config['use_time_features'] = True
    base_config['use_cumulative_features'] = True
    
    # Set up output directory
    output_dir = Path(output_path) / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor with base config
    preprocessor = DataPreprocessor(base_config)
    
    # Load and preprocess data
    print(f"Loading and preprocessing data for station {station_id}...")
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)
    
    # Generate grid search parameters
    grid_params = setup_grid_search()
    
    # Generate all configurations
    configs = generate_config_combinations(base_config, grid_params)
    
    print(f"Generated {len(configs)} configurations to test")
    
    # Write configuration summary
    with open(output_dir / "grid_search_summary.txt", "w") as f:
        f.write(f"Grid Search Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Total configurations: {len(configs)}\n\n")
        f.write(f"Parameters:\n")
        for param, values in grid_params.items():
            f.write(f"  {param}: {values}\n")
    
    # Results list to track all runs
    results = []
    
    # Run grid search
    print("Starting grid search...")
    
    # Use tqdm for progress tracking
    total_configs = len(configs)
    for i, config in enumerate(tqdm(configs, desc="Testing configurations", unit="model")):
        # Show percentage progress
        progress_pct = (i / total_configs) * 100
        print(f"\nConfiguration {i+1}/{total_configs} ({progress_pct:.1f}% complete)")
        print(f"Testing: {get_config_name(config)}")
        
        # Update base config with grid search parameters while preserving other settings
        current_config = base_config.copy()
        current_config.update(config)
        
        # Run single model with the complete configuration and preprocessor
        result, history, model = run_single_model(
            current_config, train_data, val_data, output_dir, station_id, preprocessor
        )
        
        # Store results
        result_entry = {
            "config_name": get_config_name(config),
            "hidden_size": config['hidden_size'],
            "num_layers": config['num_layers'],
            "sequence_length": config['sequence_length'],
            "learning_rate": config['learning_rate'],
            "objective_function": config['objective_function'],
            "best_val_loss": result["best_val_loss"],
            "rmse": result["rmse"],
            "mae": result["mae"],
            "r2": result["r2"]
        }
        
        # Add peak metrics if available
        if "peak_rmse" in result and not np.isnan(result["peak_rmse"]):
            result_entry["peak_rmse"] = result["peak_rmse"]
            result_entry["peak_mae"] = result["peak_mae"]
        
        if history:
            result_entry["epochs_trained"] = len(history['train_loss'])
            result_entry["final_val_loss"] = history['val_loss'][-1]
        
        results.append(result_entry)
        
        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
    
    # Create final results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by best validation loss
    results_df = results_df.sort_values("best_val_loss")
    
    # Save results
    results_df.to_csv(output_dir / "grid_search_results_final.csv", index=False)
    
    # Generate summary plots
    print("Generating summary plots...")
    generate_summary_plots(results_df, output_dir)
    
    # Create analysis notebook
    print("Creating analysis notebook...")
    create_best_models_notebook(results_df, output_dir, station_id)
    
    # Print best models
    print("\nTop 5 Models:")
    print(results_df.head(5))
    
    return results_df

def generate_summary_plots(results_df, output_dir):
    """
    Generate summary plots of the grid search results.
    
    Args:
        results_df: DataFrame with grid search results
        output_dir: Output directory for plots
    """
    # Create summary plots directory
    summary_dir = output_dir / "summary_plots"
    summary_dir.mkdir(exist_ok=True)
    
    # Ensure all plots directory exists
    all_plots_dir = output_dir / "all_plots"
    all_plots_dir.mkdir(exist_ok=True)
    
    # Create multi-metric comparison plots of the top loss functions
    plt.figure(figsize=(14, 10))
    metrics = ["best_val_loss", "rmse", "mae", "r2"]
    metric_labels = ["Validation Loss", "RMSE", "MAE", "R²"]
    
    # Find best 2 models for each loss function
    top_models = pd.DataFrame()
    for loss_func in results_df['objective_function'].unique():
        df_loss = results_df[results_df['objective_function'] == loss_func]
        # Get top 2 models by validation loss
        top_df = df_loss.nsmallest(2, 'best_val_loss')
        top_models = pd.concat([top_models, top_df])
    
    # Multi-plot comparing different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # For R², higher is better, so negate for consistent sorting
        if metric == "r2":
            sort_col = -top_models[metric]
            top_models_sorted = top_models.iloc[sort_col.argsort()]
            # Also invert the y-axis for consistent visualization (higher is better)
            ax.invert_yaxis()
        else:
            # For other metrics, lower is better
            top_models_sorted = top_models.sort_values(metric)
        
        # Plot the data
        bars = ax.barh(
            range(len(top_models_sorted)), 
            top_models_sorted[metric],
            color=[plt.cm.tab10(i) for i in range(len(top_models_sorted))]
        )
        
        # Add objective function as text
        for j, bar in enumerate(bars):
            obj_func = top_models_sorted.iloc[j]['objective_function']
            # Format the obj_func string to be shorter
            short_name = obj_func.replace('_loss', '').replace('weighted_', 'w_').replace('dynamic_', 'd_')
            # Format the metric value
            if metric == "r2":
                value_text = f"{top_models_sorted.iloc[j][metric]:.3f}"
            else:
                value_text = f"{top_models_sorted.iloc[j][metric]:.5f}"
            
            ax.text(
                bar.get_width() * 1.01, 
                bar.get_y() + bar.get_height()/2,
                f"{short_name}: {value_text}",
                va='center', 
                fontsize=9
            )
        
        # Format x-axis for R²
        if metric == "r2":
            ax.set_xlim(0, 1.0)  # R² range is 0 to 1
            
        ax.set_title(f'Top Models by {label}')
        ax.set_xlabel(label)
        ax.set_ylabel('Model Rank')
        ax.set_yticks(range(len(top_models_sorted)))
        ax.set_yticklabels([f"{i+1}" for i in range(len(top_models_sorted))])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(summary_dir / "multi_metric_comparison.png")
    plt.savefig(all_plots_dir / "multi_metric_comparison.png")
    plt.close()
    
    # Create loss function specific directories for summaries
    for loss_func in results_df['objective_function'].unique():
        loss_func_dir = summary_dir / loss_func
        loss_func_dir.mkdir(exist_ok=True)
        
        # Create a detailed performance table for this loss function
        df_loss = results_df[results_df['objective_function'] == loss_func]
        
        # Sort by validation loss
        df_loss_sorted = df_loss.sort_values('best_val_loss')
        
        # Save detailed CSV for this loss function
        df_loss_sorted.to_csv(loss_func_dir / f"{loss_func}_results.csv")
        # Also save in all_plots directory
        df_loss_sorted.to_csv(all_plots_dir / f"{loss_func}_results.csv")
        
        # Create additional parameter-specific summaries
        params_to_summarize = ['hidden_size', 'num_layers', 'sequence_length', 'learning_rate']
        metrics_to_plot = ['best_val_loss', 'rmse', 'mae']
        
        for param in params_to_summarize:
            # Create a separate plot for each metric
            for metric in metrics_to_plot:
                # Group by parameter and calculate mean/std/min for this metric
                if metric in df_loss.columns:
                    param_summary = df_loss.groupby(param)[metric].agg(['mean', 'std', 'min'])
                    
                    plt.figure(figsize=(10, 6))
                    plt.errorbar(
                        param_summary.index, 
                        param_summary['mean'], 
                        yerr=param_summary['std'],
                        marker='o', 
                        linestyle='-',
                        label=f'Mean ± Std'
                    )
                    
                    # Highlight the best parameter value
                    best_idx = param_summary['min'].idxmin() if metric != 'r2' else param_summary['min'].idxmax()
                    best_val = param_summary.loc[best_idx, 'min']
                    plt.scatter([best_idx], [best_val], 
                               color='red', s=100, label=f'Best: {best_idx}')
                    
                    plt.xlabel(param)
                    metric_label = "Validation Loss" if metric == "best_val_loss" else metric.upper()
                    plt.ylabel(metric_label)
                    plt.title(f'Effect of {param} on {metric_label} ({loss_func})')
                    plt.grid(True)
                    if param == 'learning_rate':
                        plt.xscale('log')
                    plt.legend()
                    plt.tight_layout()
                    
                    # Save in loss function directory
                    plt.savefig(loss_func_dir / f"{param}_{metric}_effect.png")
                    # Also save in all_plots directory
                    plt.savefig(all_plots_dir / f"{loss_func}_{param}_{metric}_effect.png")
                    
                    plt.close()
        
        # Create peak metrics plots if available
        if 'peak_rmse' in df_loss.columns and not df_loss['peak_rmse'].isna().all():
            for param in params_to_summarize:
                param_summary = df_loss.groupby(param)['peak_rmse'].agg(['mean', 'std', 'min'])
                
                plt.figure(figsize=(10, 6))
                plt.errorbar(
                    param_summary.index, 
                    param_summary['mean'], 
                    yerr=param_summary['std'],
                    marker='o', 
                    linestyle='-',
                    label=f'Mean ± Std'
                )
                
                # Highlight the best parameter value
                best_idx = param_summary['min'].idxmin()
                plt.scatter([best_idx], [param_summary.loc[best_idx, 'min']], 
                           color='red', s=100, label=f'Best: {best_idx}')
                
                plt.xlabel(param)
                plt.ylabel('Peak RMSE')
                plt.title(f'Effect of {param} on Peak RMSE ({loss_func})')
                plt.grid(True)
                if param == 'learning_rate':
                    plt.xscale('log')
                plt.legend()
                plt.tight_layout()
                
                # Save in loss function directory
                plt.savefig(loss_func_dir / f"{param}_peak_rmse_effect.png")
                # Also save in all_plots directory
                plt.savefig(all_plots_dir / f"{loss_func}_{param}_peak_rmse_effect.png")
                
                plt.close()
    
    # Plot validation loss by hidden size for each loss function
    plt.figure(figsize=(12, 8))
    for loss_func in results_df['objective_function'].unique():
        df_loss = results_df[results_df['objective_function'] == loss_func]
        for layers in df_loss['num_layers'].unique():
            df_filtered = df_loss[df_loss['num_layers'] == layers]
            
            # Group by hidden size and compute mean
            hidden_sizes = []
            mean_losses = []
            for hidden_size in sorted(df_filtered['hidden_size'].unique()):
                df_size = df_filtered[df_filtered['hidden_size'] == hidden_size]
                hidden_sizes.append(hidden_size)
                mean_losses.append(df_size['best_val_loss'].mean())
            
            plt.plot(hidden_sizes, mean_losses, marker='o', label=f"{loss_func} (layers={layers})")
    
    plt.xlabel('Hidden Size')
    plt.ylabel('Mean Best Validation Loss')
    plt.title('Effect of Hidden Size on Validation Loss by Loss Function and Layers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(summary_dir / "hidden_size_vs_loss.png")
    # Also save in all_plots directory
    plt.savefig(all_plots_dir / "hidden_size_vs_loss.png")
    plt.close()
    
    # Plot RMSE by loss function
    if 'rmse' in results_df.columns:
        plt.figure(figsize=(12, 8))
        for loss_func in results_df['objective_function'].unique():
            df_loss = results_df[results_df['objective_function'] == loss_func]
            
            # Group by sequence length and compute mean RMSE
            seq_lengths = []
            mean_rmse = []
            for seq_len in sorted(df_loss['sequence_length'].unique()):
                df_seq = df_loss[df_loss['sequence_length'] == seq_len]
                seq_lengths.append(seq_len)
                mean_rmse.append(df_seq['rmse'].mean())
            
            plt.plot(seq_lengths, mean_rmse, marker='o', label=loss_func)
        
        plt.xlabel('Sequence Length')
        plt.ylabel('Mean RMSE')
        plt.title('Effect of Sequence Length on RMSE by Loss Function')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(summary_dir / "sequence_length_vs_rmse.png")
        # Also save in all_plots directory
        plt.savefig(all_plots_dir / "sequence_length_vs_rmse.png")
        plt.close()
    
    # Plot validation loss by sequence length for each loss function
    plt.figure(figsize=(12, 8))
    for loss_func in results_df['objective_function'].unique():
        df_loss = results_df[results_df['objective_function'] == loss_func]
        
        # Group by sequence length and compute mean
        seq_lengths = []
        mean_losses = []
        for seq_len in sorted(df_loss['sequence_length'].unique()):
            df_seq = df_loss[df_loss['sequence_length'] == seq_len]
            seq_lengths.append(seq_len)
            mean_losses.append(df_seq['best_val_loss'].mean())
        
        plt.plot(seq_lengths, mean_losses, marker='o', label=loss_func)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Best Validation Loss')
    plt.title('Effect of Sequence Length on Validation Loss by Loss Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(summary_dir / "sequence_length_vs_loss.png")
    # Also save in all_plots directory
    plt.savefig(all_plots_dir / "sequence_length_vs_loss.png")
    plt.close()
    
    # Plot validation loss by learning rate for each loss function
    plt.figure(figsize=(12, 8))
    for loss_func in results_df['objective_function'].unique():
        df_loss = results_df[results_df['objective_function'] == loss_func]
        
        # Group by learning rate and compute mean
        learning_rates = []
        mean_losses = []
        for lr in sorted(df_loss['learning_rate'].unique()):
            df_lr = df_loss[df_loss['learning_rate'] == lr]
            learning_rates.append(lr)
            mean_losses.append(df_lr['best_val_loss'].mean())
        
        plt.plot(learning_rates, mean_losses, marker='o', label=loss_func)
    
    plt.xlabel('Learning Rate')
    plt.xscale('log')  # Log scale for learning rate
    plt.ylabel('Mean Best Validation Loss')
    plt.title('Effect of Learning Rate on Validation Loss by Loss Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(summary_dir / "learning_rate_vs_loss.png")
    # Also save in all_plots directory
    plt.savefig(all_plots_dir / "learning_rate_vs_loss.png")
    plt.close()
    
    # Create a heatmap of validation loss by hidden size and sequence length
    for loss_func in results_df['objective_function'].unique():
        df_loss = results_df[results_df['objective_function'] == loss_func]
        
        # Check if we have enough data for a meaningful heatmap
        if len(df_loss['hidden_size'].unique()) <= 1 or len(df_loss['sequence_length'].unique()) <= 1:
            continue
            
        plt.figure(figsize=(12, 8))
        pivot_table = pd.pivot_table(
            df_loss, 
            values='best_val_loss', 
            index='hidden_size',
            columns='sequence_length',
            aggfunc='mean'
        )
        
        plt.imshow(pivot_table, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Mean Best Validation Loss')
        
        # Add value annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                plt.text(j, i, f"{pivot_table.iloc[i, j]:.4f}", 
                        ha="center", va="center", color="white" if pivot_table.iloc[i, j] > 0.5 else "black")
        
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        plt.xlabel('Sequence Length')
        plt.ylabel('Hidden Size')
        plt.title(f'Validation Loss Heatmap - {loss_func}')
        plt.tight_layout()
        plt.savefig(summary_dir / f"heatmap_{loss_func}.png")
        # Also save in the loss function's directory
        plt.savefig(summary_dir / loss_func / f"heatmap.png")
        # Also save in all_plots directory
        plt.savefig(all_plots_dir / f"heatmap_{loss_func}.png")
        plt.close()
        
        # If RMSE is available, create a heatmap for it too
        if 'rmse' in df_loss.columns:
            plt.figure(figsize=(12, 8))
            rmse_pivot = pd.pivot_table(
                df_loss, 
                values='rmse', 
                index='hidden_size',
                columns='sequence_length',
                aggfunc='mean'
            )
            
            plt.imshow(rmse_pivot, cmap='coolwarm', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Mean RMSE')
            
            # Add value annotations
            for i in range(len(rmse_pivot.index)):
                for j in range(len(rmse_pivot.columns)):
                    plt.text(j, i, f"{rmse_pivot.iloc[i, j]:.4f}", 
                            ha="center", va="center", color="white" if rmse_pivot.iloc[i, j] > 50 else "black")
            
            plt.xticks(range(len(rmse_pivot.columns)), rmse_pivot.columns)
            plt.yticks(range(len(rmse_pivot.index)), rmse_pivot.index)
            plt.xlabel('Sequence Length')
            plt.ylabel('Hidden Size')
            plt.title(f'RMSE Heatmap - {loss_func}')
            plt.tight_layout()
            plt.savefig(summary_dir / f"rmse_heatmap_{loss_func}.png")
            plt.savefig(summary_dir / loss_func / f"rmse_heatmap.png")
            plt.savefig(all_plots_dir / f"rmse_heatmap_{loss_func}.png")
            plt.close()
        
    # Create summary tables
    summary_tables = {}
    
    # Best model for each loss function by validation loss
    best_by_loss = results_df.loc[results_df.groupby('objective_function')['best_val_loss'].idxmin()]
    best_by_loss.to_csv(summary_dir / "best_models_by_loss_function.csv", index=False)
    # Also save in all_plots directory
    best_by_loss.to_csv(all_plots_dir / "best_models_by_loss_function.csv", index=False)
    
    # Best model for each loss function by RMSE if available
    if 'rmse' in results_df.columns:
        best_by_rmse = results_df.loc[results_df.groupby('objective_function')['rmse'].idxmin()]
        best_by_rmse.to_csv(summary_dir / "best_models_by_rmse.csv", index=False)
        best_by_rmse.to_csv(all_plots_dir / "best_models_by_rmse.csv", index=False)
    
    # Best model for each loss function by peak RMSE if available
    if 'peak_rmse' in results_df.columns and not results_df['peak_rmse'].isna().all():
        best_by_peak = results_df.loc[results_df.groupby('objective_function')['peak_rmse'].idxmin()]
        best_by_peak.to_csv(summary_dir / "best_models_by_peak_rmse.csv", index=False)
        best_by_peak.to_csv(all_plots_dir / "best_models_by_peak_rmse.csv", index=False)
    
    # Best model for each hidden size
    best_by_hidden = results_df.loc[results_df.groupby('hidden_size')['best_val_loss'].idxmin()]
    best_by_hidden.to_csv(summary_dir / "best_models_by_hidden_size.csv", index=False)
    # Also save in all_plots directory
    best_by_hidden.to_csv(all_plots_dir / "best_models_by_hidden_size.csv", index=False)
    
    # Best model for each sequence length
    best_by_seq = results_df.loc[results_df.groupby('sequence_length')['best_val_loss'].idxmin()]
    best_by_seq.to_csv(summary_dir / "best_models_by_sequence_length.csv", index=False)
    # Also save in all_plots directory
    best_by_seq.to_csv(all_plots_dir / "best_models_by_sequence_length.csv", index=False)
    
    # Best model overall by validation loss
    best_overall = results_df.loc[results_df['best_val_loss'].idxmin()]
    
    # Best model overall by RMSE if available
    best_by_rmse_overall = None
    if 'rmse' in results_df.columns:
        best_by_rmse_overall = results_df.loc[results_df['rmse'].idxmin()]
    
    # Create a comprehensive summary text file
    summary_text_path = summary_dir / "grid_search_summary.txt"
    with open(summary_text_path, "w") as f:
        f.write("Grid Search Summary\n")
        f.write("=================\n\n")
        
        f.write("Best Model Overall (by validation loss):\n")
        for col, val in best_overall.items():
            f.write(f"  {col}: {val}\n")
        f.write("\n")
        
        if best_by_rmse_overall is not None:
            f.write("Best Model Overall (by RMSE):\n")
            for col, val in best_by_rmse_overall.items():
                f.write(f"  {col}: {val}\n")
            f.write("\n")
        
        f.write("Best Models by Loss Function (validation loss):\n")
        for _, row in best_by_loss.iterrows():
            f.write(f"  {row['objective_function']}: {row['config_name']} (loss: {row['best_val_loss']:.6f})")
            if 'rmse' in row:
                f.write(f", RMSE: {row['rmse']:.4f}")
            if 'mae' in row:
                f.write(f", MAE: {row['mae']:.4f}")
            if 'r2' in row:
                f.write(f", R²: {row['r2']:.4f}")
            f.write("\n")
        f.write("\n")
        
        # If we have RMSE data, add a section for best models by RMSE
        if 'rmse' in results_df.columns:
            f.write("Best Models by Loss Function (RMSE):\n")
            for _, row in best_by_rmse.iterrows():
                f.write(f"  {row['objective_function']}: {row['config_name']} (RMSE: {row['rmse']:.4f}, loss: {row['best_val_loss']:.6f})\n")
            f.write("\n")
        
        # If we have peak RMSE data, add a section for best models by peak RMSE
        if 'peak_rmse' in results_df.columns and not results_df['peak_rmse'].isna().all():
            f.write("Best Models by Loss Function (Peak RMSE):\n")
            for _, row in best_by_peak.iterrows():
                f.write(f"  {row['objective_function']}: {row['config_name']} (Peak RMSE: {row['peak_rmse']:.4f}, loss: {row['best_val_loss']:.6f})\n")
            f.write("\n")
        
        # Averages by various parameters
        f.write("Average Performance by Loss Function:\n")
        avg_by_loss = results_df.groupby('objective_function')['best_val_loss'].mean()
        for loss_func, mean_loss in avg_by_loss.items():
            f.write(f"  {loss_func}: {mean_loss:.6f}")
            if 'rmse' in results_df.columns:
                avg_rmse = results_df[results_df['objective_function'] == loss_func]['rmse'].mean()
                f.write(f", RMSE: {avg_rmse:.4f}")
            if 'mae' in results_df.columns:
                avg_mae = results_df[results_df['objective_function'] == loss_func]['mae'].mean()
                f.write(f", MAE: {avg_mae:.4f}")
            f.write("\n")
        f.write("\n")
        
        f.write("Average Performance by Hidden Size:\n")
        avg_by_hidden = results_df.groupby('hidden_size')['best_val_loss'].mean()
        for hidden, mean_loss in avg_by_hidden.items():
            f.write(f"  {hidden}: {mean_loss:.6f}\n")
        f.write("\n")
        
        f.write("Average Performance by Sequence Length:\n")
        avg_by_seq = results_df.groupby('sequence_length')['best_val_loss'].mean()
        for seq, mean_loss in avg_by_seq.items():
            f.write(f"  {seq}: {mean_loss:.6f}\n")
        f.write("\n")
        
        f.write("Average Performance by Learning Rate:\n")
        avg_by_lr = results_df.groupby('learning_rate')['best_val_loss'].mean()
        for lr, mean_loss in avg_by_lr.items():
            f.write(f"  {lr}: {mean_loss:.6f}\n")
        f.write("\n")
        
        f.write("Average Performance by Number of Layers:\n")
        avg_by_layers = results_df.groupby('num_layers')['best_val_loss'].mean()
        for layers, mean_loss in avg_by_layers.items():
            f.write(f"  {layers}: {mean_loss:.6f}\n")
    
    # Also save a copy in all_plots directory
    import shutil
    shutil.copy(summary_text_path, all_plots_dir / "grid_search_summary.txt")

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent
    output_path = project_root / "results"
    station_id = '21006846'  # Example station ID
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Running grid search for station {station_id}")
    
    # Run grid search
    results_df = run_grid_search(project_root, output_path, station_id)
    
    print("Grid search completed. Results saved to:")
    print(f"  {output_path / 'grid_search' / 'grid_search_results_final.csv'}") 