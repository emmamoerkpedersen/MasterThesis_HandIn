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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    # Set up grid search parameters - this is a sample configuration
    # Modify these parameters based on your specific research needs
    grid_params = {
        'hidden_size': [24, 64, 128, 256],
        'num_layers': [1,2,3],
        'sequence_length': [5000, 10000, 20000, 35000],
        'learning_rate': [0.001, 0.0001],
        'objective_function': ['smoothL1_loss']
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
        
        # --- Calculate Performance Metrics Here ---
        performance_metrics = {
            'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
            'mean_error': float('nan'), 'std_error': float('nan'),
            'peak_mae': float('nan'), 'peak_rmse': float('nan')
        }
        
        # Get original target values from validation data
        target_col = config.get('output_features', ['vst_raw'])[0]
        if target_col in val_data.columns:
            target_series = val_data[target_col]
            
            # Align predictions and targets based on index
            aligned_targets, aligned_predictions = target_series.align(val_predictions_df['vst_raw'], join='inner')
            
            # Ensure we have valid data after alignment
            valid_mask = (~aligned_targets.isna()) & (~aligned_predictions.isna())
            valid_targets = aligned_targets[valid_mask].values
            valid_predictions = aligned_predictions[valid_mask].values
            valid_index = aligned_targets[valid_mask].index
            
            if len(valid_targets) > 0:
                try:
                    # Calculate standard metrics
                    rmse = np.sqrt(mean_squared_error(valid_targets, valid_predictions))
                    mae = mean_absolute_error(valid_targets, valid_predictions)
                    r2 = r2_score(valid_targets, valid_predictions)
                    errors = valid_predictions - valid_targets
                    mean_error = np.mean(errors)
                    std_error = np.std(errors)
                    
                    performance_metrics['rmse'] = rmse
                    performance_metrics['mae'] = mae
                    performance_metrics['r2'] = r2
                    performance_metrics['mean_error'] = mean_error
                    performance_metrics['std_error'] = std_error

                    # Calculate peak metrics (top 10%)
                    peak_threshold = np.percentile(valid_targets, 90)
                    peak_mask = valid_targets >= peak_threshold
                    
                    if np.sum(peak_mask) > 0:
                        peak_mae = mean_absolute_error(
                            valid_targets[peak_mask], 
                            valid_predictions[peak_mask]
                        )
                        peak_rmse = np.sqrt(mean_squared_error(
                            valid_targets[peak_mask], 
                            valid_predictions[peak_mask]
                        ))
                        performance_metrics['peak_mae'] = peak_mae
                        performance_metrics['peak_rmse'] = peak_rmse
                        
                except Exception as e:
                    print(f"Warning: Error calculating metrics: {e}")
            else:
                print("Warning: No overlapping valid targets and predictions found for metric calculation.")
        else:
            print(f"Warning: Target column '{target_col}' not found in validation data for metric calculation.")
        # --- End Metric Calculation ---
        
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
            metrics=performance_metrics,  # Pass metrics to the plotting function
            show_config=True
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
    
    # Find best 2 models for each loss function - these are the most interesting models
    top_models = pd.DataFrame()
    for loss_func in results_df['objective_function'].unique():
        df_loss = results_df[results_df['objective_function'] == loss_func]
        # Get top 2 models by validation loss
        top_df = df_loss.nsmallest(2, 'best_val_loss')
        top_models = pd.concat([top_models, top_df])
    
    # Multi-plot comparing different metrics for the best models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # For R², higher is better, so negate for consistent sorting
        if metric == "r2":
            # Handle NaN values by replacing them with a very low value
            sort_values = -top_models[metric].fillna(-999)
            top_models_sorted = top_models.iloc[sort_values.argsort()]
            # Also invert the y-axis for consistent visualization (higher is better)
            ax.invert_yaxis()
        else:
            # For other metrics, lower is better, handle NaN by using a very high value
            sort_values = top_models[metric].fillna(999999)
            top_models_sorted = top_models.sort_values(metric, na_position='last')
        
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
    
    # Create loss function specific directories for summaries (but with fewer plots)
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
    
    # Only generate key parameter effect plots (most influential parameters)
    
    # 1. Effect of learning rate on validation loss by loss function
    try:
        plt.figure(figsize=(12, 8))
        for loss_func in results_df['objective_function'].unique():
            df_loss = results_df[results_df['objective_function'] == loss_func]
            
            # Group by learning rate and compute mean
            learning_rates = []
            mean_losses = []
            for lr in sorted(df_loss['learning_rate'].unique()):
                df_lr = df_loss[df_loss['learning_rate'] == lr]
                # Skip if there are no valid values
                if df_lr['best_val_loss'].isna().all():
                    continue
                learning_rates.append(lr)
                mean_losses.append(df_lr['best_val_loss'].mean())
            
            # Only plot if we have valid data points
            if learning_rates and mean_losses:
                plt.plot(learning_rates, mean_losses, marker='o', label=loss_func)
        
        plt.xlabel('Learning Rate')
        plt.xscale('log')  # Log scale for learning rate
        plt.ylabel('Mean Best Validation Loss')
        plt.title('Effect of Learning Rate on Validation Loss by Loss Function')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(summary_dir / "learning_rate_vs_loss.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create learning rate plot: {str(e)}")
    
    # 2. Effect of sequence length on validation loss and RMSE
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Validation loss plot
        ax = axes[0]
        for loss_func in results_df['objective_function'].unique():
            df_loss = results_df[results_df['objective_function'] == loss_func]
            
            # Group by sequence length and compute mean
            seq_lengths = []
            mean_losses = []
            for seq_len in sorted(df_loss['sequence_length'].unique()):
                df_seq = df_loss[df_loss['sequence_length'] == seq_len]
                # Skip if there are no valid values
                if df_seq['best_val_loss'].isna().all():
                    continue
                seq_lengths.append(seq_len)
                mean_losses.append(df_seq['best_val_loss'].mean())
            
            # Only plot if we have valid data
            if seq_lengths and mean_losses:
                ax.plot(seq_lengths, mean_losses, marker='o', label=loss_func)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Mean Best Validation Loss')
        ax.set_title('Effect of Sequence Length on Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # RMSE plot (if available)
        ax = axes[1]
        has_valid_rmse = False
        if 'rmse' in results_df.columns:
            for loss_func in results_df['objective_function'].unique():
                df_loss = results_df[results_df['objective_function'] == loss_func]
                
                # Skip if all RMSE values are NaN
                if df_loss['rmse'].isna().all():
                    continue
                
                # Group by sequence length and compute mean RMSE
                seq_lengths = []
                mean_rmse = []
                for seq_len in sorted(df_loss['sequence_length'].unique()):
                    df_seq = df_loss[df_loss['sequence_length'] == seq_len]
                    # Use only non-NaN values
                    valid_rmse = df_seq['rmse'].dropna()
                    if valid_rmse.empty:
                        continue
                    seq_lengths.append(seq_len)
                    mean_rmse.append(valid_rmse.mean())
                
                # Only plot if we have valid data
                if seq_lengths and mean_rmse:
                    ax.plot(seq_lengths, mean_rmse, marker='o', label=loss_func)
                    has_valid_rmse = True
            
            if has_valid_rmse:
                ax.set_xlabel('Sequence Length')
                ax.set_ylabel('Mean RMSE')
                ax.set_title('Effect of Sequence Length on RMSE')
                ax.legend()
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "RMSE data not available or all NaN", 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "RMSE data not available", 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(summary_dir / "sequence_length_effects.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create sequence length effects plot: {str(e)}")
    
    # 3. Effect of hidden size on validation loss and RMSE
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Validation loss plot
        ax = axes[0]
        for loss_func in results_df['objective_function'].unique():
            df_loss = results_df[results_df['objective_function'] == loss_func]
            
            # Group by hidden size and compute mean
            hidden_sizes = []
            mean_losses = []
            for hidden_size in sorted(df_loss['hidden_size'].unique()):
                df_size = df_loss[df_loss['hidden_size'] == hidden_size]
                # Skip if there are no valid values
                if df_size['best_val_loss'].isna().all():
                    continue
                hidden_sizes.append(hidden_size)
                mean_losses.append(df_size['best_val_loss'].mean())
            
            # Only plot if we have valid data
            if hidden_sizes and mean_losses:
                ax.plot(hidden_sizes, mean_losses, marker='o', label=loss_func)
        
        ax.set_xlabel('Hidden Size')
        ax.set_ylabel('Mean Best Validation Loss')
        ax.set_title('Effect of Hidden Size on Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # RMSE plot (if available)
        ax = axes[1]
        has_valid_rmse = False
        if 'rmse' in results_df.columns:
            for loss_func in results_df['objective_function'].unique():
                df_loss = results_df[results_df['objective_function'] == loss_func]
                
                # Skip if all RMSE values are NaN
                if df_loss['rmse'].isna().all():
                    continue
                
                # Group by hidden size and compute mean RMSE
                hidden_sizes = []
                mean_rmse = []
                for hidden_size in sorted(df_loss['hidden_size'].unique()):
                    df_size = df_loss[df_loss['hidden_size'] == hidden_size]
                    # Use only non-NaN values
                    valid_rmse = df_size['rmse'].dropna()
                    if valid_rmse.empty:
                        continue
                    hidden_sizes.append(hidden_size)
                    mean_rmse.append(valid_rmse.mean())
                
                # Only plot if we have valid data
                if hidden_sizes and mean_rmse:
                    ax.plot(hidden_sizes, mean_rmse, marker='o', label=loss_func)
                    has_valid_rmse = True
            
            if has_valid_rmse:
                ax.set_xlabel('Hidden Size')
                ax.set_ylabel('Mean RMSE')
                ax.set_title('Effect of Hidden Size on RMSE')
                ax.legend()
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "RMSE data not available or all NaN", 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "RMSE data not available", 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(summary_dir / "hidden_size_effects.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create hidden size effects plot: {str(e)}")
    
    # 4. Effect of number of layers (if more than one option)
    if len(results_df['num_layers'].unique()) > 1:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Validation loss plot
            ax = axes[0]
            for loss_func in results_df['objective_function'].unique():
                df_loss = results_df[results_df['objective_function'] == loss_func]
                
                # Group by num_layers and compute mean
                num_layers_values = []
                mean_losses = []
                for num_layers in sorted(df_loss['num_layers'].unique()):
                    df_layers = df_loss[df_loss['num_layers'] == num_layers]
                    # Skip if there are no valid values
                    if df_layers['best_val_loss'].isna().all():
                        continue
                    num_layers_values.append(num_layers)
                    mean_losses.append(df_layers['best_val_loss'].mean())
                
                # Only plot if we have valid data
                if num_layers_values and mean_losses:
                    ax.plot(num_layers_values, mean_losses, marker='o', label=loss_func)
            
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('Mean Best Validation Loss')
            ax.set_title('Effect of Number of Layers on Validation Loss')
            ax.legend()
            ax.grid(True)
            
            # RMSE plot (if available)
            ax = axes[1]
            has_valid_rmse = False
            if 'rmse' in results_df.columns:
                for loss_func in results_df['objective_function'].unique():
                    df_loss = results_df[results_df['objective_function'] == loss_func]
                    
                    # Skip if all RMSE values are NaN
                    if df_loss['rmse'].isna().all():
                        continue
                    
                    # Group by num_layers and compute mean RMSE
                    num_layers_values = []
                    mean_rmse = []
                    for num_layers in sorted(df_loss['num_layers'].unique()):
                        df_layers = df_loss[df_loss['num_layers'] == num_layers]
                        # Use only non-NaN values
                        valid_rmse = df_layers['rmse'].dropna()
                        if valid_rmse.empty:
                            continue
                        num_layers_values.append(num_layers)
                        mean_rmse.append(valid_rmse.mean())
                    
                    # Only plot if we have valid data
                    if num_layers_values and mean_rmse:
                        ax.plot(num_layers_values, mean_rmse, marker='o', label=loss_func)
                        has_valid_rmse = True
            
                if has_valid_rmse:
                    ax.set_xlabel('Number of Layers')
                    ax.set_ylabel('Mean RMSE')
                    ax.set_title('Effect of Number of Layers on RMSE')
                    ax.legend()
                    ax.grid(True)
                else:
                    ax.text(0.5, 0.5, "RMSE data not available or all NaN", 
                            ha='center', va='center', fontsize=14, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "RMSE data not available", 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(summary_dir / "num_layers_effects.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create number of layers effects plot: {str(e)}")
    
    # 5. Create a comprehensive heatmap for each loss function if we have enough data points
    for loss_func in results_df['objective_function'].unique():
        try:
            df_loss = results_df[results_df['objective_function'] == loss_func]
            
            # Check if we have enough data for a meaningful heatmap
            if (len(df_loss['hidden_size'].unique()) > 1 and 
                len(df_loss['sequence_length'].unique()) > 1):
                
                # Check if we have enough non-NaN values
                valid_data = df_loss.dropna(subset=['best_val_loss'])
                if len(valid_data) < 4:  # Need at least 4 points for a meaningful heatmap
                    continue
                    
                plt.figure(figsize=(12, 8))
                
                # Filter out NaN values for the pivot table
                pivot_table = pd.pivot_table(
                    valid_data, 
                    values='best_val_loss', 
                    index='hidden_size',
                    columns='sequence_length',
                    aggfunc='mean'
                )
                
                # Skip if pivot table is too sparse
                if pivot_table.count().sum() < 4:
                    continue
                
                plt.imshow(pivot_table, cmap='viridis', aspect='auto', interpolation='nearest')
                plt.colorbar(label='Mean Best Validation Loss')
                
                # Add value annotations
                for i in range(len(pivot_table.index)):
                    for j in range(len(pivot_table.columns)):
                        # Skip NaN values
                        if pd.isna(pivot_table.iloc[i, j]):
                            continue
                        plt.text(j, i, f"{pivot_table.iloc[i, j]:.4f}", 
                                ha="center", va="center", 
                                color="white" if pivot_table.iloc[i, j] > 0.5 else "black")
                
                plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
                plt.yticks(range(len(pivot_table.index)), pivot_table.index)
                plt.xlabel('Sequence Length')
                plt.ylabel('Hidden Size')
                plt.title(f'Validation Loss Heatmap - {loss_func}')
                plt.tight_layout()
                plt.savefig(summary_dir / f"heatmap_{loss_func}.png")
                plt.close()
                
                # If RMSE is available, create a heatmap for it too
                if 'rmse' in df_loss.columns and not df_loss['rmse'].isna().all():
                    # Filter out NaN values
                    valid_rmse_data = df_loss.dropna(subset=['rmse'])
                    
                    # Skip if not enough data points
                    if len(valid_rmse_data) < 4:
                        continue
                        
                    plt.figure(figsize=(12, 8))
                    rmse_pivot = pd.pivot_table(
                        valid_rmse_data, 
                        values='rmse', 
                        index='hidden_size',
                        columns='sequence_length',
                        aggfunc='mean'
                    )
                    
                    # Skip if pivot table is too sparse
                    if rmse_pivot.count().sum() < 4:
                        continue
                    
                    plt.imshow(rmse_pivot, cmap='coolwarm', aspect='auto', interpolation='nearest')
                    plt.colorbar(label='Mean RMSE')
                    
                    # Add value annotations
                    for i in range(len(rmse_pivot.index)):
                        for j in range(len(rmse_pivot.columns)):
                            # Skip NaN values
                            if pd.isna(rmse_pivot.iloc[i, j]):
                                continue
                            plt.text(j, i, f"{rmse_pivot.iloc[i, j]:.1f}", 
                                    ha="center", va="center", 
                                    color="white" if rmse_pivot.iloc[i, j] > 50 else "black")
                    
                    plt.xticks(range(len(rmse_pivot.columns)), rmse_pivot.columns)
                    plt.yticks(range(len(rmse_pivot.index)), rmse_pivot.index)
                    plt.xlabel('Sequence Length')
                    plt.ylabel('Hidden Size')
                    plt.title(f'RMSE Heatmap - {loss_func}')
                    plt.tight_layout()
                    plt.savefig(summary_dir / f"rmse_heatmap_{loss_func}.png")
                    plt.close()
        except Exception as e:
            print(f"Warning: Could not create heatmap for {loss_func}: {str(e)}")
    
    # 6. Bar chart of best models for each loss function
    try:
        plt.figure(figsize=(12, 8))
        best_by_loss = results_df.loc[results_df.groupby('objective_function')['best_val_loss'].idxmin()]
        best_by_loss = best_by_loss.sort_values('best_val_loss')
        
        # Skip this plot if there's no valid data
        if not best_by_loss.empty and not best_by_loss['best_val_loss'].isna().all():
            bars = plt.barh(range(len(best_by_loss)), best_by_loss['best_val_loss'], 
                           color=[plt.cm.tab10(i) for i in range(len(best_by_loss))])
            
            plt.yticks(range(len(best_by_loss)), best_by_loss['objective_function'])
            plt.xlabel('Best Validation Loss')
            plt.title('Best Models by Loss Function')
            
            # Add configuration details
            for i, bar in enumerate(bars):
                config_text = (f"h{best_by_loss.iloc[i]['hidden_size']}, "
                               f"l{best_by_loss.iloc[i]['num_layers']}, "
                               f"s{best_by_loss.iloc[i]['sequence_length']}, "
                               f"lr{best_by_loss.iloc[i]['learning_rate']}")
                plt.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                        f"{best_by_loss.iloc[i]['best_val_loss']:.6f} ({config_text})",
                        va='center')
            
            plt.tight_layout()
            plt.savefig(summary_dir / "best_models_by_loss_function.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create best models bar chart: {str(e)}")
    
    # Create summary tables
    
    # Best model for each loss function by validation loss
    best_by_loss = results_df.loc[results_df.groupby('objective_function')['best_val_loss'].idxmin()]
    best_by_loss.to_csv(summary_dir / "best_models_by_loss_function.csv", index=False)
    
    # Best model for each loss function by RMSE if available
    if 'rmse' in results_df.columns:
        # Filter out rows with NaN RMSE values and then find minimum
        rmse_df = results_df.dropna(subset=['rmse'])
        if not rmse_df.empty:
            best_by_rmse = rmse_df.loc[rmse_df.groupby('objective_function')['rmse'].idxmin()]
            best_by_rmse.to_csv(summary_dir / "best_models_by_rmse.csv", index=False)
    
    # Best model for each loss function by peak RMSE if available
    if 'peak_rmse' in results_df.columns and not results_df['peak_rmse'].isna().all():
        # Filter out rows with NaN peak_rmse values
        peak_df = results_df.dropna(subset=['peak_rmse'])
        if not peak_df.empty:
            best_by_peak = peak_df.loc[peak_df.groupby('objective_function')['peak_rmse'].idxmin()]
            best_by_peak.to_csv(summary_dir / "best_models_by_peak_rmse.csv", index=False)
    
    # Create a comprehensive summary text file with the key findings
    summary_text_path = summary_dir / "grid_search_summary.txt"
    with open(summary_text_path, "w") as f:
        f.write("Grid Search Summary\n")
        f.write("=================\n\n")
        
        f.write("Best Model Overall (by validation loss):\n")
        best_overall = results_df.loc[results_df['best_val_loss'].idxmin()]
        for col, val in best_overall.items():
            f.write(f"  {col}: {val}\n")
        f.write("\n")
        
        if 'rmse' in results_df.columns:
            f.write("Best Model Overall (by RMSE):\n")
            best_by_rmse_overall = results_df.loc[results_df['rmse'].idxmin()]
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
        
        # Parameter influence analysis - calculated by the coefficient of variation
        f.write("Parameter Influence Analysis (higher value = more influential):\n")
        
        # For each parameter, calculate how much variation it introduces in the loss
        params_to_analyze = ['hidden_size', 'num_layers', 'sequence_length', 'learning_rate', 'objective_function']
        influence_scores = {}
        
        for param in params_to_analyze:
            if len(results_df[param].unique()) > 1:  # Only analyze if we have different values
                try:
                    # Group by this parameter and calculate mean and std of validation loss, ignoring NaN values
                    grouped = results_df.groupby(param)['best_val_loss'].agg(['mean', 'std'])
                    # Skip parameters with all NaN values
                    if grouped['mean'].isna().all() or grouped['std'].isna().all():
                        continue
                    # Skip groups with only one non-NaN value
                    valid_groups = grouped.dropna()
                    if len(valid_groups) < 2:
                        continue
                    # Calculate coefficient of variation across different parameter values
                    param_influence = valid_groups['std'].mean() / valid_groups['mean'].mean()
                    influence_scores[param] = param_influence
                    
                    f.write(f"  {param}: {param_influence:.6f}\n")
                except Exception as e:
                    print(f"Warning: Could not calculate influence for {param}: {str(e)}")
        
        # Sort parameters by influence
        if influence_scores:
            most_influential = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
            f.write("\nParameters ranked by influence (most to least):\n")
            for param, score in most_influential:
                f.write(f"  {param}: {score:.6f}\n")
            
            f.write("\nBased on this analysis, focus on tuning these parameters in future experiments.\n")
        else:
            f.write("\nCould not calculate parameter influence due to insufficient or invalid data.\n")
    
    print(f"Created streamlined summary plots in {summary_dir}")
    print(f"See detailed summary at {summary_text_path}")

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