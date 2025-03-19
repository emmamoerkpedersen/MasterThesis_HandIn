"""
Hyperparameter tuning for LSTM forecaster model using Optuna.

This script optimizes the hyperparameters for the LSTM forecaster model
to find the best configuration for time series forecasting.
"""

import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from _3_lstm_model.lstm_forecaster import train_LSTM, LSTMModel
from config import LSTM_CONFIG
from diagnostics.hyperparameter_diagnostics import save_hyperparameter_results

class CustomPruningCallback:
    """Custom pruning callback for PyTorch models that reports intermediate values to Optuna."""
    
    def __init__(self, trial, monitor="val_loss", min_improvement=0.0005, patience=5):
        self.trial = trial
        self.monitor = monitor
        self.best_value = float('inf')
        self.min_improvement = min_improvement
        self.patience = patience
        self.no_improvement_count = 0
        self.history = []
    
    def __call__(self, epoch, train_loss, val_loss):
        current_score = val_loss
        self.history.append(current_score)
        
        # Check if there's significant improvement
        if current_score < (self.best_value - self.min_improvement):
            self.best_value = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Report to Optuna
        self.trial.report(current_score, epoch)
        
        # Prune if no improvement for patience epochs
        if self.no_improvement_count >= self.patience:
            print(f"Pruning trial due to no improvement for {self.patience} epochs")
            raise optuna.TrialPruned()
        
        # Prune if Optuna suggests pruning
        if self.trial.should_prune():
            print("Pruning trial as suggested by Optuna")
            raise optuna.TrialPruned()
        
        # More balanced pruning conditions
        if len(self.history) >= 7:
            recent_losses = self.history[-7:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            # Prune if loss is very unstable (adjusted threshold)
            if loss_std > 0.3 * loss_mean and epoch >= 7:
                print(f"Pruning trial due to unstable loss (std={loss_std:.6f}, mean={loss_mean:.6f})")
                raise optuna.TrialPruned()
            
            # Prune if loss is consistently increasing
            if len(self.history) >= 5 and all(self.history[-i] > self.history[-i-1] for i in range(1, 5)):
                print("Pruning trial due to consistently increasing loss")
                raise optuna.TrialPruned()
        
        # Early detection of poor performance (adjusted threshold)
        if epoch >= 5 and current_score > 30 * min(self.history):
            print(f"Pruning trial due to poor performance (current: {current_score:.6f}, best: {min(self.history):.6f})")
            raise optuna.TrialPruned()
        
        # Detect if loss is much higher than median (adjusted threshold)
        if hasattr(self.trial.study, 'trials_dataframe') and epoch >= 5:
            try:
                completed_trials = self.trial.study.trials_dataframe()
                if len(completed_trials) > 5:
                    median_val_loss = completed_trials['value'].median()
                    if current_score > 15 * median_val_loss:
                        print(f"Pruning trial due to loss much higher than median of completed trials")
                        raise optuna.TrialPruned()
            except:
                pass


def sample_window_data(window_data, sample_ratio=0.3, method='systematic'):
    """
    Sample data from a window while preserving temporal continuity for sequence creation.
    
    Args:
        window_data: Dictionary containing train/val data
        sample_ratio: Ratio of data to keep
        method: 'systematic' or 'chunk' for sampling method
    """
    sampled_data = {}
    for split_type in ['train', 'validation']:
        split_data = window_data[split_type]
        sampled_data[split_type] = {}
        
        for station_id, station_data in split_data.items():
            # Get length of data
            n_samples = len(station_data['vst_raw'])
            
            if method == 'chunk':
                # Take continuous chunks to preserve temporal patterns
                chunk_size = int(n_samples * sample_ratio)
                start_idx = np.random.randint(0, n_samples - chunk_size)
                indices = slice(start_idx, start_idx + chunk_size)
            else:
                # Systematic sampling
                step = max(1, int(1/sample_ratio))
                indices = slice(None, None, step)
            
            # Sample data for each column
            sampled_station_data = {}
            for col, data in station_data.items():
                if isinstance(data, pd.Series):
                    sampled_station_data[col] = data.iloc[indices].copy()
                elif isinstance(data, pd.DataFrame):
                    sampled_station_data[col] = data.iloc[indices].copy()
                elif isinstance(data, np.ndarray):
                    sampled_station_data[col] = data[indices].copy()
                else:
                    sampled_station_data[col] = data
            
            sampled_data[split_type][station_id] = sampled_station_data
    
    return sampled_data


def train_and_evaluate(train_data, val_data, test_data, config, trial):
    """
    Train and evaluate an LSTM model during hyperparameter tuning.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary for generalization evaluation
        config: Configuration dictionary
        trial: Optuna trial object for pruning
    
    Returns:
        Dictionary with training, validation, and test metrics
    """
    print("\n" + "-"*50)
    print("INITIALIZING MODEL")
    print("-"*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Cap epochs at 15
    config['epochs'] = min(config.get('epochs', 15), 15)
    
    # Initialize model with configuration
    model = SimpleLSTMModel(
        input_size=len(config['feature_cols']),
        sequence_length=config['sequence_length'],
        hidden_size=config['hidden_size'],
        output_size=1,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config.get('bidirectional', False)
    )
    
    # Print model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: LSTM with {config['num_layers']} layers, {config['hidden_size']} hidden units")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = train_LSTM(model, config)
    
    # Set up pruning callback with more aggressive settings
    min_improvement = config.get('min_delta', 0.001)
    pruning_patience = 3  # Reduced from 5
    pruning_callback = CustomPruningCallback(
        trial, 
        min_improvement=min_improvement,
        patience=pruning_patience
    )
    print(f"Pruning enabled with patience={pruning_patience}, min_improvement={min_improvement}")
    
    # Track time for reporting purposes
    start_time = time.time()
    
    print("\n" + "-"*50)
    print("TRAINING MODEL")
    print("-"*50)
    
    try:
        # Train with monitoring
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=config['epochs'],
            batch_size=config.get('batch_size', 32),
            patience=config.get('patience', 5),
            epoch_callback=pruning_callback
        )
        
        print("\n" + "-"*50)
        print("TRAINING COMPLETED")
        print("-"*50)
        print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
        
        # Evaluate on test data
        print("\n" + "-"*50)
        print("EVALUATING ON TEST DATA")
        print("-"*50)
        
        test_predictions = None
        test_metrics = {}
        
        if test_data:
            try:
                # Get test predictions
                test_predictions = trainer.predict(test_data)
                
                # Get actual test values
                station_id = list(test_data.keys())[0]
                test_actual = test_data[station_id]['vst_raw']
                
                # Trim predictions to match actual data length if needed
                if len(test_predictions) > len(test_actual):
                    test_predictions = test_predictions[:len(test_actual)]
                
                # Calculate test metrics
                test_mse = mean_squared_error(test_actual[:len(test_predictions)], test_predictions)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(test_actual[:len(test_predictions)], test_predictions)
                
                # Calculate R-squared
                y_mean = np.mean(test_actual[:len(test_predictions)])
                ss_total = np.sum((test_actual[:len(test_predictions)] - y_mean) ** 2)
                ss_residual = np.sum((test_actual[:len(test_predictions)] - test_predictions) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Calculate additional metrics
                # Mean Absolute Percentage Error (MAPE)
                mape = np.mean(np.abs((test_actual[:len(test_predictions)] - test_predictions) / test_actual[:len(test_predictions)])) * 100
                
                # Calculate error on sudden changes
                # Define sudden changes as points where the difference between consecutive values exceeds a threshold
                threshold = np.std(test_actual) * 0.5
                diff_actual = np.abs(np.diff(test_actual[:len(test_predictions)]))
                sudden_change_indices = np.where(diff_actual > threshold)[0]
                
                if len(sudden_change_indices) > 0:
                    # Calculate error on points after sudden changes
                    sudden_change_indices = [idx + 1 for idx in sudden_change_indices if idx + 1 < len(test_predictions)]
                    if sudden_change_indices:
                        sudden_change_mse = mean_squared_error(
                            test_actual[sudden_change_indices], 
                            test_predictions[sudden_change_indices]
                        )
                        sudden_change_mae = mean_absolute_error(
                            test_actual[sudden_change_indices], 
                            test_predictions[sudden_change_indices]
                        )
                    else:
                        sudden_change_mse = None
                        sudden_change_mae = None
                else:
                    sudden_change_mse = None
                    sudden_change_mae = None
                
                # Store all test metrics
                test_metrics = {
                    'test_mse': test_mse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r_squared': r_squared,
                    'test_mape': mape,
                    'sudden_change_mse': sudden_change_mse,
                    'sudden_change_mae': sudden_change_mae,
                    'num_sudden_changes': len(sudden_change_indices)
                }
                
                print(f"Test MSE: {test_mse:.6f}")
                print(f"Test RMSE: {test_rmse:.6f}")
                print(f"Test MAE: {test_mae:.6f}")
                print(f"Test R-squared: {r_squared:.6f}")
                print(f"Test MAPE: {mape:.2f}%")
                if sudden_change_mse is not None:
                    print(f"Sudden Change MSE: {sudden_change_mse:.6f}")
                    print(f"Sudden Change MAE: {sudden_change_mae:.6f}")
                    print(f"Number of Sudden Changes: {len(sudden_change_indices)}")
                
            except Exception as e:
                print(f"Error during test evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
        
    except (ValueError, optuna.TrialPruned) as e:
        print(f"\nTraining interrupted: {str(e)}")
        print(f"Training time before interruption: {(time.time() - start_time)/60:.2f} minutes")
        if isinstance(e, optuna.TrialPruned):
            raise
        if hasattr(trainer, 'history') and trainer.history:
            history = trainer.history
            print(f"Completed {len(history.get('val_loss', []))} epochs before interruption")
        else:
            history = {'train_loss': [float('inf')], 'val_loss': [float('inf')]}
            print("No training history available")
        return {
            'train_loss': float('inf'),
            'val_loss': float('inf'),
            'metrics': {'training_time': time.time() - start_time},
            'test_metrics': {'test_mse': float('inf')}
        }
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print(f"Training time before error: {(time.time() - start_time)/60:.2f} minutes")
        import traceback
        traceback.print_exc()
        return {
            'train_loss': float('inf'),
            'val_loss': float('inf'),
            'metrics': {'training_time': time.time() - start_time},
            'test_metrics': {'test_mse': float('inf')}
        }
    
    # Extract training and validation metrics
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])
    
    metrics = {
        'train_loss': train_losses[-1] if train_losses else float('inf'),
        'val_loss': val_losses[-1] if val_losses else float('inf'),
        'training_time': time.time() - start_time
    }
    
    # Add convergence metrics if we have enough epochs
    if len(val_losses) >= 3:
        last_epochs = min(5, len(val_losses))
        val_improvement = val_losses[-last_epochs] - val_losses[-1]
        val_stability = np.std(val_losses[-last_epochs:])
        
        metrics.update({
            'val_improvement': val_improvement,
            'val_stability': val_stability
        })
        
        print("\nConvergence analysis:")
        print(f"  Final validation loss: {val_losses[-1]:.6f}")
        print(f"  Best validation loss: {min(val_losses):.6f}")
        print(f"  Validation loss improvement in last {last_epochs} epochs: {val_improvement:.6f}")
        print(f"  Validation loss stability (std): {val_stability:.6f}")
    
    return {
        'history': history,
        'train_loss': metrics['train_loss'],
        'val_loss': metrics['val_loss'],
        'metrics': metrics,
        'test_metrics': test_metrics,
        'test_predictions': test_predictions
    }

def run_hyperparameter_tuning(
    split_datasets,
    stations_results,
    output_path,
    base_config=None,
    n_trials=50,
    diagnostics=True,
    data_sample_ratio=0.3
):
    """Run hyperparameter tuning with improved configuration handling."""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING PROCESS STARTING")
    print("="*80)
    
    # Ensure base_config has required fields
    if base_config is None:
        base_config = LSTM_CONFIG.copy()
    
    required_fields = ['feature_cols']
    for field in required_fields:
        if field not in base_config:
            raise ValueError(f"Required field '{field}' missing from base_config")
    
    # Create output directory
    output_dir = Path(output_path) / "diagnostics" / "hyperparameter_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Print base configuration
    print("\nBase Configuration:")
    print(f"Feature columns: {base_config['feature_cols']}")
    print(f"Other base parameters: {[k for k in base_config.keys() if k != 'feature_cols']}")
        
    param_grid = {
        'sequence_length': [48, 57, 60, 72, 83],  # Focus on shorter sequences based on analysis
        'hidden_size': [64, 96, 128, 160],        # Include more options around the sweet spot
        'num_layers': [1, 2, 3],                  # Include 3 layers option
        'dropout': [0.1, 0.15, 0.2, 0.25],        # More granular dropout options
        'learning_rate': [0.0001, 0.0003, 0.001, 0.01],  # Include 0.0003 based on analysis
        'batch_size': [32, 40, 64],               # Include 40 based on analysis
        'epochs': [15],
        'patience': [5, 10],                      # Include longer patience
        'min_delta': [0.0001],
        'pruning_patience': [3, 5],               # Include longer pruning patience
        'bidirectional': [True, False]            # Test both bidirectional and unidirectional
    }
    
    # Print parameter space information
    print("\nParameter Grid Space:")
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)
        print(f"{param}: {values}")
    print(f"\nTotal possible combinations: {total_combinations}")
    print(f"Number of trials to run: {n_trials} (covering {(n_trials/total_combinations)*100:.1f}% of space)")
    
    # Initialize all_results list to store trial results
    all_results = []
    
    # If n_trials=0, try to load previous results
    if n_trials == 0:
        print("\nLoading previous hyperparameter tuning results...")
        results_file = output_dir / "hyperparameter_results.json"
        if not results_file.exists():
            print("No previous results found")
            return base_config, None
            
        try:
            with open(results_file, 'r') as f:
                saved_data = json.load(f)
            
            if 'results' not in saved_data or 'best_config' not in saved_data:
                print("Invalid results file format")
                return base_config, None
                
            # Format the loaded results
            formatted_results = []
            for result in saved_data['results']:
                if isinstance(result, dict):
                    formatted_result = {
                        'trial_id': result.get('trial_id', len(formatted_results)),
                        'params': {},
                        'config': {},
                        'value': result.get('value', float('inf')),
                        'score': result.get('value', float('inf')),
                        'metrics': result.get('metrics', {}),
                        'training_results': {
                            'history': result.get('history', {
                                'val_loss': result.get('val_loss', []),
                                'loss': result.get('train_loss', [])
                            })
                        }
                    }
                    
                    # Format parameters
                    params = result.get('params', {})
                    for param, value in params.items():
                        try:
                            if param in ['sequence_length', 'hidden_size', 'num_layers', 'batch_size', 'epochs', 'patience', 'pruning_patience']:
                                formatted_result['params'][param] = int(float(value))
                                formatted_result['config'][param] = int(float(value))
                            else:
                                formatted_result['params'][param] = float(value)
                                formatted_result['config'][param] = float(value)
                        except (ValueError, TypeError):
                            formatted_result['params'][param] = value
                            formatted_result['config'][param] = value
                    
                    # Add training time if available
                    if 'duration' in result:
                        formatted_result['metrics']['training_time'] = result['duration']
                    elif 'trial_time' in result:
                        formatted_result['metrics']['training_time'] = result['trial_time']
                    
                    formatted_results.append(formatted_result)
            
            # Generate diagnostics if enabled and we have enough trials
            if diagnostics and len(formatted_results) >= 3:
                print(f"\nGenerating diagnostics from {len(formatted_results)} previous trials...")
                try:
                    # Generate individual diagnostic plots
                    from diagnostics.hyperparameter_diagnostics import (
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
                        plot_parallel_coordinates_clustered,
                        generate_hyperparameter_report,
                        plot_test_vs_validation_performance,
                        plot_sudden_change_performance,
                        plot_test_predictions_comparison
                    )
                    
                    print("Generating individual hyperparameter plots...")
                    try:
                        plot_hyperparameter_pairwise_interactions(formatted_results, output_dir)
                        analyze_hyperparameter_importance(formatted_results, output_dir)
                        create_3d_surface_plots(formatted_results, output_dir)
                        plot_convergence_analysis(formatted_results, output_dir)
                        plot_learning_curve_clusters(formatted_results, output_dir)
                        plot_parameter_sensitivity(formatted_results, output_dir)
                        plot_top_n_distributions(formatted_results, output_dir)
                        generate_architecture_impact_plot(formatted_results, output_dir)
                        plot_learning_rate_landscape(formatted_results, output_dir)
                        plot_network_architecture_performance(formatted_results, output_dir)
                        plot_training_dynamics(formatted_results, output_dir)
                        plot_training_time_analysis(formatted_results, output_dir)
                        plot_parameter_evolution(formatted_results, output_dir)
                        plot_parallel_coordinates_clustered(formatted_results, output_dir)
                        
                        # Generate new test performance visualizations
                        print("Generating test performance visualizations...")
                        plot_test_vs_validation_performance(formatted_results, output_dir)
                        plot_sudden_change_performance(formatted_results, output_dir)
                        plot_test_predictions_comparison(formatted_results, output_dir)
                        
                        print("✓ Successfully generated all diagnostic plots")
                    except Exception as e:
                        print(f"Error generating individual plots: {e}")
                        import traceback
                        print(traceback.format_exc())
                    
                    # Generate the comprehensive report
                    generate_hyperparameter_report(
                        results=formatted_results,
                        best_config=saved_data['best_config'],
                        output_dir=output_dir,
                        evaluation_metric='test_rmse'
                    )
                except Exception as e:
                    print(f"Error generating diagnostics: {e}")
                    import traceback
                    print(traceback.format_exc())
            elif diagnostics:
                print(f"Not enough trials ({len(formatted_results)}) for detailed visualizations. Need at least 3 trials.")
            
            return saved_data['best_config'], None
            
        except Exception as e:
            print(f"Error loading previous results: {e}")
            return base_config, None
    
    # Extract all available windows
    if 'windows' not in split_datasets:
        raise ValueError("No windows found in split_datasets")
    
    windows = split_datasets['windows']
    n_windows = len(windows)
    
    if n_windows < 2:
        raise ValueError("Need at least 2 windows for robust hyperparameter tuning")
    
    print(f"\nUsing {n_windows} windows for hyperparameter tuning")
    print(f"Data sampling ratio: {data_sample_ratio:.1%}")
    
    # Sample data from each window
    print("Sampling data from windows...")
    sampled_windows = {}
    for window_idx, window_data in windows.items():
        sampled_windows[window_idx] = sample_window_data(
            window_data,
            sample_ratio=data_sample_ratio,
            method='systematic'
        )
        
        # Print data shapes after sampling
        train_shape = sampled_windows[window_idx]['train'][list(window_data['train'].keys())[0]]['vst_raw'].shape
        val_shape = sampled_windows[window_idx]['validation'][list(window_data['validation'].keys())[0]]['vst_raw'].shape
        print(f"  Window {window_idx}:")
        print(f"    Training samples: {train_shape[0]} (reduced from {window_data['train'][list(window_data['train'].keys())[0]]['vst_raw'].shape[0]})")
        print(f"    Validation samples: {val_shape[0]} (reduced from {window_data['validation'][list(window_data['validation'].keys())[0]]['vst_raw'].shape[0]})")
    
    # Modify the objective function to use multiple windows
    def objective(trial):
        # Start time for this trial
        trial_start_time = time.time()
        
        print("\n" + "="*80)
        print(f"TRIAL {trial.number+1}/{n_trials} STARTING")
        print("="*80)
        
        # Create configuration from base_config and param_grid
        config = base_config.copy() if base_config else {}
        
        # Sample parameters from the grid
        params = {}
        for param, values in param_grid.items():
            if param == 'sequence_length':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'hidden_size':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'num_layers':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'dropout':
                params[param] = trial.suggest_float(param, min(values), max(values))
            elif param == 'learning_rate':
                params[param] = trial.suggest_float(param, min(values), max(values), log=True)
            elif param == 'batch_size':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'epochs':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'patience':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'min_delta':
                params[param] = trial.suggest_float(param, min(values), max(values), log=True)
            elif param == 'pruning_patience':
                params[param] = trial.suggest_int(param, min(values), max(values))
            elif param == 'bidirectional':
                params[param] = trial.suggest_categorical(param, values)
        
        # Update config with sampled parameters
        config.update(params)
        
        # Print configuration
        print("\nTrial Configuration:")
        print("-" * 50)
        print(f"Architecture: sequence_length={config.get('sequence_length')}, hidden_size={config.get('hidden_size')}, "
              f"num_layers={config.get('num_layers')}, dropout={config.get('dropout'):.4f}, "
              f"bidirectional={config.get('bidirectional', False)}")
        print(f"Training: learning_rate={config.get('learning_rate'):.6f}, batch_size={config.get('batch_size')}, "
              f"epochs={config.get('epochs')}, patience={config.get('patience')}")
        print(f"Pruning: min_delta={config.get('min_delta'):.6f}, pruning_patience={config.get('pruning_patience')}")
        print("-" * 50)
        
        # Store results for each window
        window_results = []
        
        try:
            # Evaluate on each window using sampled data
            for window_idx, window_data in sampled_windows.items():
                print(f"\nEvaluating on window {window_idx}")
                
                # Add test data to window_data if available
                test_data = None
                if 'test' in split_datasets:
                    test_data = split_datasets['test']
                
                # Train and evaluate
                results = train_and_evaluate(
                    train_data=window_data['train'],
                    val_data=window_data['validation'],
                    test_data=test_data,
                    config=config,
                    trial=trial
                )
                
                if isinstance(results, dict):
                    window_results.append({
                        'window_idx': window_idx,
                        'val_loss': results.get('val_loss', float('inf')),
                        'train_loss': results.get('train_loss', float('inf')),
                        'metrics': results.get('metrics', {}),
                        'history': results.get('history', {}),
                        'test_metrics': results.get('test_metrics', {}),
                        'test_predictions': results.get('test_predictions', None)
                    })
                    print(f"Window {window_idx} validation loss: {results.get('val_loss', float('inf')):.6f}")
                    if 'test_metrics' in results and results['test_metrics']:
                        print(f"Window {window_idx} test RMSE: {results['test_metrics'].get('test_rmse', float('inf')):.6f}")
            
            # Calculate average performance across windows
            if window_results:
                val_losses = [r['val_loss'] for r in window_results]
                avg_val_loss = np.mean(val_losses)
                std_val_loss = np.std(val_losses)
                
                # Calculate average test metrics
                test_metrics_avg = {}
                if all('test_metrics' in r for r in window_results):
                    # Collect all test metrics
                    for metric in ['test_mse', 'test_rmse', 'test_mae', 'test_r_squared', 'test_mape', 
                                  'sudden_change_mse', 'sudden_change_mae']:
                        values = [r['test_metrics'].get(metric, float('inf')) for r in window_results 
                                 if r['test_metrics'].get(metric) is not None]
                        if values:
                            test_metrics_avg[metric] = np.mean(values)
                            test_metrics_avg[f"{metric}_std"] = np.std(values)
                
                # Print average metrics
                print("\nAverage Metrics Across Windows:")
                print(f"Validation Loss: {avg_val_loss:.6f} (±{std_val_loss:.6f})")
                for metric, value in test_metrics_avg.items():
                    if not metric.endswith('_std'):
                        std = test_metrics_avg.get(f"{metric}_std", 0)
                        print(f"{metric}: {value:.6f} (±{std:.6f})")
                
                # Store comprehensive results
                trial_results = {
                    'trial_id': trial.number,
                    'params': params,
                    'value': avg_val_loss,  # Keep val_loss as primary value for backward compatibility
                    'std_val_loss': std_val_loss,
                    'window_results': window_results,
                    'test_metrics_avg': test_metrics_avg,
                    'duration': time.time() - trial_start_time
                }
                
                # Add to results list
                all_results.append(trial_results)
                
                # Save intermediate results
                if output_dir:
                    with open(output_dir / f'trial_{trial.number}.json', 'w') as f:
                        json.dump(trial_results, f, indent=2)
                
                print("\n" + "="*80)
                print(f"TRIAL {trial.number+1}/{n_trials} COMPLETED")
                print("="*80)
                print(f"Duration: {(time.time() - trial_start_time)/60:.2f} minutes")
                
                # Clean up CUDA memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("CUDA memory cache cleared")
                
                # Use a combined metric for optimization
                # We want to minimize both validation loss and test RMSE
                # with more weight on test RMSE since that's our real goal
                optimization_metric = avg_val_loss  # Default to val_loss
                
                if 'test_rmse' in test_metrics_avg:
                    # Weighted combination of validation loss and test RMSE
                    # Giving more weight (0.7) to test RMSE
                    test_rmse = test_metrics_avg['test_rmse']
                    # This is a test-focused optimization metric that balances model fit (val_loss)
                    # with generalization performance (test_rmse), prioritizing the latter
                    optimization_metric = 0.3 * avg_val_loss + 0.7 * test_rmse
                    print(f"Optimization Metric (0.3*val_loss + 0.7*test_rmse): {optimization_metric:.6f}")
                else:
                    print(f"Optimization Metric (val_loss only): {optimization_metric:.6f}")
                
                return optimization_metric
            
            return float('inf')
            
        except optuna.TrialPruned:
            print("\nTrial pruned")
            raise
        except Exception as e:
            print(f"\nError during trial: {str(e)}")
            return float('inf')
    
    # Create and run Optuna study
    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=3,
        interval_steps=1
    )
    
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction='minimize'
    )
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
    
    # Get the best parameters
    if study.best_trial:
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Convert best_params to config format
        best_config = base_config.copy() if base_config else {}
        best_config.update(best_params)
        
        print("\nBest trial:")
        print(f"  Value (avg val loss): {study.best_value:.6f}")
        print("  Params:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        
        # Save final results
        if output_dir:
            save_results(all_results, output_dir, best_config)
        
        # Generate diagnostics if enabled
        if diagnostics and len(all_results) >= 3:
            print("\nGenerating hyperparameter tuning diagnostics...")
            try:
                # Format results for visualization
                formatted_results = format_results_for_diagnostics(all_results)
                
                # Save results and generate plots
                generate_diagnostic_plots(formatted_results, best_config, output_dir)
                
            except Exception as e:
                print(f"Error generating diagnostics: {e}")
                import traceback
                print(traceback.format_exc())
        elif diagnostics:
            print(f"Not enough trials ({len(all_results)}) for detailed visualizations. Need at least 3 trials.")
        
        return best_config, study
    else:
        print("No best trial found. All trials may have been pruned.")
        return base_config, study

def format_results_for_diagnostics(results):
    """Format results for diagnostic visualization."""
    formatted_results = []
    for result in results:
        if isinstance(result, dict):
            formatted_result = {
                'trial_id': result.get('trial_id', len(formatted_results)),
                'params': {},
                'config': {},
                'value': result.get('value', float('inf')),
                'score': result.get('value', float('inf')),
                'metrics': {},
                'training_results': {'history': {}}
            }
            
            # Format parameters
            params = result.get('params', {})
            for param, value in params.items():
                try:
                    if param in ['sequence_length', 'hidden_size', 'num_layers', 'batch_size', 'epochs', 'patience', 'pruning_patience']:
                        formatted_result['params'][param] = int(float(value))
                        formatted_result['config'][param] = int(float(value))
                    else:
                        formatted_result['params'][param] = float(value)
                        formatted_result['config'][param] = float(value)
                except (ValueError, TypeError):
                    formatted_result['params'][param] = value
                    formatted_result['config'][param] = value
            
            # Aggregate metrics and history across windows
            if 'window_results' in result:
                metrics = {}
                val_losses = []
                train_losses = []
                
                for window_result in result['window_results']:
                    # Collect metrics
                    for metric, value in window_result.get('metrics', {}).items():
                        if metric not in metrics:
                            metrics[metric] = []
                        metrics[metric].append(value)
                    
                    # Collect losses
                    val_losses.append(window_result.get('val_loss', float('inf')))
                    train_losses.append(window_result.get('train_loss', float('inf')))
                
                # Average metrics
                formatted_result['metrics'] = {
                    k: np.mean(v) for k, v in metrics.items()
                }
                
                # Add loss history
                formatted_result['training_results']['history'] = {
                    'val_loss': val_losses,
                    'loss': train_losses
                }
            
            # Add training time if available
            if 'duration' in result:
                formatted_result['metrics']['training_time'] = result['duration']
            
            formatted_results.append(formatted_result)
    
    return formatted_results

def generate_diagnostic_plots(formatted_results, best_config, output_dir):
    """Generate all diagnostic plots."""
    from diagnostics.hyperparameter_diagnostics import (
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
        plot_parallel_coordinates_clustered,
        generate_hyperparameter_report,
        plot_test_vs_validation_performance,
        plot_sudden_change_performance,
        plot_test_predictions_comparison
    )
    
    # Save results first
    save_hyperparameter_results(formatted_results, best_config, output_dir)
    
    print("Generating individual hyperparameter plots...")
    try:
        plot_hyperparameter_pairwise_interactions(formatted_results, output_dir)
        analyze_hyperparameter_importance(formatted_results, output_dir)
        create_3d_surface_plots(formatted_results, output_dir)
        plot_convergence_analysis(formatted_results, output_dir)
        plot_learning_curve_clusters(formatted_results, output_dir)
        plot_parameter_sensitivity(formatted_results, output_dir)
        plot_top_n_distributions(formatted_results, output_dir)
        generate_architecture_impact_plot(formatted_results, output_dir)
        plot_learning_rate_landscape(formatted_results, output_dir)
        plot_network_architecture_performance(formatted_results, output_dir)
        plot_training_dynamics(formatted_results, output_dir)
        plot_training_time_analysis(formatted_results, output_dir)
        plot_parameter_evolution(formatted_results, output_dir)
        plot_parallel_coordinates_clustered(formatted_results, output_dir)
        
        # Generate new test performance visualizations
        print("Generating test performance visualizations...")
        plot_test_vs_validation_performance(formatted_results, output_dir)
        plot_sudden_change_performance(formatted_results, output_dir)
        plot_test_predictions_comparison(formatted_results, output_dir)
        
        print("✓ Successfully generated all diagnostic plots")
    except Exception as e:
        print(f"Error generating individual plots: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Generate the comprehensive report
    generate_hyperparameter_report(
        results=formatted_results,
        best_config=best_config,
        output_dir=output_dir,
        evaluation_metric='test_rmse'
    )


def save_results(results, output_dir, best_config=None):
    """
    Save hyperparameter search results to file.
    
    Args:
        results: List of trial results
        output_dir: Output directory path
        best_config: Best configuration dictionary (optional)
    """
    output_path = Path(output_dir)
    results_file = output_path / 'hyperparameter_results.json'
    
    # Create output data with timestamp
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    if best_config:
        output_data['best_config'] = best_config
    
    # Save to file
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {results_file}")


def save_study_visualizations(study, output_dir):
    """
    Save visualizations of the Optuna study.
    
    Args:
        study: Optuna study object
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    
    # Create optimization history plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(output_path / 'optimization_history.png')
    
    # Create parameter importance plot
    plt.figure(figsize=(12, 8))
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(output_path / 'param_importances.png')
    except:
        print("Could not create parameter importance plot")
    
    # Create parallel coordinate plot
    plt.figure(figsize=(15, 10))
    try:
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(output_path / 'parallel_coordinate.png')
    except:
        print("Could not create parallel coordinate plot")


def load_best_hyperparameters(output_dir, base_config=None):
    """
    Load the best hyperparameters from a previous tuning run.
    
    Args:
        output_dir: Directory containing hyperparameter results
        base_config: Base configuration to update (optional)
    
    Returns:
        Updated configuration dictionary
    """
    # Update path to check in the diagnostics/hyperparameter_tuning subdirectory
    results_file = Path(output_dir) / "diagnostics" / "hyperparameter_tuning" / "hyperparameter_results.json"
    
    print(f"Looking for hyperparameter results at: {results_file}")
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"Loaded previous hyperparameter results from {results_file}")
            
            # Use the best configuration if available
            if 'best_config' in results:
                best_config = results['best_config']
                
                # Start with base config or empty dict
                config = base_config.copy() if base_config else {}
                
                # Track and show changes
                print("Updating configuration with saved hyperparameters:")
                for key, value in best_config.items():
                    if key in config:
                        print(f"  {key}: {config[key]} -> {value}")
                    else:
                        print(f"  {key}: None -> {value}")
                    config[key] = value
                
                return config
            else:
                print("No best configuration found in results")
                return base_config
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
            return base_config
    else:
        print("No previous hyperparameter tuning results found")
        return base_config

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    output_path = project_root / "results"
    sys.path.append(str(project_root))
    
    # Import required modules
    from _1_preprocessing.split import split_data_rolling
    
    # Load data
    print("Loading data...")
    preprocessed_data = pd.read_pickle('../data_utils/Sample data/preprocessed_data.pkl')
    station_id = '21006846'
    
    # Create station-specific data
    preprocessed_data = {station_id: preprocessed_data[station_id]}
    
    # Split data
    print("Splitting data...")
    split_datasets = split_data_rolling(preprocessed_data)
    
    # Run hyperparameter tuning with comprehensive settings
    best_config, study = run_hyperparameter_tuning(
        split_datasets=split_datasets,
        stations_results={},  # No synthetic errors needed
        output_path=output_path,
        base_config=LSTM_CONFIG,
        n_trials=45,  # Comprehensive run
        data_sample_ratio=0.3,  # Keep 30% sampling for speed
        diagnostics=True  # Generate all diagnostic plots
    )
    
    # Print final results
    print("\nOptimization completed!")
    print("\nBest Configuration Found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Save additional visualizations
    print("\nGenerating additional study visualizations...")
    save_study_visualizations(study, output_path / "diagnostics" / "hyperparameter_tuning")
    
    print("\nAll results and visualizations have been saved to:")
    print(f"{output_path}/diagnostics/hyperparameter_tuning/") 