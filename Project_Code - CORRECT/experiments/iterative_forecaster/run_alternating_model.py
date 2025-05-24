import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import local modules
from experiments.iterative_forecaster.alternating_config import ALTERNATING_CONFIG
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence
from _4_anomaly_detection.z_score import calculate_z_scores_mad
from _4_anomaly_detection.anomaly_visualization import plot_water_level_anomalies
from experiments.iterative_forecaster.alternating_trainer import AlternatingTrainer
from experiments.iterative_forecaster.alternating_forecast_model import AlternatingForecastModel

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run alternating LSTM model for water level forecasting')
    parser.add_argument('--station_id', type=str, default='21006846', help='Station ID to process')
    parser.add_argument('--error_multiplier', type=float, default=None, 
                      help='Error multiplier for synthetic errors. If not provided, no errors are injected.')
    parser.add_argument('--quick_mode', action='store_true', help='Enable quick mode with reduced data (3 years training, 1 year validation)')
    parser.add_argument('--error_type', type=str, default='both', choices=['both', 'train', 'validation', 'none'],
                      help='Which datasets to inject errors into (both, train, validation, or none)')
    parser.add_argument('--experiment', type=str, default='0', help='Experiment number/name for organizing results (e.g., 0, 1, baseline, etc.)')
    return parser.parse_args()

def setup_experiment_directories(project_dir, experiment_name):
    """
    Create experiment-specific directory structure.
    
    Args:
        project_dir: Project root directory
        experiment_name: Name/number of the experiment
        
    Returns:
        dict: Dictionary with paths to experiment directories
    """
    # Base experiment directory under Iterative model results
    exp_dir = Path(project_dir) / "results" / "Iterative model results" / f"experiment_{experiment_name}"
    
    # Create subdirectories
    directories = {
        'base': exp_dir,
        'diagnostics': exp_dir / "diagnostics",
        'visualizations': exp_dir / "visualizations",
        'anomaly_detection': exp_dir / "anomaly_detection",
        'behavior_analysis': exp_dir / "visualizations" / "alternating_behavior"
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment directories created under: {exp_dir}")
    return directories

def generate_alternating_predictions(model, val_data, trainer, config):
    """
    Generate predictions for validation data using the alternating pattern 
    (alternating between using original inputs and model's own predictions).
    
    Args:
        model: Trained AlternatingForecastModel instance
        val_data: Validation DataFrame
        trainer: AlternatingTrainer instance
        config: Model configuration with week_steps
        
    Returns:
        DataFrame with original and alternating predictions
    """
    print("\nGenerating alternating predictions for validation data...")
    
    # Get week_steps from config
    week_steps = config.get('week_steps', 672)  # Default: 672 steps (1 week of 15-min data)
    
    # Prepare validation data for the model
    x_val, y_val = trainer.prepare_sequences(val_data, is_training=False)
    
    # Move to model's device
    device = next(model.parameters()).device
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate standard predictions (without alternating)
    with torch.no_grad():
        # Reset states
        hidden_state, cell_state = None, None
        
        # Get predictions using original data
        standard_outputs, _, _ = model(
            x_val, 
            hidden_state, 
            cell_state,
            use_predictions=False  # Use original data as input
        )
    
    # Generate alternating predictions
    with torch.no_grad():
        # Reset states
        hidden_state, cell_state = None, None
        
        # Get predictions with alternating pattern
        alternating_outputs, _, _ = model(
            x_val, 
            hidden_state, 
            cell_state,
            use_predictions=True,  # Enable using model's own predictions
            alternating_weeks=True  # Use alternating week pattern
        )
    
    # Convert predictions back to original scale
    standard_pred_np = standard_outputs.cpu().numpy()
    alternating_pred_np = alternating_outputs.cpu().numpy()
    
    # Handle dimensionality if needed
    if standard_pred_np.ndim == 3:
        standard_pred_np = standard_pred_np.reshape(standard_pred_np.shape[0] * standard_pred_np.shape[1], -1)
        alternating_pred_np = alternating_pred_np.reshape(alternating_pred_np.shape[0] * alternating_pred_np.shape[1], -1)
    
    # Apply inverse transform
    standard_pred_original = trainer.target_scaler.inverse_transform(standard_pred_np).flatten()
    alternating_pred_original = trainer.target_scaler.inverse_transform(alternating_pred_np).flatten()
    
    # Create DataFrames with predictions
    standard_pred_df = pd.DataFrame({
        'vst_raw': standard_pred_original,
    }, index=val_data.index[:len(standard_pred_original)])
    
    alternating_pred_df = pd.DataFrame({
        'vst_raw': alternating_pred_original,
    }, index=val_data.index[:len(alternating_pred_original)])
    
    print("Successfully generated standard and alternating predictions")
    
    return standard_pred_df, alternating_pred_df

def generate_behavior_visualizations(val_data, predictions, station_id, config, output_path, model=None, trainer=None):
    """
    Generate visualizations specific to the alternating forecaster's behavior.
    
    Args:
        val_data: Validation DataFrame
        predictions: Predicted values (Series or array)
        station_id: Station identifier
        config: Model configuration
        output_path: Output directory path
        model: AlternatingForecastModel instance (needed for hidden state visualization)
        trainer: AlternatingTrainer instance (needed for generating alternating predictions)
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    try:
        from experiments.iterative_forecaster.alternating_visualization import (
            plot_alternating_pattern_performance,
            plot_error_accumulation,
            plot_recovery_after_transition,
            plot_input_flag_impact,
            plot_hidden_state_evolution
        )
        
        # Create output directories
        vis_dir = Path(output_path) / "visualizations" / "alternating_behavior"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating behavior visualizations for alternating model...")
        
        # Generate visualizations
        week_steps = config.get('week_steps', 672)  # Default to 672 (1 week of 15-min intervals)
        
        # Generate alternating predictions if both model and trainer are provided
        if model is not None and trainer is not None:
            try:
                print("\nGenerating explicit alternating predictions for visualization...")
                standard_pred_df, alternating_pred_df = generate_alternating_predictions(
                    model=model,
                    val_data=val_data,
                    trainer=trainer,
                    config=config
                )
                
                # Use alternating predictions for behavior analysis
                behavior_predictions = alternating_pred_df['vst_raw']
                
                # Also create comparative visualization
                plot_comparison_path = vis_dir / f'prediction_comparison_{station_id}.png'
                plt.figure(figsize=(14, 8))
                
                # Plot the actual values
                plt.plot(val_data.index, val_data['vst_raw'], 'b-', label='Actual', linewidth=1.5, alpha=0.8)
                
                # Plot standard predictions
                plt.plot(standard_pred_df.index, standard_pred_df['vst_raw'], 'g-', 
                       label='Standard Predictions', linewidth=1.5, alpha=0.8)
                
                # Plot alternating predictions
                plt.plot(alternating_pred_df.index, alternating_pred_df['vst_raw'], 'r-', 
                       label='Alternating Predictions', linewidth=1.5, alpha=0.8)
                
                # Shade the background differently for original vs prediction weeks
                week_change_idx = [i * week_steps for i in range(1, (len(val_data) // week_steps) + 1)]
                week_start_idx = 0
                
                for i, change_idx in enumerate(week_change_idx):
                    if change_idx <= len(val_data):
                        # Determine if this is an original or prediction-based week
                        is_original = (i % 2 == 0)
                        color = 'lightblue' if is_original else 'lightsalmon'
                        alpha = 0.2
                        label = 'Original Data Period' if is_original else 'Prediction-based Period'
                        
                        # Only add label for the first occurrence of each type
                        if i < 2:  # Only add labels for the first two weeks
                            plt.axvspan(val_data.index[week_start_idx], val_data.index[min(change_idx-1, len(val_data)-1)], 
                                    alpha=alpha, color=color, label=label)
                        else:
                            plt.axvspan(val_data.index[week_start_idx], val_data.index[min(change_idx-1, len(val_data)-1)], 
                                    alpha=alpha, color=color)
                        
                        week_start_idx = change_idx
                
                plt.title(f'Standard vs Alternating Prediction Comparison - Station {station_id}', fontweight='bold')
                plt.ylabel('Water Level [mm]', fontweight='bold')
                plt.xlabel('Date', fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(plot_comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved prediction comparison plot to: {plot_comparison_path}")
                
            except Exception as e:
                print(f"Error generating alternating predictions: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Fallback to standard predictions if alternating generation fails
                behavior_predictions = predictions
        else:
            # Use standard predictions if model or trainer not provided
            behavior_predictions = predictions
        
        # Plot alternating pattern performance
        pattern_perf_path = plot_alternating_pattern_performance(
            test_data=val_data,
            predictions=behavior_predictions,
            week_steps=week_steps,
            output_dir=vis_dir,
            station_id=station_id
        )
        
        # Plot error accumulation
        error_accum_path = plot_error_accumulation(
            test_data=val_data,
            predictions=behavior_predictions,
            week_steps=week_steps,
            output_dir=vis_dir,
            station_id=station_id
        )
        
        # Plot recovery after transitions
        recovery_path = plot_recovery_after_transition(
            test_data=val_data,
            predictions=behavior_predictions,
            week_steps=week_steps,
            output_dir=vis_dir,
            station_id=station_id,
            transition_window=48  # 12 hours before/after transitions (at 15-min intervals)
        )
        
        # Plot input flag impact
        flag_impact_path = plot_input_flag_impact(
            test_data=val_data,
            predictions=behavior_predictions,
            week_steps=week_steps,
            output_dir=vis_dir,
            station_id=station_id
        )
        
        # Initialize hidden state path as None
        hidden_state_path = None
        
        # Plot hidden state evolution if model is provided
        if model is not None:
            try:
                hidden_state_path = plot_hidden_state_evolution(
                    model=model,
                    test_data=val_data,
                    week_steps=week_steps,
                    output_dir=vis_dir,
                    station_id=station_id,
                    num_states_to_show=5  # Show top 5 dimensions by variance
                )
            except Exception as e:
                print(f"Error generating hidden state visualization: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Return paths to all visualizations
        vis_paths = {
            'alternating_pattern': pattern_perf_path,
            'error_accumulation': error_accum_path,
            'recovery_pattern': recovery_path,
            'input_flag_impact': flag_impact_path
        }
        
        if hidden_state_path:
            vis_paths['hidden_state_evolution'] = hidden_state_path
        
        print(f"Behavior visualizations saved to: {vis_dir}")
        return vis_paths
        
    except Exception as e:
        print(f"Error generating behavior visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def run_alternating_model(args):
    """Run the alternating LSTM model with the specified parameters."""
    # Setup experiment directories
    exp_dirs = setup_experiment_directories(project_dir, args.experiment)
    
    # Update configuration with command line arguments
    config = ALTERNATING_CONFIG.copy()
    config.update({
        'quick_mode': args.quick_mode,
    })
    
    # Ensure the config has feature_stations entry (needed by DataPreprocessor)
    if 'feature_stations' not in config:
        config['feature_stations'] = []
    
    # Print configuration
    print("\nModel Configuration:")
    for key, value in config.items():
        if not isinstance(value, (list, dict)):
            print(f"  {key}: {value}")
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor(config)
    print(f"\nInitializing trainer...")
    trainer = AlternatingTrainer(config, preprocessor)
    
    # Use our custom data loading method instead of the preprocessor's method
    print(f"\nLoading data for station {args.station_id}...")
    train_data, val_data, test_data = trainer.load_data(project_dir, args.station_id)
    
    # Store original data for visualization
    original_train_data = train_data.copy()
    original_val_data = val_data.copy()
    
    # Check if we need to inject synthetic errors
    if args.error_multiplier is not None and args.error_type != 'none':
        from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
        from utils.error_utils import configure_error_params, inject_errors_into_dataset
        from config import SYNTHETIC_ERROR_PARAMS
        
        print(f"\nInjecting synthetic errors with multiplier {args.error_multiplier:.1f}x...")
        print(f"Error injection mode: {args.error_type}")
        error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, args.error_multiplier)
        
        # Identify target columns for error injection
        water_level_cols = ['vst_raw','vst_raw_feature']
        print(f"Injecting errors into columns: {water_level_cols}")
        
        # Process training data if needed
        if args.error_type in ['both', 'train']:
            print("\nProcessing TRAINING data - injecting errors...")
            error_generator = SyntheticErrorGenerator(error_config)
            train_data_with_errors, train_error_report = inject_errors_into_dataset(
                original_train_data, error_generator, f"{args.station_id}_train", water_level_cols
            )
            train_data = train_data_with_errors
            
            # Handle error reporting based on actual returned format
            if isinstance(train_error_report, dict) and 'total_errors' in train_error_report:
                print(f"Training errors injected: {train_error_report['total_errors']} errors")
                if 'error_counts' in train_error_report:
                    for error_type, count in train_error_report['error_counts'].items():
                        print(f"  - {error_type}: {count}")
            else:
                print(f"Training errors injected successfully")
        
        # Process validation data if needed
        if args.error_type in ['both', 'validation']:
            print("\nProcessing VALIDATION data - injecting errors...")
            error_generator = SyntheticErrorGenerator(error_config)
            val_data_with_errors, val_error_report = inject_errors_into_dataset(
                original_val_data, error_generator, f"{args.station_id}_val", water_level_cols
            )
            val_data = val_data_with_errors
            
            # Handle error reporting based on actual returned format
            if isinstance(val_error_report, dict) and 'total_errors' in val_error_report:
                print(f"Validation errors injected: {val_error_report['total_errors']} errors")
                if 'error_counts' in val_error_report:
                    for error_type, count in val_error_report['error_counts'].items():
                        print(f"  - {error_type}: {count}")
            else:
                print(f"Validation errors injected successfully")
    else:
        print("\nNo synthetic errors injected.")
    
    # Make sure the model's week_steps matches our config
    trainer.model.week_steps = config['week_steps']
    
    print("\nTraining model...")
    history, val_predictions, val_targets = trainer.train(
        train_data, val_data, config['epochs'], config['batch_size']
    )
    
    # Skip test predictions and focus only on validation results
    print("\nSkipping test predictions, focusing on validation results only...")
    
    # Inverse transform the validation predictions to original scale
    print("\nConverting predictions back to original scale...")
    val_predictions_np = val_predictions.numpy()
    
    # Check dimensionality and reshape if necessary
    if val_predictions_np.ndim == 3:
        val_predictions_np = val_predictions_np.reshape(val_predictions_np.shape[0] * val_predictions_np.shape[1], -1)
    elif val_predictions_np.ndim == 2:
        # If predictions are already 2D but need to ensure it's [samples, features]
        val_predictions_np = val_predictions_np.reshape(-1, 1)
    elif val_predictions_np.ndim == 1:
        # If predictions are 1D, reshape to [samples, 1]
        val_predictions_np = val_predictions_np.reshape(-1, 1)
    
    # Apply inverse transform to get back to original scale
    val_predictions_original = trainer.target_scaler.inverse_transform(val_predictions_np)
    
    # Flatten for DataFrame creation
    val_predictions_original = val_predictions_original.flatten()
    
    # Create prediction DataFrame for visualization
    val_pred_df = pd.DataFrame({
        'vst_raw': val_predictions_original,
    }, index=val_data.index[:len(val_predictions_original)])
    
    print(f"Predictions shape: {val_predictions_original.shape}")
    print(f"Min prediction value: {np.min(val_predictions_original)}")
    print(f"Max prediction value: {np.max(val_predictions_original)}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    print(f"All plots will be saved to experiment directory: {exp_dirs['base']}")
    
    # Training convergence plot
    plot_convergence(history, str(args.station_id), 
                     title=f"Training and Validation Loss - Station {args.station_id}",
                     output_dir=exp_dirs['visualizations'])
    
    # Define visualization title including error info if applicable
    viz_title_suffix = "Validation Predictions"
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        viz_title_suffix += f" (Error Mult: {args.error_multiplier}x)"
    
    # Validation predictions plot - if errors were injected, we want to show both original and corrupted data
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        # Prepare synthetic data for create_full_plot
        synthetic_data = {
            'data': val_data.copy(),
            'error_periods': []  # We don't have explicit error period info, but still can show corrupted data
        }
        
        # Create visualization showing both original and corrupted data
        create_full_plot(
            original_val_data,  # Original data as the reference
            val_pred_df,        # Model predictions
            str(args.station_id), 
            config, 
            min(history['val_loss']), 
            title_suffix=viz_title_suffix,
            synthetic_data=synthetic_data,  # Pass synthetic data to show corrupted values
            output_dir=exp_dirs['visualizations']
        )
    else:
        # Standard plot with just the validation data and predictions
        create_full_plot(
            original_val_data, 
            val_pred_df, 
            str(args.station_id), 
            config, 
            min(history['val_loss']), 
            title_suffix=viz_title_suffix,
            output_dir=exp_dirs['visualizations']
        )
    
    # Calculate anomalies for the validation set
    print("\nCalculating anomalies...")
    
    # If we're using synthetic errors, compare against the corrupted data to detect anomalies
    if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
        # Calculate anomalies between corrupted data and predictions
        z_scores, anomalies = calculate_z_scores_mad(
            val_data['vst_raw'].values, 
            val_pred_df['vst_raw'].values,
            window_size=config['window_size'],
            threshold=config['threshold']  
        )
        
        print(f"Number of anomalies detected in corrupted data: {np.sum(anomalies)}")
        
        # Generate anomaly visualization with reference to corrupted data
        plot_title = f"Water Level Forecasting with Anomaly Detection - Station {args.station_id}"
        plot_title += f" (Data with Synthetic Errors {args.error_multiplier}x)"
        
        # Create output directory
        anomaly_viz_dir = exp_dirs['anomaly_detection']
        
        #png_path, html_path = plot_water_level_anomalies(
        #    test_data=val_data,
        #    predictions=val_pred_df['vst_raw'],
        #    z_scores=z_scores,
        #    anomalies=anomalies,
        #    threshold=config['threshold'],
        #    title=plot_title,
        #    output_dir=anomaly_viz_dir,
        #    save_png=True,
        #    save_html=True,
        #    show_plot=False
        #)
        
        #print(f"Anomaly visualization (on corrupted data) saved to:")
        #print(f"PNG: {png_path}")
        #print(f"HTML: {html_path}")
        
        # Also calculate anomalies between original data and corrupted data to verify synthetic errors
        original_vs_corrupted_zscores, original_vs_corrupted_anomalies = calculate_z_scores_mad(
            original_val_data['vst_raw'].values, 
            val_data['vst_raw'].values,
            window_size=config['window_size'],
            threshold=config['threshold']  
        )
        
        print(f"Number of anomalies detected between original and corrupted data: {np.sum(original_vs_corrupted_anomalies)}")
        
        # Plot these anomalies to verify synthetic error injection
        synthetic_error_plot_title = f"Synthetic Errors Detected - Station {args.station_id} (Error Multiplier: {args.error_multiplier}x)"
        
        #syn_png_path, syn_html_path = plot_water_level_anomalies(
        #    test_data=original_val_data,
        #    predictions=val_data['vst_raw'],
        #    z_scores=original_vs_corrupted_zscores,
        #    anomalies=original_vs_corrupted_anomalies,
        #    title=synthetic_error_plot_title,
        #    output_dir=anomaly_viz_dir,
        #    save_png=True,
        #    save_html=True,
        #    show_plot=False,
        #    filename_prefix="synthetic_errors_"
        #)
        
        #print(f"Synthetic errors visualization saved to:")
        #print(f"PNG: {syn_png_path}")
        #print(f"HTML: {syn_html_path}")
    else:
        # Standard anomaly detection against original data
        z_scores, anomalies = calculate_z_scores_mad(
            val_data['vst_raw'].values, 
            val_pred_df['vst_raw'].values,
            window_size=config['window_size'],
            threshold=config['threshold']  
        )
        
        print(f"Number of anomalies detected: {np.sum(anomalies)}")
        
        # Generate anomaly visualization
        plot_title = f"Water Level Forecasting with Anomaly Detection - Station {args.station_id}"
        
        # Create output directory
        anomaly_viz_dir = exp_dirs['anomaly_detection']
        
        #png_path, html_path = plot_water_level_anomalies(
        #    test_data=val_data,
        #    predictions=val_pred_df['vst_raw'],
        #    z_scores=z_scores,
        #    anomalies=anomalies,
        #    title=plot_title,
        #    output_dir=anomaly_viz_dir,
        #    save_png=True,
        #    save_html=True,
        #    show_plot=False
        #)
        
        # Comment out or remove the line causing the error (around line 405-407)
        #print(f"Anomaly visualization saved to:")
        #print(f"PNG: {png_path}")
        #print(f"HTML: {html_path}")
    
    # Calculate metrics on validation data instead of test data
    from utils.pipeline_utils import calculate_performance_metrics
    valid_mask = ~np.isnan(original_val_data['vst_raw'].values)
    pred_mask = ~np.isnan(val_predictions_original)
    combined_mask = valid_mask[:len(pred_mask)] & pred_mask
    
    # Only calculate metrics if we have sufficient valid data points
    if np.sum(combined_mask) > 0:
        metrics = calculate_performance_metrics(
            original_val_data['vst_raw'].values[:len(pred_mask)], 
            val_predictions_original, 
            combined_mask
        )
        
        # Print metrics
        print("\nValidation Metrics (against original data):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        # If we injected errors, also calculate metrics against the corrupted data
        if args.error_multiplier is not None and args.error_type in ['both', 'validation']:
            try:
                # Calculate error correction improvement
                corrupted_metrics = calculate_performance_metrics(
                    val_data['vst_raw'].values[:len(pred_mask)], 
                    val_predictions_original, 
                    combined_mask
                )
                
                print("\nValidation Metrics (against corrupted data):")
                for metric, value in corrupted_metrics.items():
                    print(f"  {metric}: {value:.6f}")
                    
                print("\nError Correction Analysis:")
                
                # Original vs Corrupted (how bad are the errors)
                error_metrics = calculate_performance_metrics(
                    original_val_data['vst_raw'].values[:len(pred_mask)],
                    val_data['vst_raw'].values[:len(pred_mask)],
                    combined_mask
                )
                
                print("  Error Impact (original vs corrupted):")
                for metric, value in error_metrics.items():
                    print(f"    {metric}: {value:.6f}")
                
                # Calculate correction percentage
                if error_metrics['rmse'] > 0:
                    correction_rmse = (1 - metrics['rmse'] / error_metrics['rmse']) * 100
                    print(f"    RMSE improvement: {correction_rmse:.2f}%")
                
                if error_metrics['mae'] > 0:
                    correction_mae = (1 - metrics['mae'] / error_metrics['mae']) * 100
                    print(f"    MAE improvement: {correction_mae:.2f}%")
                
                r2_improvement = metrics['r2'] - error_metrics['r2']
                print(f"    R² improvement: {r2_improvement:.4f}")
                
            except Exception as e:
                print(f"\nError during error correction analysis: {str(e)}")
                print("Continuing with model evaluation...")
    else:
        print("\nNot enough valid data points to calculate metrics")
        metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan')}
    
    # Generate new behavior visualizations
    #behavior_vis_paths = generate_behavior_visualizations(
    #    val_data=val_data,
    #    predictions=val_pred_df['vst_raw'],
    #    station_id=args.station_id,
    #    config=config,
    #    output_path=project_dir / "results" / "Iterative model results",
    #    model=trainer.model,
    #    trainer=trainer
    #)

    # Generate residual plots and other diagnostics
    from _3_lstm_model.model_diagnostics import generate_all_diagnostics
    
    # Create features DataFrame for residual analysis
    features_df = pd.DataFrame({
        'temperature': val_data['temperature'],
        'rainfall': val_data['rainfall']
    })
    
    # Generate all diagnostic plots
    diagnostic_vis_paths = generate_all_diagnostics(
        actual=val_data['vst_raw'],
        predictions=val_pred_df['vst_raw'],
        output_dir=exp_dirs['diagnostics'],
        station_id=args.station_id,
        features_df=features_df
    )
    
    return metrics

if __name__ == "__main__":
    args = parse_arguments()
    metrics = run_alternating_model(args)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {args.experiment} COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Results saved to: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}'}")
    print(f"  - Diagnostics: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}' / 'diagnostics'}")
    print(f"  - Anomaly Detection: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}' / 'anomaly_detection'}")
    print(f"  - Visualizations: {Path(project_dir) / 'results' / 'Iterative model results' / f'experiment_{args.experiment}' / 'visualizations'}")
    print(f"{'='*60}") 