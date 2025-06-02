import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_dir = current_dir.parent.parent
sys.path.append(str(project_dir))

# Import local modules
from experiments.iterative_forecaster.alternating_config import ALTERNATING_CONFIG
from experiments.iterative_forecaster.simple_anomaly_detector import SimpleAnomalyDetector
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from _3_lstm_model.model_plots import create_full_plot, plot_convergence
from _4_anomaly_detection.z_score import calculate_z_scores_mad
from _4_anomaly_detection.anomaly_visualization import plot_water_level_anomalies
from experiments.iterative_forecaster.alternating_trainer import AlternatingTrainer
from experiments.iterative_forecaster.alternating_forecast_model import AlternatingForecastModel

def set_random_seeds(seed):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_experiment_directories(project_dir, experiment_name):
    """Create experiment-specific directory structure."""
    exp_dir = Path(project_dir) / "results" / "Iterative model results" / f"experiment_{experiment_name}"
    
    directories = {
        'base': exp_dir,
        'diagnostics': exp_dir / "diagnostics",
        'visualizations': exp_dir / "visualizations",
        'anomaly_detection': exp_dir / "anomaly_detection",
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🏷️ ANOMALY FLAGGING APPROACH")
    print(f"Experiment directories created under: {exp_dir}")
    return directories

def test_flagging_approach():
    """Test the new anomaly flagging approach."""
    
    # Set random seeds
    set_random_seeds(42)
    
    # Setup experiment
    exp_dirs = setup_experiment_directories(project_dir, "flagging_test")
    
    # Configuration
    config = ALTERNATING_CONFIG.copy()
    config['quick_mode'] = True  # Use quick mode for faster testing
    
    print("\n🏷️ Anomaly Flagging Configuration:")
    print(f"  use_anomaly_flags: {config.get('use_anomaly_flags', False)}")
    print(f"  use_weighted_loss: {config.get('use_weighted_loss', False)}")
    print(f"  anomaly_weight: {config.get('anomaly_weight', 1.0)}")
    print(f"  use_perfect_flags: {config.get('use_perfect_flags', False)}")
    
    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor(config)
    trainer = AlternatingTrainer(config, preprocessor)
    
    # Load data
    print(f"\nLoading data for station 21006846...")
    train_data, val_data, test_data = trainer.load_data(project_dir, "21006846")
    
    # Store original data
    original_train_data = train_data.copy()
    original_val_data = val_data.copy()
    
    # Inject synthetic errors
    print(f"\nInjecting synthetic errors...")
    from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
    from utils.error_utils import configure_error_params, inject_errors_into_dataset
    from config import SYNTHETIC_ERROR_PARAMS
    
    error_config = configure_error_params(SYNTHETIC_ERROR_PARAMS, 1.0)
    water_level_cols = ['vst_raw', 'vst_raw_feature']
    
    # Process training data
    train_error_generator = SyntheticErrorGenerator(error_config)
    train_data_with_errors, train_error_report = inject_errors_into_dataset(
        original_train_data, train_error_generator, "21006846_train", water_level_cols
    )
    
    # Process validation data  
    val_error_generator = SyntheticErrorGenerator(error_config)
    val_data_with_errors, val_error_report = inject_errors_into_dataset(
        original_val_data, val_error_generator, "21006846_val", water_level_cols
    )
    
    print(f"Errors injected in training and validation data")
    
    # Create anomaly detector
    detector = SimpleAnomalyDetector(
        threshold=config.get('anomaly_detection_threshold', 3.0),
        window_size=config.get('anomaly_detection_window', 1500)
    )
    
    # Add anomaly flags to data
    if config.get('use_perfect_flags', False):
        print("\n🎯 Using PERFECT anomaly flags from known error locations")
        
        # Create perfect flags for training data
        train_flags = detector.create_perfect_flags(
            train_error_generator, 
            len(train_data_with_errors), 
            train_data_with_errors.index
        )
        train_data_flagged = detector.add_anomaly_flags_to_dataframe(
            train_data_with_errors, train_flags, config.get('anomaly_flag_column', 'anomaly_flag')
        )
        
        # Create perfect flags for validation data
        val_flags = detector.create_perfect_flags(
            val_error_generator, 
            len(val_data_with_errors),
            val_data_with_errors.index
        )
        val_data_flagged = detector.add_anomaly_flags_to_dataframe(
            val_data_with_errors, val_flags, config.get('anomaly_flag_column', 'anomaly_flag')
        )
        
        # Get statistics
        train_stats = detector.get_anomaly_statistics(train_flags)
        val_stats = detector.get_anomaly_statistics(val_flags)
        
        print(f"\nTraining data anomaly statistics:")
        print(f"  - Total points: {train_stats['total_points']}")
        print(f"  - Anomalous points: {train_stats['anomalous_points']}")
        print(f"  - Anomaly percentage: {train_stats['anomaly_percentage']:.2f}%")
        print(f"  - Number of anomalous periods: {train_stats['number_of_periods']}")
        
        print(f"\nValidation data anomaly statistics:")
        print(f"  - Total points: {val_stats['total_points']}")
        print(f"  - Anomalous points: {val_stats['anomalous_points']}")
        print(f"  - Anomaly percentage: {val_stats['anomaly_percentage']:.2f}%")
        print(f"  - Number of anomalous periods: {val_stats['number_of_periods']}")
        
    else:
        print("\n🤖 Using AUTOMATIC anomaly detection")
        
        # Detect anomalies automatically
        train_flags, _ = detector.detect_anomalies(
            train_data_with_errors['vst_raw'], 
            original_train_data['vst_raw']
        )
        train_data_flagged = detector.add_anomaly_flags_to_dataframe(
            train_data_with_errors, train_flags, config.get('anomaly_flag_column', 'anomaly_flag')
        )
        
        val_flags, _ = detector.detect_anomalies(
            val_data_with_errors['vst_raw'],
            original_val_data['vst_raw']
        )
        val_data_flagged = detector.add_anomaly_flags_to_dataframe(
            val_data_with_errors, val_flags, config.get('anomaly_flag_column', 'anomaly_flag')
        )
    
    print(f"\n🏷️ Data with anomaly flags ready for training!")
    print(f"Training features: {list(train_data_flagged.columns)}")
    print(f"Validation features: {list(val_data_flagged.columns)}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  ✅ Synthetic errors injected")
    print(f"  ✅ Anomaly flags created ({'perfect' if config.get('use_perfect_flags') else 'automatic'})")
    print(f"  ✅ Data ready for weighted loss training")
    print(f"  🎯 Next: Train model with anomaly flags as input features")
    
    # Create comprehensive visualization to verify flagging
    print(f"\n📊 Creating verification plots...")
    
    # Plot 1: Training data verification
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: Full training data view
    plt.subplot(3, 1, 1)
    plt.plot(original_train_data.index, original_train_data['vst_raw'], 
             label='Original Data', color='blue', alpha=0.7, linewidth=1)
    plt.plot(train_data_with_errors.index, train_data_with_errors['vst_raw'], 
             label='Corrupted Data', color='orange', alpha=0.8, linewidth=1)
    
    # Mark flagged periods
    flagged_indices = train_data_flagged.index[train_data_flagged['anomaly_flag'] == 1]
    if len(flagged_indices) > 0:
        plt.scatter(flagged_indices, train_data_flagged.loc[flagged_indices, 'vst_raw'], 
                   color='red', s=3, alpha=0.8, label=f'Flagged as Anomalous ({len(flagged_indices)} points)')
    
    plt.title(f'Training Data: Original vs Corrupted with Anomaly Flags\n{train_stats["anomalous_points"]} points flagged ({train_stats["anomaly_percentage"]:.2f}%)')
    plt.xlabel('Time')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Validation data view
    plt.subplot(3, 1, 2)
    plt.plot(original_val_data.index, original_val_data['vst_raw'], 
             label='Original Data', color='blue', alpha=0.7, linewidth=1)
    plt.plot(val_data_with_errors.index, val_data_with_errors['vst_raw'], 
             label='Corrupted Data', color='orange', alpha=0.8, linewidth=1)
    
    # Mark flagged periods
    val_flagged_indices = val_data_flagged.index[val_data_flagged['anomaly_flag'] == 1]
    if len(val_flagged_indices) > 0:
        plt.scatter(val_flagged_indices, val_data_flagged.loc[val_flagged_indices, 'vst_raw'], 
                   color='red', s=5, alpha=0.8, label=f'Flagged as Anomalous ({len(val_flagged_indices)} points)')
    
    plt.title(f'Validation Data: Original vs Corrupted with Anomaly Flags\n{val_stats["anomalous_points"]} points flagged ({val_stats["anomaly_percentage"]:.2f}%)')
    plt.xlabel('Time')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Flag pattern visualization
    plt.subplot(3, 1, 3)
    plt.plot(train_data_flagged.index, train_data_flagged['anomaly_flag'], 
             label='Training Flags', color='red', alpha=0.7, linewidth=2)
    plt.plot(val_data_flagged.index, val_data_flagged['anomaly_flag'], 
             label='Validation Flags', color='darkred', alpha=0.9, linewidth=2)
    plt.title('Anomaly Flag Pattern (1 = Anomalous, 0 = Normal)')
    plt.xlabel('Time')
    plt.ylabel('Flag Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed error analysis if we have error generators
    if hasattr(train_error_generator, 'error_periods') and train_error_generator.error_periods:
        print(f"\n🔍 DETAILED ERROR ANALYSIS:")
        print(f"\nTraining Data Errors:")
        for i, error_period in enumerate(train_error_generator.error_periods):
            print(f"  {i+1}. {error_period.error_type.upper()}: {error_period.start_time} to {error_period.end_time}")
            
        print(f"\nValidation Data Errors:")
        for i, error_period in enumerate(val_error_generator.error_periods):
            print(f"  {i+1}. {error_period.error_type.upper()}: {error_period.start_time} to {error_period.end_time}")
    
    # Verify flagging accuracy
    print(f"\n✅ FLAGGING VERIFICATION:")
    print(f"Training data:")
    print(f"  - Total periods flagged: {train_stats['number_of_periods']}")
    print(f"  - Total points flagged: {train_stats['anomalous_points']}")
    print(f"  - Flagging percentage: {train_stats['anomaly_percentage']:.2f}%")
    
    print(f"\nValidation data:")
    print(f"  - Total periods flagged: {val_stats['number_of_periods']}")
    print(f"  - Total points flagged: {val_stats['anomalous_points']}")
    print(f"  - Flagging percentage: {val_stats['anomaly_percentage']:.2f}%")

    return train_data_flagged, val_data_flagged, original_val_data

if __name__ == "__main__":
    train_data, val_data, original_data = test_flagging_approach()
    print(f"\n🎉 Flagging approach test completed successfully!")
    print(f"🚀 Ready to train model with anomaly flags!") 