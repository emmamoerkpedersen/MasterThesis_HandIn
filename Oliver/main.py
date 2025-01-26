"""
Main script to run the error detection pipeline.
"""

import pandas as pd
from error_detection import synthetic_errors, error_detection, imputation, validation
from plot_code import error_plots
from error_detection.utils import calculate_error_statistics, export_results
from data_loading import prepare_data_for_error_detection

def create_synthetic_training_data(raw_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Create training dataset by injecting synthetic errors into clean data.
    
    Args:
        raw_data_path: Path to VST_RAW.txt
    
    Returns:
        tuple containing:
        - original_data: Clean, prepared DataFrame
        - modified_data: DataFrame with injected errors
        - validation_results: Dictionary with error injection statistics
    """
    # Load and prepare clean data
    clean_data = prepare_data_for_error_detection(raw_data_path)
    
    # Split into train/test
    train_data, test_data = validation.create_train_test_split(clean_data)
    
    # Generate synthetic errors
    error_generator = synthetic_errors.SyntheticErrorGenerator()
    modified_train_data, ground_truth = error_generator.inject_all_errors(train_data)
    
    # Validate error injection
    validation_results = validation.validate_error_injection(
        train_data,
        modified_train_data,
        ground_truth
    )
    
    return train_data, modified_train_data, validation_results

def run_pipeline(data_path: str, output_path: str):
    """
    Run the complete error detection and imputation pipeline.
    
    Args:
        data_path: Path to input data
        output_path: Path for results
    """
    # Create training data with synthetic errors
    train_data, modified_data, validation_results = create_synthetic_training_data(data_path)
    
    # Detect errors
    detector = error_detection.SpikeDetector()
    error_flags = detector.detect(modified_data)
    
    # Impute errors
    imputer = imputation.SimpleImputer()
    imputed_data = imputer.impute(modified_data, error_flags)
    
    # Calculate performance metrics
    metrics = validation.calculate_metrics(validation_results['ground_truth'], error_flags)
    
    # Generate plots
    error_plots.plot_detected_errors(modified_data, error_flags)
    error_plots.plot_imputation_results(modified_data, imputed_data, error_flags)
    
    # Export results
    export_results(metrics, output_path)

if __name__ == "__main__":
    run_pipeline("path/to/data", "path/to/output") 