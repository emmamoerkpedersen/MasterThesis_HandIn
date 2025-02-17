"""
Main script to run the error detection pipeline.
"""

import pandas as pd
from pathlib import Path
#from error_detection import synthetic_errors, imputation, validation
#from error_detection.anomaly_detection import AnomalyDetector
#from plot_code import error_plots
#from error_detection.utils import export_results
#from data_loading import prepare_data_for_error_detection
#from error_detection.config import SYNTHETIC_ERROR_PARAMS, PHYSICAL_LIMITS
from _1_preprocessing.Processing_data import preprocess_data
from _1_preprocessing.split import split_data
from data_utils.data_loading import load_all_station_data
from diagnostics.preprocessing_diagnostics import plot_preprocessing_comparison, generate_preprocessing_report, plot_additional_data
from diagnostics.split_diagnostics import plot_split_visualization, generate_split_report
from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import SYNTHETIC_ERROR_PARAMS
from diagnostics.synthetic_diagnostics import plot_synthetic_errors, generate_synthetic_report, create_interactive_plot

def run_pipeline(
    data_path: str, 
    output_path: str, 
    preprocess_diagnostics: bool = False,
    split_diagnostics: bool = False,
    synthetic_diagnostics: bool = True,  # Default to True since we want to verify error injection
    detection_diagnostics: bool = False,  # For future use
    imputation_diagnostics: bool = False  # For future use
):
    """
    Run the complete error detection and imputation pipeline.
    
    Args:
        data_path: Path to input data
        output_path: Path for results
        preprocess_diagnostics: Generate diagnostics for preprocessing step
        split_diagnostics: Generate diagnostics for data splitting
        synthetic_diagnostics: Generate diagnostics for synthetic error generation
        detection_diagnostics: Generate diagnostics for error detection
        imputation_diagnostics: Generate diagnostics for imputation
    """
    
    #########################################################
    #    Step 1: Load and preprocess all station data       #
    #########################################################
    
    print("Loading and preprocessing station data...")
    
    # Load original data first (if preprocessing diagnostics enabled)
    if preprocess_diagnostics:
        original_data = load_all_station_data()
    
    # Preprocess data
    preprocessed_data = preprocess_data()
    
    # Generate preprocessing diagnostics if enabled
    if preprocess_diagnostics:
        print("\nGenerating preprocessing diagnostics...")
        plot_preprocessing_comparison(
            original_data, 
            preprocessed_data, 
            Path(output_path)
        )
        plot_additional_data(
            preprocessed_data,
            Path(output_path)
        )
        generate_preprocessing_report(
            original_data, 
            preprocessed_data, 
            Path(output_path)
        )
    
        # Print summary of processed stations
        print("\nPreprocessing complete. Summary:")
        for station_name, station_data in preprocessed_data.items():
            if station_data['vst_raw'] is not None:
                print(f"\nStation: {station_name}")
                for data_type, data in station_data.items():
                    if data is not None:
                        print(f"  - {data_type}: {len(data)} points")
                        if isinstance(data, pd.DataFrame):
                            print(f"    Time range: {data.index.min()} to {data.index.max()}")
                            
    #########################################################
    #  Step 2: Split data into train/validation/test sets   #
    #########################################################
    
    print("\nSplitting data into train/validation/test sets...")
    split_datasets = split_data(preprocessed_data)
    
    # Generate split diagnostics if enabled
    if split_diagnostics:
        print("Generating split diagnostics...")
        plot_split_visualization(
            split_datasets,
            Path(output_path)
        )
        generate_split_report(
            split_datasets,
            Path(output_path)
        )

    #########################################################
    # Step 3: Generate synthetic errors for testing data    #
    #########################################################
    
    print("\nInjecting synthetic errors into test data...")
    error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)
    stations_results = {}
    error_types = ['spike', 'drift', 'offset', 'baseline_shift']
    for station_name, station_data in split_datasets['test'].items():
        if 'vst_raw' in station_data and station_data['vst_raw'] is not None:
            print(f"\nProcessing station: {station_name}")
            
            # Get test data and inject errors
            test_data = station_data['vst_raw'].copy()
            modified_data, ground_truth = error_generator.inject_all_errors(test_data, error_types=error_types)
            
            # Store modified data and results
            split_datasets['test'][station_name]['vst_raw_modified'] = modified_data
            stations_results[station_name] = {
                'error_periods': error_generator.error_periods.copy(),  # Store a copy to avoid reference issues
                'ground_truth': ground_truth
            }
            
            # Clear error periods for next station
            error_generator.error_periods = []
            error_generator.used_indices = set()
    
    # Generate synthetic error diagnostics if enabled
    if synthetic_diagnostics:
        print("\nGenerating synthetic error diagnostics...")
        for station_name, station_data in split_datasets['test'].items():
            if 'vst_raw_modified' in station_data:
                # Create static plot
                plot_synthetic_errors(
                    original_data=station_data['vst_raw'],
                    modified_data=station_data['vst_raw_modified'],
                    error_periods=stations_results[station_name]['error_periods'],
                    station_name=station_name,
                    output_dir=Path(output_path)
                )
                # Create interactive plot
                create_interactive_plot(
                    original_data=station_data['vst_raw'],
                    modified_data=station_data['vst_raw_modified'],
                    error_periods=stations_results[station_name]['error_periods'],
                    station_name=station_name,
                    output_dir=Path(output_path)
                )
        generate_synthetic_report(stations_results, Path(output_path))

    # #########################################################
    # # Step 4: Initialize and run anomaly detector          #
    # #########################################################
    
    # # Step 4.1: Initialize and run simple anomaly detector
    # detector = AnomalyDetector()
    # error_flags = detector.detect(modified_data)
    
    # #########################################################
    # # Step 5: Impute errors                                #
    # #########################################################
    
    # imputer = imputation.SimpleImputer()
    # imputed_data = imputer.impute(modified_data, error_flags)
    
    # #########################################################
    # # Step 6: Calculate performance metrics                #
    # #########################################################
    
    # metrics = validation.calculate_metrics(ground_truth, error_flags)
    
    # #########################################################
    # # Step 7: Generate plots                              #
    # #########################################################
    
    # error_plots.plot_detected_errors(modified_data, error_flags)
    # error_plots.plot_imputation_results(modified_data, imputed_data, error_flags)
    
    # #########################################################
    # # Step 8: Export results                               #
    # #########################################################
    
    # export_results({
    #     'metrics': metrics,
    #     'config': SYNTHETIC_ERROR_PARAMS,
    #     'physical_limits': PHYSICAL_LIMITS
    # }, output_path)

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "VST_RAW.txt"
    output_path = project_root / "results"
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline with preprocessing, split and synthetic diagnostics enabled
    run_pipeline(
        str(data_path), 
        str(output_path), 
        preprocess_diagnostics=True,
        split_diagnostics=False,
        synthetic_diagnostics=False
    )
