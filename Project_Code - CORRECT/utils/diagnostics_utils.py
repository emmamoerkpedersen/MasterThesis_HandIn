"""
Diagnostics utilities for the water level prediction pipeline.
This module contains functions to handle diagnostic generation.
"""
import pandas as pd
from pathlib import Path

def run_preprocessing_diagnostics(project_root, output_path, station_id):
    """
    Run preprocessing diagnostics for the specified station.
    
    Args:
        project_root: Project root directory path
        output_path: Output directory path
        station_id: Station identifier
        
    Returns:
        Boolean indicating if diagnostics were generated successfully
    """
    try:
        from diagnostics.preprocessing_diagnostics import (
            plot_preprocessing_comparison, 
            plot_station_data_overview, 
            plot_vst_vinge_comparison,
            create_detailed_plot,
            get_time_windows
        )
        
        print("Generating preprocessing diagnostics...")
        data_dir = project_root / "results" / "preprocessing_diagnostics"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the original and preprocessed data directly from pickles
        original_data = pd.read_pickle(project_root / "data_utils" / "Sample data" / "original_data.pkl")
        preprocessed_data = pd.read_pickle(project_root / "data_utils" / "Sample data" / "preprocessed_data.pkl")
        
        # Try to load frost periods
        frost_periods = []
        frost_path = project_root / "data_utils" / "Sample data" / "frost_periods.pkl"
        if frost_path.exists():
            try:
                frost_periods = pd.read_pickle(frost_path)
                print(f"Loaded {len(frost_periods)} frost periods from {frost_path}")
            except Exception as e:
                print(f"Error loading frost periods: {e}")
        else:
            print(f"No frost periods file found at {frost_path}")
        
        # For individual station plots, filter the data
        station_original_data = {station_id: original_data[station_id]} if station_id in original_data else {}
        station_preprocessed_data = {station_id: preprocessed_data[station_id]} if station_id in preprocessed_data else {}
        
        # Generate preprocessing plots for individual station
        if station_original_data and station_preprocessed_data:
            print("Generating preprocessing plots...")
            plot_preprocessing_comparison(station_original_data, station_preprocessed_data, Path(output_path), frost_periods)
            plot_station_data_overview(station_original_data, station_preprocessed_data, Path(output_path))
            plot_vst_vinge_comparison(station_preprocessed_data, Path(output_path), station_original_data)
            
            # Generate detailed analysis plot using ALL stations
            print("Generating detailed error analysis plot...")
            
            # Create data dictionary for all stations
            data_dict = {}
            for current_station_id in original_data.keys():
                station_data = original_data[current_station_id]
                data_dict[current_station_id] = {
                    'vst_raw': pd.DataFrame({
                        'Date': station_data['vst_raw'].index,
                        'Value': station_data['vst_raw'].iloc[:, 0]  # First column after Date
                    }),
                    'vinge': None
                }
                
                # Add VINGE data if available
                if 'vinge' in station_data and station_data['vinge'] is not None:
                    vinge_df = station_data['vinge'].copy()
                    if 'W.L [cm]' not in vinge_df.columns:
                        # Try to find a suitable column for VINGE data
                        value_cols = [col for col in vinge_df.columns if col != 'Date']
                        if value_cols:
                            vinge_df['W.L [cm]'] = vinge_df[value_cols[0]]
                    
                    data_dict[current_station_id]['vinge'] = pd.DataFrame({
                        'Date': vinge_df.index,
                        'W.L [cm]': vinge_df['W.L [cm]']
                    })
            
            # Get time windows and create the detailed plot
            time_windows_dict = get_time_windows()
            detailed_plot_path = create_detailed_plot(
                data_dict=data_dict,
                time_windows_dict=time_windows_dict,
                folder=station_id,
                output_dir=output_path
            )
            
            if detailed_plot_path:
                print(f"Detailed analysis plot saved to: {detailed_plot_path}")
            
            return True
        else:
            print(f"Warning: No data found for station {station_id}")
            return False
    except Exception as e:
        print(f"Error generating preprocessing diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_synthetic_diagnostics(split_datasets, stations_results, output_path):
    """
    Run diagnostics for synthetic errors.
    
    Args:
        split_datasets: Dictionary of split datasets
        stations_results: Dictionary of station results with synthetic data
        output_path: Output directory path
        
    Returns:
        Dictionary with diagnostic results or None if failed
    """
    try:
        from diagnostics.synthetic_diagnostics import run_all_synthetic_diagnostics
        
        print("\nGenerating synthetic error diagnostics...")
        synthetic_diagnostic_results = run_all_synthetic_diagnostics(
            split_datasets=split_datasets,
            stations_results=stations_results,
            output_dir=Path(output_path)
        )
        return synthetic_diagnostic_results
    except Exception as e:
        print(f"Error generating synthetic diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def setup_basic_diagnostics(train_data, feature_cols, output_path):
    """
    Set up basic diagnostics such as feature plots.
    
    Args:
        train_data: Training data DataFrame
        feature_cols: List of feature column names
        output_path: Output directory path
    """
    try:
        from _3_lstm_model.model_plots import plot_features_stacked_plots
        
        feature_plot_dir = Path(output_path) / "feature_plots"
        feature_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_features_stacked_plots(train_data, feature_cols, output_dir=feature_plot_dir, years_to_show=3)
        print(f"Generated feature plots in {feature_plot_dir}")
    except Exception as e:
        print(f"Error setting up basic diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()

def run_advanced_diagnostics(test_data, predictions, station_id, output_path, is_comparative=False):
    """
    Run advanced diagnostics on model predictions.
    
    Args:
        test_data: Test data DataFrame
        predictions: Predictions (can be a dict for comparative or single Series/array)
        station_id: Station identifier
        output_path: Output directory path
        is_comparative: Whether this is a comparative diagnostic (multiple models)
        
    Returns:
        Dictionary with visualization paths or None if failed
    """
    try:
        # Create diagnostics output directory
        diagnostics_dir = Path(output_path) / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare feature data for residual analysis
        from utils.pipeline_utils import prepare_features_df
        features_df = prepare_features_df(test_data)
        
        # Prepare rainfall data if available
        rainfall_series = None
        if 'rainfall' in test_data.columns:
            rainfall_series = pd.Series(test_data['rainfall'], index=test_data.index)
        
        # Run the appropriate diagnostics based on whether this is comparative
        if is_comparative:
            from _3_lstm_model.model_diagnostics import generate_comparative_diagnostics
            
            # For comparative diagnostics, predictions should be a dictionary
            # with model names as keys and prediction Series as values
            all_visualization_paths = generate_comparative_diagnostics(
                actual=pd.Series(test_data['vst_raw'], index=test_data.index),
                predictions_dict=predictions,
                output_dir=diagnostics_dir,
                station_id=station_id,
                rainfall=rainfall_series,
                features_df=features_df
            )
        else:
            from _3_lstm_model.model_diagnostics import (
                analyze_individual_residuals, 
                create_actual_vs_predicted_plot,
                create_feature_importance_plot,
                create_correlation_analysis
            )
            
            # For single model diagnostics, predictions should be a Series
            actual = pd.Series(test_data['vst_raw'], index=test_data.index)
            
            # Create individual residual plots
            print("\nGenerating individual residual plots...")
            individual_plots = analyze_individual_residuals(
                actual=actual,
                predictions=predictions,
                output_dir=diagnostics_dir,
                station_id=station_id,
                features_df=features_df
            )
            
            # Create actual vs predicted plot
            print("\nGenerating actual vs predicted plot...")
            actual_pred_path = create_actual_vs_predicted_plot(
                actual=actual,
                predictions=predictions,
                output_dir=diagnostics_dir,
                station_id=station_id
            )
            
            # Get all feature columns from the test data
            feature_cols = [col for col in test_data.columns if col != 'vst_raw']
            
            # Create feature importance plot
            print("\nGenerating feature importance plot...")
            feature_importance_path = create_feature_importance_plot(
                test_data=test_data,
                predictions=predictions,
                feature_cols=feature_cols,
                output_dir=diagnostics_dir,
                station_id=station_id
            )
            
            # Create correlation analysis plots
            print("\nGenerating correlation analysis plots...")
            correlation_paths = create_correlation_analysis(
                test_data=test_data,
                predictions=predictions,
                feature_cols=feature_cols,
                output_dir=diagnostics_dir,
                station_id=station_id
            )
            
            # Combine all visualization paths
            all_visualization_paths = {
                'residual_plots': individual_plots,
                'actual_vs_predicted': actual_pred_path,
                'feature_importance': feature_importance_path,
                'correlation_analysis': correlation_paths
            }
        
        return all_visualization_paths
    except Exception as e:
        print(f"Error generating {'comparative ' if is_comparative else ''}diagnostic visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 