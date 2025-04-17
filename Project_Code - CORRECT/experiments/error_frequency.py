"""
This module contains functions for running error frequency experiments.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import torch
from datetime import datetime

# Import plot functions from model_plots module
from _3_lstm_model.model_plots import create_water_level_plot_png

def run_error_frequency_experiments(run_pipeline, error_frequencies=None):
    """
    Run experiments with different error frequencies to analyze impact on model performance.
    
    Args:
        run_pipeline: Function to run the model pipeline
        error_frequencies: List of error frequencies to test (between 0 and 1)
                          If None, uses default values [0.0, 0.05, 0.1, 0.15]
    """
    if error_frequencies is None:
        error_frequencies = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]
    
    # Set up paths
    project_root = Path(__file__).parents[1]
    data_path = project_root / "data_utils" / "Sample data" / "VST_RAW.txt"
    output_path = project_root / "results"
    compare_plots_dir = output_path / "error_frequency_plots"
    
    # Create output directories if they don't exist
    output_path.mkdir(parents=True, exist_ok=True)
    compare_plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning LSTM model with varying error frequencies to analyze impact")
    
    # Create a dataframe to store all results
    all_results = pd.DataFrame()
    
    # First run the clean model once (error_frequency = 0.0)
    print(f"\n{'='*80}")
    print(f"Running baseline model with clean data (0% error frequency)")
    print(f"{'='*80}")
    
    try:
        # Run pipeline with clean data
        clean_metrics = run_pipeline(
            project_root=project_root,
            data_path=data_path, 
            output_path=output_path,
            preprocess_diagnostics=False,
            synthetic_diagnostics=False,
            inject_synthetic_errors=False,  # No synthetic errors for baseline
            model_diagnostics=True,         # Enable basic model plots
            advanced_diagnostics=False,     # But disable advanced diagnostics
            error_frequency=0.0,
        )
        
        # Save the clean model for later use
        clean_model_path = output_path / "clean_model.pth"
        if Path("final_model.pth").exists():
            # Copy the final model to our clean model path
            import shutil
            shutil.copy("final_model.pth", clean_model_path)
            print(f"\nSaved clean model to: {clean_model_path}")
        
        print(f"\nBaseline model with clean data completed!")
        
        # Store baseline metrics
        baseline_metrics = {
            'clean_rmse': clean_metrics.get('model', {}).get('rmse', np.nan),
            'clean_mae': clean_metrics.get('model', {}).get('mae', np.nan),
            'clean_r2': clean_metrics.get('model', {}).get('r2', np.nan),
            'clean_nse': clean_metrics.get('model', {}).get('nse', np.nan),
            'clean_val_loss': clean_metrics.get('model', {}).get('val_loss', np.nan)
        }
        
    except Exception as e:
        print(f"\nError running baseline model: {e}")
        traceback.print_exc()
        baseline_metrics = {
            'clean_rmse': np.nan,
            'clean_mae': np.nan,
            'clean_r2': np.nan,
            'clean_nse': np.nan,
            'clean_val_loss': np.nan
        }
    
    # Now run for each non-zero error frequency
    for error_freq in [f for f in error_frequencies if f > 0]:
        print(f"\n{'='*80}")
        print(f"Running experiment with error frequency: {error_freq*100:.1f}%")
        print(f"{'='*80}")
        
        try:
            # Run pipeline with current error frequency
            performance_metrics = run_pipeline(
                project_root=project_root,
                data_path=data_path, 
                output_path=output_path,
                preprocess_diagnostics=False,
                synthetic_diagnostics=False,
                inject_synthetic_errors=True,  # Enable synthetic error injection
                model_diagnostics=True,        # Enable basic model plots 
                advanced_diagnostics=False,    # But disable advanced diagnostics
                error_frequency=error_freq,
            )
            
            # Create a record for this experiment
            experiment_record = {
                'error_frequency': error_freq,
                'error_rmse': performance_metrics.get('error_model', {}).get('rmse', np.nan),
                'error_mae': performance_metrics.get('error_model', {}).get('mae', np.nan),
                'error_r2': performance_metrics.get('error_model', {}).get('r2', np.nan),
                'error_nse': performance_metrics.get('error_model', {}).get('nse', np.nan),
                'error_val_loss': performance_metrics.get('error_model', {}).get('val_loss', np.nan),
                **baseline_metrics  # Add the baseline metrics
            }
            
            # Calculate percent changes
            if not np.isnan(baseline_metrics['clean_rmse']) and baseline_metrics['clean_rmse'] != 0:
                experiment_record['rmse_pct_increase'] = ((experiment_record['error_rmse'] - baseline_metrics['clean_rmse']) / baseline_metrics['clean_rmse']) * 100
            else:
                experiment_record['rmse_pct_increase'] = np.nan
                
            if not np.isnan(baseline_metrics['clean_mae']) and baseline_metrics['clean_mae'] != 0:
                experiment_record['mae_pct_increase'] = ((experiment_record['error_mae'] - baseline_metrics['clean_mae']) / baseline_metrics['clean_mae']) * 100
            else:
                experiment_record['mae_pct_increase'] = np.nan
                
            if not np.isnan(baseline_metrics['clean_nse']) and baseline_metrics['clean_nse'] != 0:
                experiment_record['nse_pct_decrease'] = ((baseline_metrics['clean_nse'] - experiment_record['error_nse']) / baseline_metrics['clean_nse']) * 100
            else:
                experiment_record['nse_pct_decrease'] = np.nan
            
            # Add to results dataframe
            all_results = pd.concat([all_results, pd.DataFrame([experiment_record])], ignore_index=True)
            
            # Save the cumulative results after each run
            cumulative_results_path = output_path / "error_frequency_results.csv"
            all_results.to_csv(cumulative_results_path, index=False)
            print(f"\nSaved cumulative results to: {cumulative_results_path}")
            
            # Also add to the standard error comparison file
            standard_metrics_file = output_path / "error_comparison_metrics.csv"
            if standard_metrics_file.exists():
                # Read existing file, check if we already have this frequency
                existing_df = pd.read_csv(standard_metrics_file)
                # Remove any existing row with this error frequency to avoid duplicates
                existing_df = existing_df[existing_df['error_frequency'] != error_freq]
                # Add our new result
                updated_df = pd.concat([existing_df, pd.DataFrame([experiment_record])], ignore_index=True)
                updated_df.to_csv(standard_metrics_file, index=False)
            else:
                # Create new file with just this result
                pd.DataFrame([experiment_record]).to_csv(standard_metrics_file, index=False)
            
            print(f"\nExperiment with {error_freq*100:.1f}% error frequency completed!")
            
        except Exception as e:
            print(f"\nError running pipeline with {error_freq*100:.1f}% error frequency: {e}")
            traceback.print_exc()
    
    print("\nAll experiments completed!")
    
    # Load the final results CSV
    try:
        final_results = pd.read_csv(output_path / "error_comparison_metrics.csv")
        
        print("\nSummary of Results:")
        print(f"{'='*80}")
        print(final_results[['error_frequency', 'clean_rmse', 'error_rmse', 'clean_mae', 'error_mae', 'clean_nse', 'error_nse']])
        
        # Calculate percentage degradation if not already done
        if 'rmse_pct_increase' not in final_results.columns:
            final_results['rmse_pct_increase'] = ((final_results['error_rmse'] - final_results['clean_rmse']) / final_results['clean_rmse']) * 100
        if 'mae_pct_increase' not in final_results.columns:
            final_results['mae_pct_increase'] = ((final_results['error_mae'] - final_results['clean_mae']) / final_results['clean_mae']) * 100
        if 'nse_pct_decrease' not in final_results.columns:
            final_results['nse_pct_decrease'] = ((final_results['clean_nse'] - final_results['error_nse']) / final_results['clean_nse']) * 100
        
        print("\nPerformance Degradation:")
        print(f"{'='*80}")
        print(final_results[['error_frequency', 'rmse_pct_increase', 'mae_pct_increase', 'nse_pct_decrease']])
        
        # Create plots to visualize the relationship between error frequency and model performance
        plt.figure(figsize=(15, 10))
        
        # Sort results by error frequency for smooth plots
        final_results = final_results.sort_values('error_frequency')
        
        # Plot 1: Absolute metrics by frequency
        plt.subplot(2, 2, 1)
        plt.plot(final_results['error_frequency'], final_results['clean_rmse'], 'o-', label='Clean Model RMSE')
        plt.plot(final_results['error_frequency'], final_results['error_rmse'], 'o-', label='Error Model RMSE')
        plt.title('RMSE vs Error Frequency')
        plt.xlabel('Error Frequency')
        plt.ylabel('RMSE (mm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(final_results['error_frequency'], final_results['clean_mae'], 'o-', label='Clean Model MAE')
        plt.plot(final_results['error_frequency'], final_results['error_mae'], 'o-', label='Error Model MAE')
        plt.title('MAE vs Error Frequency')
        plt.xlabel('Error Frequency')
        plt.ylabel('MAE (mm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Percentage changes by frequency
        plt.subplot(2, 2, 3)
        plt.plot(final_results['error_frequency'], final_results['rmse_pct_increase'], 'o-', label='RMSE % Increase')
        plt.plot(final_results['error_frequency'], final_results['mae_pct_increase'], 'o-', label='MAE % Increase')
        plt.title('Error Metrics % Increase vs Error Frequency')
        plt.xlabel('Error Frequency')
        plt.ylabel('Percentage Increase (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: NSE by frequency
        plt.subplot(2, 2, 4)
        plt.plot(final_results['error_frequency'], final_results['clean_nse'], 'o-', label='Clean Model NSE')
        plt.plot(final_results['error_frequency'], final_results['error_nse'], 'o-', label='Error Model NSE')
        plt.title('NSE vs Error Frequency')
        plt.xlabel('Error Frequency')
        plt.ylabel('NSE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plot_path = output_path / "error_frequency_impact.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved error frequency impact visualization to: {plot_path}")
        
        # Create a summary HTML file
        try:
            create_summary_html(final_results, output_path, compare_plots_dir)
            print(f"\nCreated summary HTML file at: {output_path / 'error_frequency_summary.html'}")
        except Exception as e:
            print(f"\nError creating summary HTML: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"\nError processing results: {e}")
        traceback.print_exc()

def create_summary_html(results_df, output_path, plots_dir):
    """
    Create an HTML summary page of all error frequency experiment results.
    
    Args:
        results_df: DataFrame containing results from all experiments
        output_path: Path to the output directory
        plots_dir: Directory containing prediction plots
    """
    # HTML template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error Frequency Experiment Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333366; }}
            h2 {{ color: #333366; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            th {{ background-color: #f2f2f2; text-align: center; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .plot-container {{ margin-top: 30px; text-align: center; }}
            .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 30px; }}
            .comparison-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            .comparison-item h3 {{ margin-top: 0; color: #333366; }}
            .comparison-item img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Error Frequency Experiment Results</h1>
        <p>Generated on: {timestamp}</p>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Error Frequency</th>
                <th>Clean RMSE</th>
                <th>Error RMSE</th>
                <th>RMSE % Increase</th>
                <th>Clean MAE</th>
                <th>Error MAE</th>
                <th>MAE % Increase</th>
                <th>Clean NSE</th>
                <th>Error NSE</th>
                <th>NSE % Decrease</th>
            </tr>
            {table_rows}
        </table>
        
        <h2>Performance Visualization</h2>
        <div class="plot-container">
            <img src="{impact_plot_path}" alt="Error Frequency Impact" />
        </div>
        
        <h2>Prediction Comparison by Error Frequency</h2>
        <div class="comparison-grid">
            {prediction_plots}
        </div>
    </body>
    </html>
    """
    
    # Generate table rows
    table_rows = ""
    for _, row in results_df.sort_values('error_frequency').iterrows():
        table_rows += f"""
        <tr>
            <td>{row['error_frequency']:.3f}</td>
            <td>{row['clean_rmse']:.4f}</td>
            <td>{row['error_rmse']:.4f}</td>
            <td>{row.get('rmse_pct_increase', np.nan):.2f}%</td>
            <td>{row['clean_mae']:.4f}</td>
            <td>{row['error_mae']:.4f}</td>
            <td>{row.get('mae_pct_increase', np.nan):.2f}%</td>
            <td>{row['clean_nse']:.4f}</td>
            <td>{row['error_nse']:.4f}</td>
            <td>{row.get('nse_pct_decrease', np.nan):.2f}%</td>
        </tr>
        """
    
    # Find prediction plot files
    prediction_plots = ""
    for plot_file in plots_dir.glob("*.png"):
        if "water_level" in plot_file.name:
            # Extract error frequency from filename if possible, otherwise use the filename
            freq_str = "Unknown"
            try:
                # Try to extract error frequency from filename
                freq_parts = plot_file.stem.split("_")
                for i, part in enumerate(freq_parts):
                    if part == "freq" and i < len(freq_parts) - 1:
                        freq_str = f"Error Frequency: {float(freq_parts[i+1]):.2f}"
                        break
            except:
                freq_str = plot_file.stem
                
            prediction_plots += f"""
            <div class="comparison-item">
                <h3>{freq_str}</h3>
                <img src="{plots_dir.name}/{plot_file.name}" alt="Predictions with {freq_str}" />
            </div>
            """
    
    # Find the main impact plot
    impact_plot_path = "error_frequency_impact.png"
    
    # Format the HTML
    html_content = html_content.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        table_rows=table_rows,
        impact_plot_path=impact_plot_path,
        prediction_plots=prediction_plots
    )
    
    # Write the HTML file
    with open(output_path / "error_frequency_summary.html", "w") as f:
        f.write(html_content) 