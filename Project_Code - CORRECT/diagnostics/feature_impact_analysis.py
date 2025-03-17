import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import torch

# Import from other modules
from _3_lstm_model.lstm_model import LSTMForecaster
from _3_lstm_model.hyperparameter_tuning import train_and_evaluate, load_best_hyperparameters
from _3_lstm_model.data_preparation import prepare_data_for_lstm
from diagnostics.hyperparameter_diagnostics import plot_best_model_performance

def run_feature_impact_analysis(
    split_datasets,
    stations_results,
    output_path,
    base_config=None,
    n_repeats=5,
    diagnostics=True
):
    """
    Run a dedicated analysis to evaluate the impact of different feature combinations
    on model performance.
    
    Args:
        split_datasets: Dictionary containing train, val, test datasets
        stations_results: Dictionary containing station-specific results
        output_path: Path to save results
        base_config: Base configuration for the model
        n_repeats: Number of times to repeat each experiment for statistical significance
        diagnostics: Whether to generate diagnostic plots
    
    Returns:
        Dictionary containing results of the analysis
    """
    print("\n" + "="*80)
    print("RUNNING FEATURE IMPACT ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path(output_path) / "feature_impact_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define feature combinations to test
    feature_combinations = [
        ['vst_raw'],
        ['vst_raw', 'temperature'],
        ['vst_raw', 'rainfall'],
        ['vst_raw', 'temperature', 'rainfall']
    ]
    
    # Load best hyperparameters if available, otherwise use defaults
    if base_config is None:
        base_config = {
            'sequence_length': 48,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'patience': 3
        }
    else:
        # Make a copy to avoid modifying the original
        base_config = base_config.copy()
    
    # Try to load best hyperparameters from previous runs
    hyperparams_dir = Path(output_path) / "hyperparameter_tuning"
    if hyperparams_dir.exists():
        try:
            best_config = load_best_hyperparameters(hyperparams_dir, base_config)
            if best_config:
                base_config.update(best_config)
                print(f"Loaded best hyperparameters from previous run")
        except Exception as e:
            print(f"Could not load best hyperparameters: {e}")
    
    # Prepare results storage
    all_results = []
    best_models = {}
    
    # Run experiments for each feature combination
    for feature_cols in feature_combinations:
        feature_name = ', '.join(feature_cols)
        print(f"\nTesting feature combination: {feature_name}")
        
        # Update config with current feature columns
        config = base_config.copy()
        config['feature_cols'] = feature_cols
        
        # Run multiple times for statistical significance
        combination_results = []
        
        for i in range(n_repeats):
            print(f"  Run {i+1}/{n_repeats}")
            
            # Train and evaluate model
            result = train_and_evaluate(
                split_datasets['train'], 
                split_datasets['val'], 
                split_datasets['test'],
                config
            )
            
            # Store results
            result_entry = {
                'feature_cols': feature_cols,
                'feature_name': feature_name,
                'run_id': i,
                'train_loss': result['train_loss'],
                'val_loss': result['val_loss'],
                'test_loss': result['test_loss'],
                'train_rmse': result['train_rmse'],
                'val_rmse': result['val_rmse'],
                'test_rmse': result['test_rmse'],
                'train_mae': result['train_mae'],
                'val_mae': result['val_mae'],
                'test_mae': result['test_mae'],
                'history': result['history'],
                'forecast': result['forecast'].tolist() if isinstance(result['forecast'], np.ndarray) else result['forecast'],
                'truth': result['truth'].tolist() if isinstance(result['truth'], np.ndarray) else result['truth']
            }
            
            combination_results.append(result_entry)
            all_results.append(result_entry)
        
        # Find best model for this feature combination
        best_run = min(combination_results, key=lambda x: x['val_loss'])
        best_models[feature_name] = best_run
        
        # Generate plots for the best model in this feature combination
        if diagnostics:
            feature_dir = output_dir / feature_name.replace(', ', '_')
            feature_dir.mkdir(exist_ok=True)
            
            # Plot performance
            plot_best_model_performance(
                best_run['forecast'],
                best_run['truth'],
                feature_dir
            )
    
    # Save all results
    with open(output_dir / 'feature_impact_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparative analysis
    if diagnostics:
        generate_feature_impact_report(all_results, best_models, output_dir)
    
    return {
        'all_results': all_results,
        'best_models': best_models
    }

def generate_feature_impact_report(results, best_models, output_dir):
    """
    Generate comprehensive visualizations comparing different feature combinations.
    
    Args:
        results: List of all experiment results
        best_models: Dictionary of best models for each feature combination
        output_dir: Directory to save visualizations
    """
    print("\nGenerating feature impact analysis visualizations...")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # 1. Box plots of performance metrics by feature combination
    for metric in ['test_rmse', 'test_mae', 'test_loss']:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='feature_name', y=metric, data=df)
        plt.title(f'Distribution of {metric.upper()} by Feature Combination', fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel(metric.upper(), fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_impact_{metric}.png', dpi=300)
        plt.close()
    
    # 2. Bar chart with error bars
    metrics_summary = df.groupby('feature_name').agg(
        mean_rmse=('test_rmse', 'mean'),
        std_rmse=('test_rmse', 'std'),
        mean_mae=('test_mae', 'mean'),
        std_mae=('test_mae', 'std'),
        count=('test_rmse', 'count')
    ).reset_index()
    
    plt.figure(figsize=(14, 7))
    
    # Plot RMSE
    x = np.arange(len(metrics_summary))
    width = 0.35
    
    plt.bar(
        x - width/2, 
        metrics_summary['mean_rmse'], 
        width, 
        yerr=metrics_summary['std_rmse'],
        label='RMSE',
        capsize=7,
        color='steelblue',
        alpha=0.7
    )
    
    # Plot MAE
    plt.bar(
        x + width/2, 
        metrics_summary['mean_mae'], 
        width, 
        yerr=metrics_summary['std_mae'],
        label='MAE',
        capsize=7,
        color='darkorange',
        alpha=0.7
    )
    
    # Add count annotations
    for i, row in metrics_summary.iterrows():
        plt.annotate(
            f"n={row['count']}",
            xy=(i, max(row['mean_rmse'] + row['std_rmse'], row['mean_mae'] + row['std_mae']) + 0.01),
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.xlabel('Feature Combination', fontsize=14)
    plt.ylabel('Error Metric Value', fontsize=14)
    plt.title('Impact of Feature Selection on Model Performance', fontsize=16)
    plt.xticks(x, metrics_summary['feature_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_impact_comparison.png', dpi=300)
    plt.close()
    
    # 3. Learning curves comparison
    plt.figure(figsize=(14, 8))
    
    # Plot validation loss curves for best models of each feature combination
    for feature_name, model in best_models.items():
        if 'history' in model and 'val_loss' in model['history']:
            epochs = range(1, len(model['history']['val_loss']) + 1)
            plt.plot(
                epochs, 
                model['history']['val_loss'], 
                'o-', 
                linewidth=2, 
                label=f"{feature_name} (Final: {model['val_loss']:.6f})"
            )
    
    plt.title('Validation Loss Curves by Feature Combination', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_impact_learning_curves.png', dpi=300)
    plt.close()
    
    # 4. Forecast comparison for best models
    plt.figure(figsize=(16, 10))
    
    # Get the maximum sequence length for plotting
    max_len = 0
    for model in best_models.values():
        if isinstance(model['truth'], list):
            max_len = max(max_len, len(model['truth']))
        else:
            max_len = max(max_len, model['truth'].shape[0])
    
    # Only plot a reasonable number of points
    plot_len = min(max_len, 200)
    
    # Plot actual values
    first_model = list(best_models.values())[0]
    truth = first_model['truth']
    if isinstance(truth, list):
        truth = np.array(truth)
    
    plt.plot(
        range(plot_len), 
        truth[:plot_len], 
        'k-', 
        linewidth=2, 
        label='Actual'
    )
    
    # Plot forecasts for each feature combination
    for feature_name, model in best_models.items():
        forecast = model['forecast']
        if isinstance(forecast, list):
            forecast = np.array(forecast)
        
        plt.plot(
            range(plot_len), 
            forecast[:plot_len], 
            '--', 
            linewidth=1.5, 
            alpha=0.8,
            label=f"{feature_name} Forecast"
        )
    
    plt.title('Forecast Comparison by Feature Combination', fontsize=16)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_impact_forecast_comparison.png', dpi=300)
    plt.close()
    
    # 5. Percentage improvement analysis
    # Calculate percentage improvement relative to baseline (vst_raw only)
    baseline_feature = 'vst_raw'
    baseline_metrics = df[df['feature_name'] == baseline_feature].agg({
        'test_rmse': 'mean',
        'test_mae': 'mean'
    })
    
    improvement_data = []
    
    for feature_name, group in df.groupby('feature_name'):
        if feature_name == baseline_feature:
            continue
            
        mean_metrics = group.agg({
            'test_rmse': 'mean',
            'test_mae': 'mean'
        })
        
        rmse_improvement = (baseline_metrics['test_rmse'] - mean_metrics['test_rmse']) / baseline_metrics['test_rmse'] * 100
        mae_improvement = (baseline_metrics['test_mae'] - mean_metrics['test_mae']) / baseline_metrics['test_mae'] * 100
        
        improvement_data.append({
            'feature_name': feature_name,
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement
        })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(improvement_df))
        width = 0.35
        
        plt.bar(
            x - width/2, 
            improvement_df['rmse_improvement'], 
            width, 
            label='RMSE Improvement',
            color='green',
            alpha=0.7
        )
        
        plt.bar(
            x + width/2, 
            improvement_df['mae_improvement'], 
            width, 
            label='MAE Improvement',
            color='purple',
            alpha=0.7
        )
        
        # Add value annotations
        for i, row in improvement_df.iterrows():
            plt.annotate(
                f"{row['rmse_improvement']:.1f}%",
                xy=(i - width/2, row['rmse_improvement'] + 0.5),
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.annotate(
                f"{row['mae_improvement']:.1f}%",
                xy=(i + width/2, row['mae_improvement'] + 0.5),
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Feature Combination', fontsize=14)
        plt.ylabel('Percentage Improvement (%)', fontsize=14)
        plt.title(f'Percentage Improvement Relative to Baseline ({baseline_feature} only)', fontsize=16)
        plt.xticks(x, improvement_df['feature_name'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_impact_percentage_improvement.png', dpi=300)
        plt.close()
    
    # Generate HTML report
    generate_html_report(results, best_models, metrics_summary, output_dir)

def generate_html_report(results, best_models, metrics_summary, output_dir):
    """Generate an HTML report summarizing the feature impact analysis."""
    report_path = output_dir / "feature_impact_report.html"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .section { margin: 20px 0; }
                .plot { margin: 20px 0; text-align: center; }
                .plot img { max-width: 100%; border: 1px solid #ddd; }
                .highlight { background-color: #e6ffe6; }
            </style>
            <title>Feature Impact Analysis Report</title>
        </head>
        <body>
            <h1>Feature Impact Analysis Report</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <div class='section'>
                <h2>Summary of Results</h2>
                <table>
                    <tr>
                        <th>Feature Combination</th>
                        <th>Mean RMSE</th>
                        <th>Std RMSE</th>
                        <th>Mean MAE</th>
                        <th>Std MAE</th>
                        <th>Number of Runs</th>
                    </tr>
        """)
        
        # Find best performing feature set
        best_feature = metrics_summary.loc[metrics_summary['mean_rmse'].idxmin(), 'feature_name']
        
        for _, row in metrics_summary.iterrows():
            highlight = ' class="highlight"' if row['feature_name'] == best_feature else ''
            f.write(f"""
                    <tr{highlight}>
                        <td>{row['feature_name']}</td>
                        <td>{row['mean_rmse']:.6f}</td>
                        <td>{row['std_rmse']:.6f}</td>
                        <td>{row['mean_mae']:.6f}</td>
                        <td>{row['std_mae']:.6f}</td>
                        <td>{row['count']}</td>
                    </tr>
            """)
        
        f.write("""
                </table>
            </div>
            
            <div class='section'>
                <h2>Key Findings</h2>
                <ul>
        """)
        
        # Add key findings based on the results
        baseline_metrics = metrics_summary[metrics_summary['feature_name'] == 'vst_raw']
        if not baseline_metrics.empty:
            baseline_rmse = baseline_metrics.iloc[0]['mean_rmse']
            best_metrics = metrics_summary[metrics_summary['mean_rmse'] == metrics_summary['mean_rmse'].min()]
            best_rmse = best_metrics.iloc[0]['mean_rmse']
            improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100
            
            f.write(f"""
                    <li>The best performing feature combination is <strong>{best_feature}</strong>.</li>
                    <li>Using {best_feature} provides a <strong>{improvement:.2f}%</strong> improvement in RMSE compared to using only VST_raw.</li>
            """)
            
            # Add more insights based on the data
            if 'temperature' in best_feature and 'rainfall' in best_feature:
                f.write("""
                    <li>Including both temperature and rainfall data together provides the best predictive performance.</li>
                """)
            elif 'temperature' in best_feature:
                f.write("""
                    <li>Temperature data appears to be more important than rainfall data for prediction.</li>
                """)
            elif 'rainfall' in best_feature:
                f.write("""
                    <li>Rainfall data appears to be more important than temperature data for prediction.</li>
                """)
        
        f.write("""
                </ul>
            </div>
            
            <div class='section'>
                <h2>Visualizations</h2>
                
                <div class='plot'>
                    <h3>Feature Impact Comparison</h3>
                    <img src='feature_impact_comparison.png' alt='Feature Impact Comparison'>
                    <p>Comparison of RMSE and MAE across different feature combinations.</p>
                </div>
                
                <div class='plot'>
                    <h3>Percentage Improvement</h3>
                    <img src='feature_impact_percentage_improvement.png' alt='Percentage Improvement'>
                    <p>Percentage improvement in error metrics relative to using only VST_raw.</p>
                </div>
                
                <div class='plot'>
                    <h3>Learning Curves</h3>
                    <img src='feature_impact_learning_curves.png' alt='Learning Curves'>
                    <p>Validation loss curves for the best model of each feature combination.</p>
                </div>
                
                <div class='plot'>
                    <h3>Forecast Comparison</h3>
                    <img src='feature_impact_forecast_comparison.png' alt='Forecast Comparison'>
                    <p>Comparison of forecasts from the best model of each feature combination.</p>
                </div>
            </div>
            
            <div class='section'>
                <h2>Conclusion</h2>
                <p>
                    This analysis demonstrates the impact of including additional meteorological features 
                    (temperature and rainfall) on the performance of the LSTM forecasting model. 
                    The results show that """ + (f"the combination of {best_feature} yields the best performance, " 
                    if 'best_feature' in locals() else "different feature combinations affect model performance, ") + """
                    highlighting the importance of feature selection in time series forecasting.
                </p>
            </div>
        </body>
        </html>
        """)
    
    print(f"Feature impact report generated at {report_path}")

def plot_feature_decision_tree(results, output_dir):
    """
    Create a decision tree visualization to show which features are most important
    and how they interact with other parameters.
    
    Args:
        results: List of trial results
        output_dir: Directory to save the plot
    """
    try:
        from sklearn.tree import DecisionTreeRegressor, plot_tree
        
        # Extract data
        tree_data = []
        
        for result in results:
            if 'params' in result and 'value' in result and not np.isinf(result['value']):
                params = result.get('params', {})
                
                # Create a row with all parameters
                row = {}
                
                # Add feature selection as one-hot encoded columns
                if 'feature_cols' in params:
                    feature_set = params['feature_cols']
                    if isinstance(feature_set, list):
                        # Create indicator for each possible feature
                        for feature in ['vst_raw', 'temperature', 'rainfall']:
                            row[f'has_{feature}'] = 1 if feature in feature_set else 0
                
                # Add other parameters
                for param in ['sequence_length', 'hidden_size', 'num_layers', 'dropout', 
                             'learning_rate', 'batch_size', 'epochs', 'patience']:
                    if param in params:
                        row[param] = params[param]
                
                row['value'] = result['value']
                tree_data.append(row)
        
        if not tree_data or len(tree_data) < 10:
            print("Not enough data for decision tree analysis")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(tree_data)
        
        # Prepare features and target
        X = df.drop('value', axis=1)
        y = df['value']
        
        # Train a decision tree
        tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3)
        tree.fit(X, y)
        
        # Create visualization
        plt.figure(figsize=(20, 12))
        plot_tree(
            tree, 
            feature_names=X.columns,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=3
        )
        
        plt.title('Decision Tree for Hyperparameter Importance', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'feature_decision_tree.png', dpi=300)
        plt.close()
        
        # Calculate and print feature importances
        importances = tree.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Create bar chart of feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance from Decision Tree', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'feature_tree_importance.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating decision tree visualization: {e}")
        import traceback
        print(traceback.format_exc())
        
if __name__ == "__main__":
    # This allows running the analysis directly
    print("This module provides feature impact analysis functionality.")
    print("Import and use the run_feature_impact_analysis function from your main script.") 