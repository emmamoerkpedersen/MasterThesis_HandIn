"""
Anomaly detection utilities for water level data.

This module provides functions for detecting anomalies in water level data
using machine learning methods like Isolation Forest and One-Class SVM.
These methods can be applied to model residuals to identify unexpected patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def detect_anomalies_isolation_forest(
    data,
    contamination=0.1,
    n_estimators=100,
    max_features=1.0,
    bootstrap=False,
    random_state=42
):
    """
    Detect anomalies in time series data using Isolation Forest.
    
    Args:
        data: pandas Series or numpy array containing time series data
        contamination: expected proportion of anomalies (default 0.1)
        n_estimators: number of trees in the forest (default 100)
        max_features: features to draw when building trees (default 1.0 = all)
        bootstrap: whether to use bootstrap when building trees (default False)
        random_state: random state for reproducibility (default 42)
        
    Returns:
        DataFrame with original data and anomaly indicators
    """
    # Convert input to numpy array if it's a Series
    if isinstance(data, pd.Series):
        data_index = data.index
        data_values = data.values.reshape(-1, 1)
    else:
        data_index = np.arange(len(data))
        data_values = np.array(data).reshape(-1, 1)
        
    # Initialize and fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state
    )
    
    # Fit and predict
    anomaly_labels = iso_forest.fit_predict(data_values)
    
    # Convert predictions (-1 for anomalies, 1 for normal) to 0/1 indicator (1 for anomalies)
    anomaly_indicator = np.where(anomaly_labels == -1, 1, 0)
    anomaly_scores = iso_forest.decision_function(data_values)
    
    # Create a DataFrame with results
    result_df = pd.DataFrame({
        'value': data_values.flatten(),
        'anomaly': anomaly_indicator,
        'score': anomaly_scores
    }, index=data_index)
    
    return result_df

def detect_anomalies_one_class_svm(
    data,
    nu=0.1,
    kernel='rbf',
    gamma='scale'
):
    """
    Detect anomalies in time series data using One-Class SVM.
    
    Args:
        data: pandas Series or numpy array containing time series data
        nu: upper bound on the fraction of training errors (default 0.1)
        kernel: kernel type (default 'rbf')
        gamma: kernel coefficient (default 'scale')
        
    Returns:
        DataFrame with original data and anomaly indicators
    """
    # Convert input to numpy array if it's a Series
    if isinstance(data, pd.Series):
        data_index = data.index
        data_values = data.values.reshape(-1, 1)
    else:
        data_index = np.arange(len(data))
        data_values = np.array(data).reshape(-1, 1)
    
    # Scale data for better SVM performance
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)
    
    # Initialize and fit One-Class SVM
    ocsvm = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma
    )
    
    # Fit and predict
    anomaly_labels = ocsvm.fit_predict(data_scaled)
    
    # Convert predictions (1 for normal, -1 for anomalies) to 0/1 indicator (1 for anomalies)
    anomaly_indicator = np.where(anomaly_labels == -1, 1, 0)
    
    # Create a DataFrame with results
    result_df = pd.DataFrame({
        'value': data_values.flatten(),
        'anomaly': anomaly_indicator,
    }, index=data_index)
    
    return result_df

def detect_anomalies_from_residuals(
    actual_values,
    predicted_values,
    method='isolation_forest',
    threshold_std=3.0,
    contamination=0.05,
    apply_smoothing=True,
    smoothing_window=5
):
    """
    Detect anomalies by analyzing residuals between actual and predicted values.
    
    Args:
        actual_values: pandas Series or numpy array of actual values
        predicted_values: pandas Series or numpy array of predicted values
        method: anomaly detection method ('isolation_forest', 'one_class_svm', or 'statistical')
        threshold_std: number of standard deviations for statistical method (default 3.0)
        contamination: expected proportion of anomalies for ML methods (default 0.05)
        apply_smoothing: whether to smooth residuals before anomaly detection (default True)
        smoothing_window: window size for residual smoothing (default 5)
        
    Returns:
        DataFrame with actual values, predictions, residuals, and anomaly indicators
    """
    # Ensure inputs are pandas Series with matching indices
    if isinstance(actual_values, pd.Series) and isinstance(predicted_values, pd.Series):
        # Make sure indices align
        common_idx = actual_values.index.intersection(predicted_values.index)
        actual = actual_values.loc[common_idx]
        predicted = predicted_values.loc[common_idx]
    else:
        # Convert to pandas Series if they aren't already
        actual = pd.Series(actual_values)
        predicted = pd.Series(predicted_values, index=actual.index)
    
    # Calculate residuals (actual - predicted)
    residuals = actual - predicted
    
    # Apply smoothing to residuals if enabled
    if apply_smoothing:
        smoothed_residuals = residuals.rolling(window=smoothing_window, center=True, min_periods=1).mean()
    else:
        smoothed_residuals = residuals
    
    # Detect anomalies using the specified method
    if method == 'isolation_forest':
        # Apply Isolation Forest to residuals
        anomaly_results = detect_anomalies_isolation_forest(
            smoothed_residuals,
            contamination=contamination
        )
        anomaly_indicator = anomaly_results['anomaly']
        
    elif method == 'one_class_svm':
        # Apply One-Class SVM to residuals
        anomaly_results = detect_anomalies_one_class_svm(
            smoothed_residuals,
            nu=contamination
        )
        anomaly_indicator = anomaly_results['anomaly']
        
    elif method == 'statistical':
        # Simple statistical approach: mark as anomaly if residual exceeds threshold_std * std_dev
        residual_mean = smoothed_residuals.mean()
        residual_std = smoothed_residuals.std()
        lower_bound = residual_mean - threshold_std * residual_std
        upper_bound = residual_mean + threshold_std * residual_std
        anomaly_indicator = ((smoothed_residuals < lower_bound) | (smoothed_residuals > upper_bound)).astype(int)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'residual': residuals,
        'smoothed_residual': smoothed_residuals,
        'anomaly': anomaly_indicator
    })
    
    return result_df

def visualize_anomalies(result_df, title='Anomaly Detection Results', figsize=(12, 10)):
    """
    Visualize the anomaly detection results.
    
    Args:
        result_df: DataFrame with actual values, predictions, residuals, and anomaly indicators
        title: plot title (default 'Anomaly Detection Results')
        figsize: figure size as tuple (width, height) (default (12, 10))
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Actual vs Predicted values
    axes[0].plot(result_df.index, result_df['actual'], 'b-', label='Actual')
    axes[0].plot(result_df.index, result_df['predicted'], 'r-', label='Predicted')
    
    # Highlight anomalies in the actual data
    anomaly_points = result_df[result_df['anomaly'] == 1]
    axes[0].scatter(anomaly_points.index, anomaly_points['actual'], 
                   color='orange', marker='o', s=50, label='Anomalies')
    
    axes[0].set_title(f'{title} - Actual vs Predicted')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Residuals with anomalies highlighted
    axes[1].plot(result_df.index, result_df['residual'], 'g-', label='Residual')
    axes[1].plot(result_df.index, result_df['smoothed_residual'], 'k--', alpha=0.5, label='Smoothed Residual')
    
    # Highlight anomalies in residuals
    axes[1].scatter(anomaly_points.index, anomaly_points['residual'], 
                   color='red', marker='x', s=50, label='Anomalies')
    
    axes[1].set_title('Residuals (Actual - Predicted)')
    axes[1].set_ylabel('Residual')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Anomaly indicator (binary)
    axes[2].fill_between(result_df.index, 0, result_df['anomaly'], color='red', alpha=0.3)
    axes[2].set_title('Anomaly Indicator')
    axes[2].set_ylabel('Anomaly (1=Yes)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True)
    
    plt.tight_layout()
    return fig

def evaluate_anomaly_detection(anomaly_indicators, known_anomalies):
    """
    Evaluate anomaly detection performance using known anomalies.
    
    Args:
        anomaly_indicators: Series or array with detected anomalies (0/1)
        known_anomalies: Series or array with known/true anomalies (0/1)
        
    Returns:
        dict with confusion matrix and metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(known_anomalies, anomaly_indicators)
    
    try:
        # Calculate precision, recall, F1 score
        precision = precision_score(known_anomalies, anomaly_indicators)
        recall = recall_score(known_anomalies, anomaly_indicators)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (true negative rate)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            'confusion_matrix': cm,
            'error': str(e)
        }

def run_anomaly_detection_pipeline(
    actual_values, 
    predicted_values,
    output_path=None,
    methods=['isolation_forest', 'one_class_svm', 'statistical'],
    contamination=0.05
):
    """
    Run the complete anomaly detection pipeline and create visualizations.
    
    Args:
        actual_values: pandas Series of actual values
        predicted_values: pandas Series of predicted values
        output_path: path to save output visualizations (default None, no saving)
        methods: list of methods to use for anomaly detection
        contamination: expected proportion of anomalies (default 0.05)
        
    Returns:
        dict with anomaly detection results for each method
    """
    results = {}
    
    for method in methods:
        print(f"Running anomaly detection with method: {method}")
        
        # Detect anomalies using the current method
        result_df = detect_anomalies_from_residuals(
            actual_values,
            predicted_values,
            method=method,
            contamination=contamination
        )
        
        # Generate visualization
        fig = visualize_anomalies(
            result_df, 
            title=f'Anomaly Detection - {method.replace("_", " ").title()}'
        )
        
        # Save the visualization if path is provided
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path = output_dir / f"anomalies_{method}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"  Saved visualization to {fig_path}")
        
        # Store results
        results[method] = {
            'dataframe': result_df,
            'anomaly_count': result_df['anomaly'].sum(),
            'anomaly_percentage': (result_df['anomaly'].sum() / len(result_df)) * 100
        }
        
        print(f"  Detected {results[method]['anomaly_count']} anomalies "
              f"({results[method]['anomaly_percentage']:.2f}%)")
    
    return results 