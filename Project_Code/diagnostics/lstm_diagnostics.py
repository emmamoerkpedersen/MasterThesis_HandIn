"""Diagnostic tools for LSTM model performance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_history(history: dict, station_name: str, output_dir: Path):
    """
    Plot training and validation losses over epochs.
    
    Args:
        history: Dictionary containing training metrics
        station_name: Name of the station for plot title
        output_dir: Directory to save diagnostic plots
    """
    diagnostic_dir = output_dir / "diagnostics" / "lstm"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create static plot
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Training History - {station_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(diagnostic_dir / f"{station_name}_training_history.png")
    plt.close()
    
    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['train_loss'], name='Training Loss'))
    fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
    fig.update_layout(
        title=f'LSTM Training History - {station_name}',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x'
    )
    fig.write_html(diagnostic_dir / f"{station_name}_training_history.html")

def plot_detection_results(original_data: pd.DataFrame,
                         lstm_predictions: np.ndarray,
                         anomaly_flags: np.ndarray,
                         confidence_scores: np.ndarray,
                         station_name: str,
                         output_dir: Path):
    """
    Create visualization of anomaly detection results.
    
    Args:
        original_data: Original time series data
        lstm_predictions: LSTM model predictions
        anomaly_flags: Boolean array indicating detected anomalies
        confidence_scores: Confidence scores for anomaly detection
        station_name: Name of the station
        output_dir: Directory to save diagnostic plots
    """
    diagnostic_dir = output_dir / "diagnostics" / "lstm"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Create interactive plot
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=['Water Level and Predictions',
                                     'Detection Confidence Scores'])
    
    # Original data and predictions
    fig.add_trace(
        go.Scatter(x=original_data.index, y=original_data['Value'],
                  name='Original Data', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=original_data.index, y=lstm_predictions,
                  name='LSTM Predictions', line=dict(color='green')),
        row=1, col=1
    )
    
    # Highlight anomalies
    anomaly_points = original_data[anomaly_flags]
    fig.add_trace(
        go.Scatter(x=anomaly_points.index, y=anomaly_points['Value'],
                  mode='markers', name='Detected Anomalies',
                  marker=dict(color='red', size=8)),
        row=1, col=1
    )
    
    # Confidence scores
    fig.add_trace(
        go.Scatter(x=original_data.index, y=confidence_scores,
                  name='Confidence Score', line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'LSTM Anomaly Detection Results - {station_name}',
        height=800,
        showlegend=True,
        hovermode='x'
    )
    
    fig.write_html(diagnostic_dir / f"{station_name}_detection_results.html")

def generate_lstm_report(results: dict, output_dir: Path):
    """
    Generate detailed report of LSTM model performance.
    
    Args:
        results: Dictionary containing results for each station
        output_dir: Directory to save report
    """
    report_dir = output_dir / "diagnostics" / "lstm"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_dir / "lstm_report.txt", "w") as f:
        f.write("LSTM Model Performance Report\n")
        f.write("===========================\n\n")
        
        for station_name, station_results in results.items():
            f.write(f"\nStation: {station_name}\n")
            f.write("-" * (len(station_name) + 9) + "\n")
            
            # Training statistics
            f.write("\nTraining Statistics:\n")
            f.write(f"  Final training loss: {station_results['final_train_loss']:.4f}\n")
            f.write(f"  Final validation loss: {station_results['final_val_loss']:.4f}\n")
            f.write(f"  Training epochs: {station_results['epochs']}\n")
            
            # Detection statistics
            f.write("\nDetection Statistics:\n")
            f.write(f"  Total anomalies detected: {station_results['num_anomalies']}\n")
            f.write(f"  Average confidence score: {station_results['avg_confidence']:.4f}\n")
            
            # Confidence distribution
            confidence_ranges = {
                'High (>0.9)': station_results['high_confidence'],
                'Medium (0.7-0.9)': station_results['medium_confidence'],
                'Low (<0.7)': station_results['low_confidence']
            }
            f.write("\nConfidence Distribution:\n")
            for range_name, count in confidence_ranges.items():
                f.write(f"  {range_name}: {count} anomalies\n")
            
            f.write("\n" + "="*50 + "\n") 