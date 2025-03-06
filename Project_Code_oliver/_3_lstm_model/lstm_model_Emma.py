"""
LSTM Model for Time Series Anomaly Detection and Imputation

This module implements a Long Short-Term Memory (LSTM) neural network for analyzing
time series data with multiple features (VST, rainfall, and temperature).

Key Components:
- Data preprocessing and sequence creation
- LSTM model architecture
- Training pipeline with early stopping
- Prediction and analysis utilities

Dependencies:
    torch
    numpy
    pandas
    matplotlib
    sklearn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import copy
from sklearn.svm import OneClassSVM


def create_sequences(raw_data: torch.Tensor, rainfall_data: torch.Tensor, temperature_data: torch.Tensor, window_size: int = 48) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding window sequences for time series prediction.

    Args:
        raw_data: Tensor of VST values [N, 1]
        rainfall_data: Tensor of rainfall measurements [N, 1]
        temperature_data: Tensor of temperature measurements [N, 1]
        window_size: Number of time steps in each sequence

    Returns:
        Tuple containing:
            - Tensor of input sequences (batch_size, window_size, features)
            - Tensor of target values (batch_size, 1)
    """
    sequences = []
    targets = []
    
    # Check dimensions
    if len(raw_data.shape) == 1:
        raw_data = raw_data.unsqueeze(1)
    if len(rainfall_data.shape) == 1:
        rainfall_data = rainfall_data.unsqueeze(1)
    if len(temperature_data.shape) == 1:
        temperature_data = temperature_data.unsqueeze(1)
    
    # Ensure all data have the same length
    min_length = min(len(raw_data), len(rainfall_data), len(temperature_data))
    raw_data = raw_data[:min_length]
    rainfall_data = rainfall_data[:min_length]
    temperature_data = temperature_data[:min_length]
    
    # Create a combined feature tensor [N, 3]
    combined_data = torch.cat([raw_data, rainfall_data, temperature_data], dim=1)
    
    for i in range(len(combined_data) - window_size):
        # Extract window of multivariate data [window_size, 3]
        sequence = combined_data[i:i + window_size]
        
        # Target is the next VST value after the window
        target = raw_data[i + window_size]
        
        sequences.append(sequence)
        targets.append(target)
    
    # Stack sequences: [batch_size, window_size, 3]
    # Stack targets: [batch_size, 1]
    return torch.stack(sequences), torch.stack(targets)

def process_data(data_dict: Dict, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process and normalize input data for model training.

    Args:
        data_dict: Dictionary containing 'vst_raw', 'rainfall', and 'temperature' data
        device: Device to place tensors on ('cpu')

    Returns:
        Tuple containing:
            - Processed input sequences
            - Target values
    """
    # Convert input features to float type first
    vst_data = torch.FloatTensor(data_dict['vst_raw']['vst_raw'].astype(float)).unsqueeze(-1)
    rainfall_data = torch.FloatTensor(data_dict['rainfall']['rainfall'].astype(float)).unsqueeze(-1)
    temperature_data = torch.FloatTensor(data_dict['temperature']['temperature'].astype(float)).unsqueeze(-1)

    # Make sure all data have the same length
    min_length = min(vst_data.size(0), rainfall_data.size(0), temperature_data.size(0))
    vst_data = vst_data[:min_length]
    rainfall_data = rainfall_data[:min_length]
    temperature_data = temperature_data[:min_length]
    
    # Initialize and fit scalers
    vst_scaler = MinMaxScaler(feature_range=(0, 1))
    rain_scaler = MinMaxScaler(feature_range=(0, 1))
    temp_scaler = MinMaxScaler(feature_range=(0, 1))

    # Convert to numpy, scale, and back to torch
    vst_data_normalized = torch.FloatTensor(vst_scaler.fit_transform(vst_data.numpy()))
    rainfall_data_normalized = torch.FloatTensor(rain_scaler.fit_transform(rainfall_data.numpy()))
    temperature_data_normalized = torch.FloatTensor(temp_scaler.fit_transform(temperature_data.numpy()))
    
    # Create sequences
    sequences, targets = create_sequences(vst_data_normalized, rainfall_data_normalized, temperature_data_normalized)

    # Store scalers as attributes
    process_data.vst_scaler = vst_scaler
    process_data.rain_scaler = rain_scaler
    process_data.temp_scaler = temp_scaler
    
    return sequences.to(device), targets.to(device)

def smooth_hinge_loss(x, tau=10):
    """
    Smooth approximation of the hinge loss function for One-Class SVM.
    
    For One-Class SVM, we want to penalize points with x > 0 (negative decision scores).
    
    Args:
        x: Input tensor (typically -decision_score)
        tau: Smoothing parameter (higher values = sharper approximation)
        
    Returns:
        Smoothed hinge loss values
    """
    # For numerical stability, use a piecewise function:
    # 1. If x <= 0, loss is 0 (point is correctly classified)
    # 2. If x > 0, loss increases linearly (point is potentially an anomaly)
    return torch.where(
        x <= 0,
        torch.zeros_like(x),
        x
    )

class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for anomaly detection.
    
    This model extracts fixed-length feature representations from variable-length
    time series data using an LSTM network followed by mean pooling.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of features in the hidden state
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_mean = torch.mean(lstm_out, dim=1)  # Mean pooling over time
        return h_mean

class LSTMOcSVM(nn.Module):
    """
    End-to-end LSTM-based One-Class SVM for anomaly detection.
    
    This model combines an LSTM feature extractor with a differentiable
    One-Class SVM formulation that can be trained jointly using backpropagation.
    
    Args:
        input_size: Number of input features per timestep
        hidden_size: Number of features in the hidden state
        lambda_param: Regularization parameter (default=0.1)
        tau: Smoothing parameter for hinge loss (default=10)
        num_layers: Number of recurrent layers (default=1)
        dropout: Dropout rate for regularization (default=0.0)
    """
    def __init__(self, input_size, hidden_size, lambda_param=0.1, tau=10, num_layers=1, dropout=0.0):
        super(LSTMOcSVM, self).__init__()
        self.lstm = LSTMFeatureExtractor(input_size, hidden_size, num_layers, dropout)
        self.w = nn.Parameter(torch.randn(hidden_size))
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.lambda_param = lambda_param
        self.tau = tau
        self.bias = nn.Parameter(torch.tensor(0.0))  # Add bias term
    
    def forward(self, x):
        h = self.lstm(x)
        decision_score = torch.matmul(h, self.w) + self.bias - self.rho  # Add bias
        return decision_score, h
    
    def loss(self, x):
        """
        Compute the One-Class SVM loss.
        
        This loss function aims to find a hyperplane that separates the data from the origin
        with maximum margin, which is suitable for anomaly detection.
        
        Args:
            x: Input tensor of shape [batch_size, window_size, features]
            
        Returns:
            Loss value
        """
        h = self.lstm(x)
        # Calculate decision scores: w^T * h + bias - rho
        decision_score = torch.matmul(h, self.w) + self.bias - self.rho
        
        # Calculate hinge loss term (slack variables)
        # For One-Class SVM, we want most points to have positive decision scores
        # Points with negative scores are potential anomalies
        slack = smooth_hinge_loss(-decision_score, self.tau)
        
        # Regularization term (minimize ||w||^2)
        reg_term = 0.5 * torch.norm(self.w, p=2)**2
        
        # Margin term (maximize rho)
        margin_term = self.rho
        
        # Final loss: regularization + slack penalty - margin
        return reg_term + (1 / (len(x) * self.lambda_param)) * torch.sum(slack) - margin_term

def train_model(model: LSTMOcSVM, data_dict: Dict, epochs: int = 100, lr: float = 0.01, 
                device: str = 'cpu', patience: int = 20) -> LSTMOcSVM:
    """
    Train the LSTM-OcSVM model end-to-end.
    
    Args:
        model: The LSTMOcSVM model to train
        data_dict: Dictionary containing 'vst_raw', 'rainfall', and 'temperature' data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to place tensors on ('cpu' or 'cuda')
        patience: Number of epochs to wait for improvement before early stopping
        
    Returns:
        Trained LSTMOcSVM model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    try:
        # Get sequences
        sequences, _ = process_data(data_dict, device)
        
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Calculate loss
            loss = model.loss(sequences)
            
            # Update model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Learning rate scheduling
            scheduler.step(loss.item())
            
            # Check for improvement
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs")
                break
            
            if (epoch + 1) % 10 == 0:
                # Get current anomaly predictions
                with torch.no_grad():
                    decision_scores, _ = model(sequences)
                    predictions = (decision_scores < 0).cpu().numpy().astype(int) * -2 + 1
                    anomaly_rate = (predictions == -1).mean() * 100
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                    print(f'Current anomaly rate: {anomaly_rate:.2f}%')
                    print(f'Current rho value: {model.rho.item():.4f}')
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        # Return the model in its current state
        return model

def detect_anomalies(model: LSTMOcSVM, data_dict: Dict, 
                    device: str = 'cpu', threshold: Optional[float] = None,
                    contamination: float = 0.05) -> np.ndarray:
    """
    Detect anomalies using the trained LSTM-OcSVM model.
    
    Args:
        model: Trained LSTMOcSVM model
        data_dict: Dictionary containing 'vst_raw', 'rainfall', and 'temperature' data
        device: Device to place tensors on ('cpu' or 'cuda')
        threshold: Optional custom threshold for anomaly detection
        contamination: Expected proportion of anomalies (default=0.05 or 5%)
        
    Returns:
        Array of predictions (1 for normal, -1 for anomalies)
    """
    model.eval()
    
    try:
        sequences, _ = process_data(data_dict, device)
        
        with torch.no_grad():
            decision_scores, _ = model(sequences)
            decision_scores = decision_scores.cpu().numpy()
        
        # If threshold is provided, use it
        if threshold is not None:
            predictions = (decision_scores < threshold).astype(int) * -2 + 1
        else:
            # Use percentile-based threshold based on contamination parameter
            # This assumes that approximately 'contamination' proportion of the data are anomalies
            threshold = np.percentile(decision_scores, contamination * 100)
            predictions = (decision_scores < threshold).astype(int) * -2 + 1
            print(f"Auto-selected threshold: {threshold:.4f} (based on {contamination:.1%} contamination)")
        
        # Count anomalies
        num_anomalies = (predictions == -1).sum()
        print(f"Detected {num_anomalies} anomalies ({num_anomalies/len(predictions):.2%} of data)")
        
        return predictions
    
    except Exception as e:
        print(f"Error in anomaly detection: {str(e)}")
        # Return empty array in case of error
        return np.array([])

def analyze_results(predictions: np.ndarray, data_dict: Dict) -> pd.DataFrame:
    """
    Analyze anomaly detection results with proper scaling consideration.
    
    Args:
        predictions: Array of predictions (1 for normal, -1 for anomalies)
        data_dict: Dictionary containing 'vst_raw', 'rainfall', and 'temperature' data
        
    Returns:
        DataFrame with results including timestamps, values, and anomaly flags
    """
    try:
        window_size = 48
        
        # Validate input data
        if len(predictions) == 0:
            print("Warning: Empty predictions array")
            return pd.DataFrame()
            
        if not all(key in data_dict for key in ['vst_raw', 'rainfall', 'temperature']):
            print("Warning: Missing required data keys")
            return pd.DataFrame()
        
        # Get the original data for visualization
        timestamps = data_dict['vst_raw'].index[window_size:]
        vst_values = data_dict['vst_raw']['vst_raw'].values[window_size:]
        rainfall = data_dict['rainfall']['rainfall'].values[window_size:]
        temperature = data_dict['temperature']['temperature'].values[window_size:]
        
        # Ensure all arrays have valid lengths
        n_points = min(len(predictions), len(timestamps), len(vst_values), 
                      len(rainfall), len(temperature))
        
        if n_points == 0:
            print("Warning: No valid data points after alignment")
            return pd.DataFrame()
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'Timestamp': timestamps[:n_points],
            'VST': vst_values[:n_points],
            'Rainfall': rainfall[:n_points],
            'Temperature': temperature[:n_points],
            'Is_Anomaly': predictions[:n_points] == -1
        })
        
        # Plot results
        plt.figure(figsize=(15, 8))
        
        # Plot the time series
        plt.plot(results_df['Timestamp'], results_df['VST'], label='VST', alpha=0.6)
        
        # Highlight anomalies
        anomalies = results_df[results_df['Is_Anomaly']]
        if not anomalies.empty:
            plt.scatter(anomalies['Timestamp'], anomalies['VST'], 
                        color='red', label='Anomalies', alpha=0.6)
        
        plt.title('VST Time Series with Detected Anomalies')
        plt.xlabel('Time')
        plt.ylabel('VST Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Add a second plot showing the scaled data and decision boundary if available
        if hasattr(process_data, 'vst_scaler'):
            plt.figure(figsize=(15, 8))
            
            # Get scaled values
            vst_scaled = process_data.vst_scaler.transform(vst_values[:n_points].reshape(-1, 1)).flatten()
            
            plt.plot(results_df['Timestamp'], vst_scaled, label='Scaled VST', color='blue', alpha=0.6)
            
            # Highlight anomalies in scaled space
            if not anomalies.empty:
                anomalies_scaled = vst_scaled[results_df['Is_Anomaly'].values]
                anomaly_timestamps = results_df.loc[results_df['Is_Anomaly'], 'Timestamp']
                
                plt.scatter(anomaly_timestamps, anomalies_scaled, 
                            color='red', label='Anomalies (scaled)', alpha=0.6)
            
            plt.title('Scaled VST Time Series with Detected Anomalies')
            plt.xlabel('Time')
            plt.ylabel('Scaled VST Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        
        plt.show()
        
        # Print statistics
        anomaly_percentage = (predictions[:n_points] == -1).mean() * 100
        print(f"\nAnomaly Detection Results:")
        print(f"Total points analyzed: {n_points}")
        print(f"Anomalies detected: {(predictions[:n_points] == -1).sum()}")
        print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
        
        return results_df
        
    except Exception as e:
        print(f"Error in analyzing results: {str(e)}")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Load and prepare data
    stationID = '21006845'
    data_dict = pd.read_pickle('../data_utils/Sample data/preprocessed_data.pkl')
    data_dict = data_dict[stationID]
    
    # Create a new dictionary with truncated data
    data_dict_truncated = {}
    # Iterate through each dataframe in the station data
    for key, df in data_dict.items():
        # Use a reasonable amount of data for demonstration
        data_dict_truncated[key] = df.iloc[100000:110000]
    # Use the truncated data for model training
    data_dict = data_dict_truncated
    
    # Initialize model with better hyperparameters
    # input_size=3 because we have 3 features (VST, rainfall, temperature)
    model = LSTMOcSVM(
        input_size=3,  
        hidden_size=128,  # Larger hidden size for better representation
        lambda_param=0.01,  # Smaller lambda for less regularization
        tau=1.0,  # Smaller tau for smoother loss
        num_layers=2,  # Use 2 LSTM layers
        dropout=0.2  # Add dropout for regularization
    )
    
    print("Training model...")
    # Train the model with more epochs and early stopping
    trained_model = train_model(
        model, 
        data_dict, 
        epochs=200,  # More epochs
        lr=0.001,  # Lower learning rate
        patience=30  # More patience for early stopping
    )
    
    print("\nDetecting anomalies...")
    # Try different contamination rates
    for contamination in [0.01, 0.05, 0.1]:
        print(f"\nTesting with contamination rate: {contamination:.2%}")
        predictions = detect_anomalies(
            trained_model, 
            data_dict, 
            contamination=contamination
        )
        results_df = analyze_results(predictions, data_dict)
    
    # Try specific thresholds if needed
    print("\nTesting with specific thresholds:")
    for threshold in [-0.5, -0.3, -0.1, 0.0, 0.1]:
        print(f"\nTesting threshold: {threshold}")
        predictions = detect_anomalies(trained_model, data_dict, threshold=threshold)
        results_df = analyze_results(predictions, data_dict)



