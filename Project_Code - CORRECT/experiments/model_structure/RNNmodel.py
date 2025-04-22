import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to allow importing from _3_lstm_model
sys.path.append(str(Path(__file__).parent.parent.parent))
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, water_level_index=0, meteo_level_index=1):
        super(RNNModel, self).__init__()
        self.model_name = 'AnomalyAwareLSTM'
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.water_level_index = water_level_index
        self.meteo_level_index = meteo_level_index
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            - predictions: Tensor of shape (batch_size, seq_len, output_size)
            - anomaly_flags: Tensor of shape (batch_size, seq_len) indicating anomalies
        """
        batch_size, seq_len, input_size = x.size()
        device = x.device

        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)

        preds = []
        anomaly_flags_list = []

        for t in range(seq_len):
            if t == 0:
                input_t = x[:, t, :]  # use first input as-is
                anomaly_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                # Anomaly detection on previous prediction
                anomaly_flags = self.check_anomaly(pred, x[:, t, self.water_level_index].unsqueeze(1))
                
                # Create a new input tensor
                new_input = x[:, t, :].clone()
                
                # Replace water level with predicted value only for anomalous samples
                # Ensure all tensors have compatible shapes
                water_level_values = new_input[:, self.water_level_index]
                predicted_values = pred.squeeze(1)
                
                # Use torch.where with properly shaped tensors
                new_input[:, self.water_level_index] = torch.where(
                    anomaly_flags,
                    predicted_values,
                    water_level_values
                )
                
                input_t = new_input

            # Apply dropout to input
            input_t = self.dropout(input_t)
            
            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))
            pred = self.linear(h_t)  # Shape: (batch_size, output_size)
            preds.append(pred.unsqueeze(1))  # For seq build-up
            anomaly_flags_list.append(anomaly_flags.unsqueeze(1))  # For seq build-up
        
        predictions = torch.cat(preds, dim=1)  # Final shape: (batch_size, seq_len, output_size)
        anomaly_flags = torch.cat(anomaly_flags_list, dim=1)  # Final shape: (batch_size, seq_len)
        
        return predictions, anomaly_flags

    def check_anomaly(self, pred, actual, threshold=0.1):
        """
        pred: predicted water level (batch_size, 1)
        actual: actual water level (batch_size, 1)
        Returns: Boolean tensor of shape (batch_size,) indicating anomalies
        """
        error = torch.abs(pred - actual)
        return (error > threshold).squeeze(1)  # returns boolean flags per batch

# Configuration for the data preprocessor
config = {
    'feature_cols': ['vst_raw', 'temperature', 'rainfall'],
    'output_features': ['vst_raw'],
    'feature_stations': [
        {'station_id': '21006845', 'features': ['vst_raw', 'rainfall']},
        {'station_id': '21006847', 'features': ['vst_raw', 'rainfall']}
    ],
    'sequence_length': 1000,
    'use_time_features': False,
    'use_cumulative_features': False
}

# Initialize the data preprocessor
project_root = Path(__file__).parent.parent.parent
preprocessor = DataPreprocessor(config)

# Load and split the data
station_id = '21006845'  # Main station ID
train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)

# Prepare the data for training
X_train, y_train = preprocessor.prepare_data(train_data, is_training=True)
X_val, y_val = preprocessor.prepare_data(val_data, is_training=False)
X_test, y_test = preprocessor.prepare_data(test_data, is_training=False)

# Clean the input data by replacing NaN values with -1
# But keep the target data as is
X_train = torch.nan_to_num(X_train, nan=-1.0)
X_val = torch.nan_to_num(X_val, nan=-1.0)
X_test = torch.nan_to_num(X_test, nan=-1.0)

# Get input and output sizes from the data
input_size = X_train.shape[2]  # Number of features
output_size = y_train.shape[2]  # Number of output features
hidden_size = 164

# Initialize the model
model = RNNModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    outputs, _ = model(X_train)  # Ignore anomaly flags during training
    
    # Create mask for non-NaN values in targets
    non_nan_mask = ~torch.isnan(y_train)
    
    # Reshape outputs and targets to 2D for easier masking
    # Original shapes: (batch_size, seq_len, output_size)
    batch_size, seq_len, feat_size = outputs.shape
    outputs_reshaped = outputs.reshape(-1, feat_size)  # Shape: (batch_size * seq_len, output_size)
    y_train_reshaped = y_train.reshape(-1, feat_size)  # Shape: (batch_size * seq_len, output_size)
    non_nan_mask_reshaped = non_nan_mask.reshape(-1, feat_size)  # Shape: (batch_size * seq_len, output_size)
    
    # Apply mask to get valid outputs and targets
    valid_outputs = outputs_reshaped[non_nan_mask_reshaped]
    valid_targets = y_train_reshaped[non_nan_mask_reshaped]

    # Skip if no valid targets
    if valid_targets.size(0) == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]: No valid targets, skipping")
        continue
    
    # Calculate loss only on valid targets
    loss = criterion(valid_outputs, valid_targets)
    
    # Check if loss is NaN
    if torch.isnan(loss):
        print(f"Epoch [{epoch+1}/{num_epochs}]: Loss is NaN, skipping")
        continue
    
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients if they're too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs, _ = model(X_val)  # Ignore anomaly flags during validation
        
        # Create mask for non-NaN values in validation targets
        val_non_nan_mask = ~torch.isnan(y_val)
        val_outputs_reshaped = val_outputs.reshape(-1, feat_size)
        y_val_reshaped = y_val.reshape(-1, feat_size)
        val_non_nan_mask_reshaped = val_non_nan_mask.reshape(-1, feat_size)
        
        valid_val_outputs = val_outputs_reshaped[val_non_nan_mask_reshaped]
        valid_val_targets = y_val_reshaped[val_non_nan_mask_reshaped]
        
        if valid_val_targets.size(0) > 0:
            val_loss = criterion(valid_val_outputs, valid_val_targets)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, No valid validation targets')


# Plot the results
model.eval()
with torch.no_grad():
    # Get predictions and anomaly flags for the first sequence in the test set
    predictions, anomaly_flags = model(X_test[0:1])  # Shape: (1, seq_len, output_size), (1, seq_len)
    
    # Convert to numpy for plotting
    true_values = y_test[0].squeeze().numpy()  # Shape: (seq_len,)
    predicted_values = predictions[0].squeeze().numpy()  # Shape: (seq_len,)
    anomaly_flags_np = anomaly_flags[0].numpy()  # Shape: (seq_len,)
    
    # Reshape for inverse transform
    true_values_reshaped = true_values.reshape(-1, 1)
    predicted_values_reshaped = predicted_values.reshape(-1, 1)
    
    # Rescale values to original scale
    true_values_original = preprocessor.feature_scaler.inverse_transform_target(true_values_reshaped)
    predicted_values_original = preprocessor.feature_scaler.inverse_transform_target(predicted_values_reshaped)
    
    # Create time points for x-axis
    time_points = np.arange(len(true_values))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, true_values_original, 'b-', label='True Values', linewidth=2)
    plt.plot(time_points, predicted_values_original, 'r--', label='Predicted Values', linewidth=2)
    
    # Highlight anomalies
    anomaly_indices = np.where(anomaly_flags_np)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, true_values_original[anomaly_indices], color='red', s=50, label='Anomalies')
    
    plt.title('RNN Model Predictions vs True Values with Anomalies (Original Scale)')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the number of anomalies detected
    print(f"Number of anomalies detected: {len(anomaly_indices)} out of {len(true_values)} ({len(anomaly_indices)/len(true_values)*100:.2f}%)")
    
    # Print some statistics about the predictions
    print(f"True values range: {np.nanmin(true_values_original):.2f} to {np.nanmax(true_values_original):.2f}")
    print(f"Predicted values range: {np.nanmin(predicted_values_original):.2f} to {np.nanmax(predicted_values_original):.2f}")
    
    # Calculate and print MSE on original scale
    valid_mask = ~np.isnan(true_values_original)
    if np.any(valid_mask):
        mse = np.mean((true_values_original[valid_mask] - predicted_values_original[valid_mask])**2)
        print(f"MSE on original scale: {mse:.4f}")