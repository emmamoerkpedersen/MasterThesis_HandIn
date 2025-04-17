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

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, input_size = x.size()
        device = x.device

        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)

        preds = []

        for t in range(seq_len):
            if t == 0:
                input_t = x[:, t, :]  # use first input as-is
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

            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))
            pred = self.linear(h_t)  # Shape: (batch_size, output_size)
            preds.append(pred.unsqueeze(1))  # For seq build-up

        return torch.cat(preds, dim=1)  # Final shape: (batch_size, seq_len, output_size)

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
    'use_time_features': True,
    'use_cumulative_features': True
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


# Get input and output sizes from the data
input_size = X_train.shape[2]  # Number of features
output_size = y_train.shape[2]  # Number of output features
hidden_size = 24  # Increased hidden size for more complex data

# Initialize the model
model = RNNModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    
    # Create mask for non-NaN values
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
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the results
model.eval()
with torch.no_grad():
    # Get predictions for the first sequence in the test set
    predictions = model(X_test[0:1])  # Shape: (1, seq_len, output_size)
    
    # Convert to numpy for plotting
    true_values = y_test[0].squeeze().numpy()  # Shape: (seq_len,)
    predicted_values = predictions[0].squeeze().numpy()  # Shape: (seq_len,)
    
    # Create time points for x-axis
    time_points = np.arange(len(true_values))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, true_values, 'b-', label='True Values', linewidth=2)
    plt.plot(time_points, predicted_values, 'r--', label='Predicted Values', linewidth=2)
    plt.title('RNN Model Predictions vs True Values')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.legend()
    plt.grid(True)
    plt.show()