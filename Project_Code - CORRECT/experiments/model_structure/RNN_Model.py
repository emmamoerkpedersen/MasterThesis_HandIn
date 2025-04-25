import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import sys 
import pandas as pd
from datetime import datetime
# Add the parent directory to the Python path to allow importing from _3_lstm_model
sys.path.append(str(Path(__file__).parent.parent.parent))
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from config import LSTM_CONFIG
# =======================
# Model Definition
# =====================
class Iterative_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, prediction_window=15, num_layers=1, dropout=0.0):
        super(Iterative_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_window = prediction_window
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * prediction_window)  # Predict multiple future time steps
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        
        # LSTM forward pass
        lstm_out, _ = self.LSTM(x, (h0, c0))  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        if self.training:
            lstm_out = self.dropout(lstm_out)
            
        # Apply fully connected layer to the last time step only
        last_hidden = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        
        predictions = self.fc(last_hidden)  # shape: (batch_size, output_size * prediction_window)        
        # Reshape to (batch_size, prediction_window, output_size)
        predictions = predictions.view(predictions.size(0), self.prediction_window, -1)
        
        return predictions


# =======================
# Anomaly Detection Logic
# =======================
def compute_z_scores(pred, actual):
    """
    Compute z-scores for anomaly detection using the predicted and actual values.
    Handles NaN values and prevents NaN propagation.
    
    Args:
        pred: Predicted values, shape (prediction_window, output_size)
        actual: Actual values, shape (prediction_window, output_size)
        
    Returns:
        z_scores: Z-scores for each prediction
    """

    
    # Calculate residuals
    residual = actual - pred

    
    # Create mask for valid (non-NaN) residuals
    valid_mask = ~torch.isnan(residual)
    
    if torch.sum(valid_mask) > 1:  # If we have more than one valid residual
        # Calculate mean using only valid residuals
        mean = torch.mean(residual[valid_mask])
        
        # Calculate std using only valid residuals
        std = torch.std(residual[valid_mask])
        
        # If std is too small or zero, use a small constant
        if std < 1e-6:
            std = torch.tensor(1e-6, device=residual.device)
            print("Standard deviation too small, using minimum value")
            
        # Calculate z-scores
        z_scores = torch.zeros_like(residual)
        z_scores[valid_mask] = (residual[valid_mask] - mean) / std
        
        # Set z-scores for invalid points to 0
        z_scores[~valid_mask] = 0
        
    else:
        # If we don't have enough valid points, return zeros
        print("Not enough valid points, returning zeros")
        z_scores = torch.zeros_like(residual)
    
    print(f"Z-scores shape: {z_scores.shape}")
    return z_scores

def replace_anomalies(pred, actual, z_scores, threshold=2.5):
    """
    Replace anomalies with predicted values.
    
    Args:
        pred: Predicted values, shape (prediction_window, output_size)
        actual: Actual values, shape (prediction_window, output_size)
        z_scores: Z-scores for anomaly detection
        threshold: Threshold for anomaly detection
        
    Returns:
        edited: Edited values with anomalies replaced
        anomaly_flag: Flag indicating anomalies
    """

    
    # Check for NaN values in inputs
    if torch.isnan(pred).any() or torch.isnan(actual).any() or torch.isnan(z_scores).any():
        print("NaN values detected in inputs, replacing with zeros")
        # Replace NaN values with zeros
        pred = torch.nan_to_num(pred, nan=0.0)
        actual = torch.nan_to_num(actual, nan=0.0)
        z_scores = torch.nan_to_num(z_scores, nan=0.0)
    
    # Identify anomalies
    anomalies = torch.abs(z_scores) > threshold
    num_anomalies = torch.sum(anomalies).item()
    print(f"Number of anomalies detected: {num_anomalies}")
    
    # Replace anomalies with predicted values
    edited = torch.where(anomalies, pred, actual)
    
    # Create anomaly flag (1 for anomalies, 0 for normal points)
    anomaly_flag = anomalies.float()
    
    # Check for NaN values in outputs
    if torch.isnan(edited).any() or torch.isnan(anomaly_flag).any():
        print("NaN values detected in outputs, replacing with zeros")
        # Replace NaN values with zeros
        edited = torch.nan_to_num(edited, nan=0.0)
        anomaly_flag = torch.nan_to_num(anomaly_flag, nan=0.0)
    
    print(f"Edited shape: {edited.shape}")
    print(f"Anomaly flag shape: {anomaly_flag.shape}")
    return edited, anomaly_flag


# =======================
# Update Input Features
# =======================
def update_features(X_data, y_data, model, threshold=LSTM_CONFIG.get('z_score_threshold', 2.5)):
    """
    Update input features with predicted values and anomaly flags.
    
    Args:
        X_data: Input data, shape (num_sequences, sequence_length, num_features)
        y_data: Target data, shape (num_sequences, prediction_window, num_targets)
        model: Trained model
        threshold: Threshold for anomaly detection
        
    Returns:
        X_updated: Updated input data with edited water level and anomaly flag
    """
    print(f"\nUpdating Features:")
    print(f"X_data shape: {X_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    
    model.eval()
    X_updated = []

    with torch.no_grad():
        for i in range(X_data.shape[0]):
            # Get the current sequence
            inputs = X_data[i].unsqueeze(0)  # Add batch dimension
            print(f"\nProcessing sequence {i+1}/{X_data.shape[0]}")
            print(f"Input shape: {inputs.shape}")
            
            # Get model predictions for the future time steps
            output = model(inputs)  # Shape: (1, prediction_window, num_targets)
            target = y_data[i]  # Shape: (prediction_window, num_targets)
            print(f"Model output shape: {output.shape}")
            print(f"Target shape: {target.shape}")
            
            # Check for NaN values in the model output
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                print(f'NaNs detected in model output, count: {nan_count}')
                # Skip this sequence and use original values
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Compute z-scores and replace anomalies for the future time steps
            z_scores = compute_z_scores(output[0], target)
            edited, anomaly_flag = replace_anomalies(output[0], target, z_scores, threshold)
            
            # Check for NaN values in edited values or anomaly flags
            if torch.isnan(edited).any() or torch.isnan(anomaly_flag).any():
                print(f'NaNs detected in edited values or anomaly flags')
                # Skip this sequence and use original values
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Concatenate edited values and anomaly flags
            # Shape: (prediction_window, 2)
            new_features = torch.cat([edited, anomaly_flag], dim=1)
            print(f"New features shape: {new_features.shape}")
            
            # Create a copy of the input sequence
            new_input = inputs.clone()
            
            # Get the number of features in the input
            num_features = new_input.shape[2]
            print(f"Number of input features: {num_features}")
            
            # Make sure we're updating the correct features
            # The last two features should be the water level and anomaly flag
            if num_features >= 2:
                # Update the last two features for the last prediction_window time steps
                # We only update the last prediction_window time steps because that's what we predicted
                new_input[0, -new_features.shape[0]:, -2:] = new_features
                print(f"Updated input shape: {new_input.shape}")
            else:
                print(f'Warning: Input has fewer than 2 features ({num_features}), cannot update features')
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Final check for NaN values
            if torch.isnan(new_input).any():
                nan_count = torch.isnan(new_input).sum().item()
                print(f'NaNs detected in final updated features, count: {nan_count}')
                # Replace NaNs with original values
                nan_mask = torch.isnan(new_input)
                new_input[nan_mask] = inputs[nan_mask]
            
            # Append the updated input
            X_updated.append(new_input.squeeze(0))
    
    # Stack all sequences and return
    X_updated = torch.stack(X_updated)
    print(f"\nFinal updated features shape: {X_updated.shape}")
    return X_updated


# =======================
# Training Loop
# =======================
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, val_data_index=None, config=None):
    """
    Train the RNN model for forecasting.
    
    Args:
        X_train: Training input data, shape (num_sequences, sequence_length, num_features)
        y_train: Training target data, shape (num_sequences, prediction_window, num_targets)
        X_val: Validation input data, shape (num_sequences, sequence_length, num_features)
        y_val: Validation target data, shape (num_sequences, prediction_window, num_targets)
        input_size: Number of input features
        output_size: Number of output features (should be 1 for water level prediction)
        val_data_index: Index of the validation data containing actual dates
        config: Configuration dictionary with model parameters
        
    Returns:
        model: Trained model
        val_anomalies: Dictionary containing validation anomalies detected during training
    """
    # Use default config if none provided
    if config is None:
        config = LSTM_CONFIG
    
    # Extract parameters from config
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.25)
    num_epochs = config.get('epochs', 100)
    patience = config.get('patience', 6)
    threshold = config.get('z_score_threshold', 1.5)
    learning_rate = config.get('learning_rate', 0.0001)
    warmup_length = config.get('warmup_length', 0)
    prediction_window = config.get('prediction_window', 15)
    
    print("\nModel Configuration:")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Dropout: {dropout}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Z-score threshold: {threshold}")
    print(f"Learning rate: {learning_rate}")
    print(f"Prediction window: {prediction_window}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Iterative_LSTM_Model(input_size, hidden_size, output_size, prediction_window=prediction_window, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize arrays to store the full validation sequences
    sequence_length = config.get('sequence_length', 500)
    
    if val_data_index is not None:
        total_length = len(val_data_index)
    else:
        # If no dates provided, calculate total length based on validation data shape
        total_length = len(X_val) * prediction_window

    best_val_anomalies = {
        'dates': np.array([None] * total_length, dtype=object),
        'original': np.zeros(total_length),
        'edited': np.zeros(total_length),
        'predictions': np.zeros(total_length),
        'anomaly_points': []
    }

    print("\n=== Starting Training Loop ===")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_samples = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training phase:")
        
        for i in range(X_train.shape[0]):
            inputs = X_train[i].unsqueeze(0).to(device)
            target = y_train[i].unsqueeze(0).to(device)
            
            # Ensure target has the same sequence length as the model's output
            if target.shape[1] != prediction_window:
                target = target[:, :prediction_window, :]
            
            if len(target.shape) == 2:
                target = target.unsqueeze(-1)

            output = model(inputs)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            valid_samples += 1
 

        avg_train_loss = train_loss / valid_samples if valid_samples > 0 else float('inf')
        print(f"Average training loss: {avg_train_loss:.6f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        valid_val_samples = 0
        
        print("\nValidation phase:")
        
        # Define batch size for validation
        batch_size = 32  # You can adjust this based on your memory constraints
        
        with torch.no_grad():
            # Process validation data in batches
            for i in range(0, X_val.shape[0], batch_size):
                # Get batch of sequences
                batch_end = min(i + batch_size, X_val.shape[0])
                inputs = X_val[i:batch_end].to(device)
                targets = y_val[i:batch_end].to(device)
                
                # Ensure targets have the same sequence length as the model's output
                if targets.shape[1] != prediction_window:
                    targets = targets[:, :prediction_window, :]
                
                if len(targets.shape) == 2:
                    targets = targets.unsqueeze(-1)
                
                output = model(inputs)
                loss = criterion(output, targets)
                val_loss += loss.item() * (batch_end - i)  # Scale loss by batch size
                valid_val_samples += (batch_end - i)
                
                # Only store validation predictions and anomalies if this is the best model so far
                if valid_val_samples > 0 and val_loss / valid_val_samples < best_val_loss:
                    # Get predictions and ensure they're on CPU
                    full_outputs = output.cpu()  # Shape: (batch_size, prediction_window, num_targets)
                    target_cpu = targets.cpu()
                    
                    # Compute z-scores and detect anomalies for the batch
                    z_scores = compute_z_scores(full_outputs, target_cpu)
                    edited, anomaly_flag = replace_anomalies(full_outputs, target_cpu, z_scores, threshold)
                    
                    # Process each sequence in the batch
                    for j in range(batch_end - i):
                        # Calculate the absolute position in the validation set
                        absolute_seq_idx = i + j
                        
                        # Calculate indices for storing data
                        start_idx = absolute_seq_idx * prediction_window
                        end_idx = min(start_idx + prediction_window, total_length)
                        
                        # Skip if we would exceed the array bounds
                        if start_idx >= total_length:
                            continue
                            
                        # Calculate how many elements we can actually store
                        n_elements = end_idx - start_idx
                        
                        if val_data_index is not None:
                            # Get the actual dates for this sequence
                            sequence_dates = val_data_index[start_idx:end_idx]
                            
                            # Store the data in the correct position in the arrays
                            best_val_anomalies['dates'][start_idx:end_idx] = sequence_dates
                            best_val_anomalies['original'][start_idx:end_idx] = target_cpu[j, :n_elements, 0].numpy()
                            best_val_anomalies['edited'][start_idx:end_idx] = edited[j, :n_elements, 0].numpy()
                            best_val_anomalies['predictions'][start_idx:end_idx] = full_outputs[j, :n_elements, 0].numpy()
                            
                            # Find and store anomalies with their actual dates
                            anomaly_mask = anomaly_flag[j, :n_elements].numpy() > 0
                            if np.any(anomaly_mask):
                                anomaly_indices = np.where(anomaly_mask)[0]
                                for k in anomaly_indices:
                                    date_idx = start_idx + k
                                    if date_idx < len(val_data_index):
                                        anomaly_date = val_data_index[date_idx]
                                        anomaly_value = target_cpu[j, k, 0].item()
                                        best_val_anomalies['anomaly_points'].append((anomaly_date, anomaly_value))
                        else:
                            # If no dates provided, use sequence indices
                            best_val_anomalies['dates'][start_idx:end_idx] = np.arange(start_idx, end_idx)
                            best_val_anomalies['original'][start_idx:end_idx] = target_cpu[j, :n_elements, 0].numpy()
                            best_val_anomalies['edited'][start_idx:end_idx] = edited[j, :n_elements, 0].numpy()
                            best_val_anomalies['predictions'][start_idx:end_idx] = full_outputs[j, :n_elements, 0].numpy()
                            
                            # Store anomalies with sequence indices
                            anomaly_mask = anomaly_flag[j, :n_elements].numpy() > 0
                            if np.any(anomaly_mask):
                                anomaly_indices = np.where(anomaly_mask)[0]
                                for k in anomaly_indices:
                                    idx = start_idx + k
                                    anomaly_value = target_cpu[j, k, 0].item()
                                    best_val_anomalies['anomaly_points'].append((idx, anomaly_value))
                
            
        avg_val_loss = val_loss / valid_val_samples if valid_val_samples > 0 else float('inf')
        print(f"\nEpoch Summary:")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Validation Loss: {avg_val_loss:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"New best validation loss: {best_val_loss:.6f}")
            print("Saving model...")
            torch.save(model.state_dict(), "best_model.pt")
            # Store the current val_anomalies as the best ones
            val_anomalies = {
                'dates': best_val_anomalies['dates'].copy(),
                'original': best_val_anomalies['original'].copy(),
                'edited': best_val_anomalies['edited'].copy(),
                'predictions': best_val_anomalies['predictions'].copy(),
                'anomaly_points': best_val_anomalies['anomaly_points'].copy()
            }
            print(f"Number of anomaly points detected: {len(val_anomalies['anomaly_points'])}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

        # Update training inputs with edited values
        print("\nUpdating training inputs with edited values...")
        X_train = update_features(X_train, y_train, model, threshold).detach()

    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final number of anomaly points: {len(val_anomalies['anomaly_points'])}")
    return model, val_anomalies


# =======================
# Visualization Functions
# ======================= 
def plot_validation_anomalies(val_anomalies):
    """
    Create an interactive plot of validation anomalies data using Plotly and save it as PNG using matplotlib.
    
    Args:
        val_anomalies: Dictionary containing validation anomalies data with keys:
            - dates: List of dates/timestamps
            - original: List of original values
            - edited: List of edited values
            - predictions: List of model predictions
            - anomaly_points: List of detected anomalies
    """
    print("\n=== Creating Validation Anomalies Plot ===")
    
    # Create matplotlib figure
    plt.figure(figsize=(12, 6))
    
    # Convert dates to strings for plotting
    dates = pd.to_datetime(val_anomalies['dates'])

    # Ensure all time series are 1D numpy arrays
    original = np.array(val_anomalies['original']).flatten()
    edited = np.array(val_anomalies['edited']).flatten()
    predictions = np.array(val_anomalies['predictions']).flatten()

    # Sanity check
    assert len(dates) == len(original) == len(edited) == len(predictions)

    print("\nAdding traces to plot...")
    # Plot original values
    plt.plot(dates, original, 'b-', label='Original Values', alpha=0.7, linewidth=1)
    
    # Plot edited values
    plt.plot(dates, edited, 'g-', label='Edited Values', alpha=0.7, linewidth=1)
    
    # Plot predictions
    plt.plot(dates, predictions, 'orange', label='Predictions', alpha=0.7, linewidth=1)
    
    # Plot anomaly points
    if val_anomalies['anomaly_points']:
        print("Adding anomaly points to plot...")
        anomaly_dates = pd.to_datetime([point[0] for point in val_anomalies['anomaly_points']])
        anomaly_values = [point[1] for point in val_anomalies['anomaly_points']]
        plt.scatter(anomaly_dates, anomaly_values, color='red', s=100, marker='d', label='Anomaly Points')
    
    print("\nUpdating plot layout...")
    # Update layout
    plt.title('Validation Anomalies Analysis')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    print("Saving plot...")
    # Save the plot
    plt.savefig('validation_anomalies.png', dpi=300, bbox_inches='tight')
    
    print("Displaying plot...")
    # Show the plot
    plt.show()
    
    print("Plot creation and saving complete!")
    return plt.gcf()


# =======================
# Example Usage
# =======================
if __name__ == "__main__":
    # Load your preprocessed data
    # Initialize the data preprocessor
    config = LSTM_CONFIG
    project_root = Path(__file__).parent.parent.parent
    preprocessor = DataPreprocessor(config)

    # Load and split the data
    station_id = '21006846'  # Main station ID
    train_data, val_data, test_data = preprocessor.load_and_split_data(project_root, station_id)

    # Prepare the data for training, with shape (num_sequences, sequence_length, num_features)
    X_train, y_train = preprocessor.prepare_data(train_data, is_training=True)
    X_val, y_val = preprocessor.prepare_data(val_data, is_training=False)
    X_test, y_test = preprocessor.prepare_data(test_data, is_training=False)

    # Get input and output sizes from the data
    input_size = X_train.shape[2]  # Number of features
    output_size = 1  # We're predicting a single value (water level)

    # Train the model with configurations from LSTM_CONFIG
    trained_model, val_anomalies = train_model(
        X_train, y_train, X_val, y_val, 
        input_size, output_size,
        val_data_index=val_data.index,
        config=LSTM_CONFIG
    )
  
    plot_validation_anomalies(val_anomalies)