import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import sys 
import pandas as pd
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
        
        return predictions  # shape: (batch_size, prediction_window, output_size)


# =======================
# Anomaly Detection Logic
# =======================
def compute_z_scores(pred, actual):
    """
    Compute z-scores for anomaly detection using the predicted and actual values.
    Handles NaN values and prevents NaN propagation.
    
    Args:
        pred: Predicted values, shape (batch_size, prediction_window, output_size)
        actual: Actual values, shape (batch_size, prediction_window, output_size)
        
    Returns:
        z_scores: Z-scores for each prediction
    """
    # Calculate residuals
    residual = actual - pred
    
    # Create mask for valid (non-NaN) residuals
    valid_mask = ~torch.isnan(residual)
    
    # Print diagnostic information
    # total_points = residual.numel()
    # valid_points = torch.sum(valid_mask).item()
    #print(f"Computing z-scores: {valid_points}/{total_points} valid points")
    
    if torch.sum(valid_mask) > 1:  # If we have more than one valid residual
        # Calculate mean using only valid residuals
        mean = torch.mean(residual[valid_mask])
        
        # Calculate std using only valid residuals
        std = torch.std(residual[valid_mask])
        
        # If std is too small or zero, use a small constant
        if std < 1e-6:
            std = torch.tensor(1e-6, device=residual.device)
            print(f"Standard deviation too small ({std.item()}), using minimum value")
            
        # Calculate z-scores
        z_scores = torch.zeros_like(residual)
        z_scores[valid_mask] = (residual[valid_mask] - mean) / std
        
        # Set z-scores for invalid points to 0
        z_scores[~valid_mask] = 0
        
    else:
        # If we don't have enough valid points, return zeros
        z_scores = torch.zeros_like(residual)
        print("Not enough valid points to compute z-scores")
    
    return z_scores

def replace_anomalies(pred, actual, z_scores, threshold=LSTM_CONFIG.get('z_score_threshold', 2.5)):
    """
    Replace anomalies with predicted values.
    
    Args:
        pred: Predicted values, shape (batch_size, prediction_window, output_size)
        actual: Actual values, shape (batch_size, prediction_window, output_size)
        z_scores: Z-scores for anomaly detection
        threshold: Threshold for anomaly detection
        
    Returns:
        edited: Edited values with anomalies replaced
        anomaly_flag: Flag indicating anomalies
    """
    # Check for NaN values in inputs
    if torch.isnan(pred).any() or torch.isnan(actual).any() or torch.isnan(z_scores).any():
        print("Warning: NaN values detected in inputs to replace_anomalies")
        # Replace NaN values with zeros
        pred = torch.nan_to_num(pred, nan=0.0)
        actual = torch.nan_to_num(actual, nan=0.0)
        z_scores = torch.nan_to_num(z_scores, nan=0.0)
    
    # Identify anomalies
    anomalies = torch.abs(z_scores) > threshold
    
    # Count anomalies
    anomaly_count = torch.sum(anomalies).item()
    #print(f"Detected {anomaly_count} anomalies out of {anomalies.numel()} points")
    
    # Replace anomalies with predicted values
    edited = torch.where(anomalies, pred, actual)
    
    # Create anomaly flag (1 for anomalies, 0 for normal points)
    anomaly_flag = anomalies.float()
    
    # Check for NaN values in outputs
    if torch.isnan(edited).any() or torch.isnan(anomaly_flag).any():
        print("Warning: NaN values detected in outputs of replace_anomalies")
        # Replace NaN values with zeros
        edited = torch.nan_to_num(edited, nan=0.0)
        anomaly_flag = torch.nan_to_num(anomaly_flag, nan=0.0)
    
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
    model.eval()
    X_updated = []

    with torch.no_grad():
        for i in range(X_data.shape[0]):
            # Get the current sequence
            inputs = X_data[i].unsqueeze(0)  # Add batch dimension
            
            # Get model predictions for the future time steps
            output = model(inputs)  # Shape: (1, prediction_window, num_targets)
            target = y_data[i]  # Shape: (prediction_window, num_targets)
            
            # Check for NaN values in the model output
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                print(f'NaNs detected in model output for sequence {i}, count: {nan_count}')
                # Skip this sequence and use original values
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Compute z-scores and replace anomalies for the future time steps
            z_scores = compute_z_scores(output[0], target)
            edited, anomaly_flag = replace_anomalies(output[0], target, z_scores, threshold)
            
            # Check for NaN values in edited values or anomaly flags
            if torch.isnan(edited).any() or torch.isnan(anomaly_flag).any():
                print(f'NaNs detected in edited values or anomaly flags for sequence {i}')
                # Skip this sequence and use original values
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Concatenate edited values and anomaly flags
            # Shape: (prediction_window, 2)
            new_features = torch.cat([edited, anomaly_flag], dim=1)
            
            # Create a copy of the input sequence
            new_input = inputs.clone()
            
            # Get the number of features in the input
            num_features = new_input.shape[2]
            
            # Make sure we're updating the correct features
            # The last two features should be the water level and anomaly flag
            if num_features >= 2:
                # Update the last two features for the last prediction_window time steps
                # We only update the last prediction_window time steps because that's what we predicted
                new_input[0, -new_features.shape[0]:, -2:] = new_features
            else:
                print(f'Warning: Input has fewer than 2 features ({num_features}), cannot update features')
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Final check for NaN values
            if torch.isnan(new_input).any():
                nan_count = torch.isnan(new_input).sum().item()
                print(f'NaNs detected in final updated features for sequence {i}, count: {nan_count}')
                # Replace NaNs with original values
                nan_mask = torch.isnan(new_input)
                new_input[nan_mask] = inputs[nan_mask]
            
            # Append the updated input
            X_updated.append(new_input.squeeze(0))
    
    # Stack all sequences and return
    return torch.stack(X_updated)


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
    max_grad_norm = config.get('grad_clip_value', 1.0)
    prediction_window = config.get('prediction_window', 15)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Iterative_LSTM_Model(input_size, hidden_size, output_size, prediction_window=prediction_window, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    
    # Dictionary to store validation anomalies
    val_anomalies = {
        'dates': [],
        'original': [],
        'edited': [],
        'predictions': [],
        'anomaly_points': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_samples = 0

        for i in range(X_train.shape[0]):
            inputs = X_train[i].unsqueeze(0).to(device)
            
            # Ensure target has shape (1, prediction_window, output_size)
            target = y_train[i].unsqueeze(0).to(device)
            if len(target.shape) == 2:
                target = target.unsqueeze(-1)

            output = model(inputs)

            loss = criterion(output, target)
                    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            valid_samples += 1

        avg_train_loss = train_loss / valid_samples if valid_samples > 0 else float('inf')

        # Validation loop
        model.eval()
        val_loss = 0.0
        valid_val_samples = 0
        
        # Clear previous validation results
        val_anomalies = {key: [] for key in val_anomalies}
        
        with torch.no_grad():
            for i in range(X_val.shape[0]):
                inputs = X_val[i].unsqueeze(0).to(device)
                
                # Ensure target has shape (1, prediction_window, output_size)
                target = y_val[i].unsqueeze(0).to(device)
                if len(target.shape) == 2:
                    target = target.unsqueeze(-1)
                
                output = model(inputs)
            
                # Calculate loss
                loss = criterion(output, target)
                val_loss += loss.item()
                valid_val_samples += 1
                
                # Store validation predictions and anomalies
                if epoch == 0 or (valid_val_samples > 0 and val_loss / valid_val_samples < best_val_loss):
                    # Get predictions and ensure they're on CPU
                    full_outputs = output.squeeze(0).cpu()  # Remove batch dimension
                    target_cpu = target.squeeze(0).cpu()  # Remove batch dimension

                    # Store raw predictions before anomaly detection
                    val_anomalies['predictions'].append(full_outputs.numpy())
                    
                    # Compute z-scores and replace anomalies
                    z_scores = compute_z_scores(full_outputs, target_cpu)
                    edited, anomaly_flag = replace_anomalies(full_outputs, target_cpu, z_scores, threshold)
                    
                    # Get the actual dates for this sequence
                    if val_data_index is not None:
                        # Calculate the start and end indices for this sequence
                        sequence_length = config.get('sequence_length', 500)
                        start_idx = i * sequence_length
                        end_idx = start_idx + sequence_length + prediction_window
                        sequence_dates = val_data_index[start_idx:end_idx]
                        val_anomalies['dates'].append(sequence_dates)
                    else:
                        val_anomalies['dates'].append(i)  # Fallback to sequence index if no dates provided
                    
                    # Store the data
                    val_anomalies['original'].append(target_cpu.numpy())
                    val_anomalies['edited'].append(edited.numpy())
                    
                    # Find and store anomalies with their actual dates
                    anomaly_mask = anomaly_flag.numpy() > 0
                    if np.any(anomaly_mask):
                        anomaly_indices = np.where(anomaly_mask)
                        for j, k in zip(anomaly_indices[0], anomaly_indices[1]):
                            if val_data_index is not None:
                                date_idx = start_idx + sequence_length + j
                                if date_idx < len(val_data_index):
                                    anomaly_date = val_data_index[date_idx]
                                    val_anomalies['anomaly_points'].append((anomaly_date, target_cpu[j, k]))
                            else:
                                val_anomalies['anomaly_points'].append((i, j, k, target_cpu[j, k]))

        avg_val_loss = val_loss / valid_val_samples if valid_val_samples > 0 else float('inf')
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

        # Update training inputs with edited values
        X_train = update_features(X_train, y_train, model, threshold).detach()

    print("Training complete.")
    return model, val_anomalies


# =======================
# Visualization Functions
# =======================


def plot_full_validation(preprocessor, val_data, save_path=None, val_anomalies=None):
    """
    Plot the validation results with predictions and anomalies.
    
    Args:
        preprocessor: DataPreprocessor instance
        val_data: Validation data
        save_path: Path to save the plot
        val_anomalies: Dictionary containing validation anomalies
    """
    if val_anomalies is None:
        print("No validation anomalies provided.")
        return
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get the dates from the validation data
    dates = val_data.index
    
    # Plot the original data
    ax.plot(dates, val_data[preprocessor.output_features[0]], label='Original', color='blue', alpha=0.7)
    
    # Plot the predictions and edited data
    if 'predictions' in val_anomalies and 'edited' in val_anomalies:
        # Create a dictionary to store the predictions and edited data for each date
        predictions_dict = {}
        edited_dict = {}
        
        # Process each sequence
        for i, (pred, edit, date_range) in enumerate(zip(val_anomalies['predictions'], val_anomalies['edited'], val_anomalies['dates'])):
            # Get the sequence length and prediction window from the config
            sequence_length = preprocessor.config.get('sequence_length', 500)
            prediction_window = preprocessor.config.get('prediction_window', 15)
            
            # Calculate the start and end indices for this sequence
            start_idx = i * sequence_length
            end_idx = start_idx + sequence_length + prediction_window
            
            # Get the dates for this sequence
            if isinstance(date_range, pd.DatetimeIndex):
                sequence_dates = date_range
            else:
                # If dates are not provided, use the index from val_data
                sequence_dates = dates[start_idx:end_idx]
            
            # Get the prediction dates (the last prediction_window dates)
            pred_dates = sequence_dates[-prediction_window:]
            
            # Store the predictions and edited data for each date
            for j, date in enumerate(pred_dates):
                if date not in predictions_dict:
                    predictions_dict[date] = []
                    edited_dict[date] = []
                
                predictions_dict[date].append(pred[j, 0])
                edited_dict[date].append(edit[j, 0])
        
        # Calculate the mean predictions and edited data for each date
        pred_dates = sorted(predictions_dict.keys())
        pred_values = [np.mean(predictions_dict[date]) for date in pred_dates]
        edited_values = [np.mean(edited_dict[date]) for date in pred_dates]
        
        # Plot the predictions and edited data
        ax.plot(pred_dates, pred_values, label='Predictions', color='red', alpha=0.7)
        ax.plot(pred_dates, edited_values, label='Edited', color='green', alpha=0.7)
    
    # Plot the anomalies
    if 'anomaly_points' in val_anomalies and val_anomalies['anomaly_points']:
        anomaly_dates = []
        anomaly_values = []
        
        for point in val_anomalies['anomaly_points']:
            if len(point) == 2:  # (date, value)
                date, value = point
                anomaly_dates.append(date)
                anomaly_values.append(value)
            elif len(point) == 4:  # (sequence_idx, pred_idx, target_idx, value)
                sequence_idx, pred_idx, target_idx, value = point
                # Calculate the date index
                sequence_length = preprocessor.config.get('sequence_length', 500)
                date_idx = sequence_idx * sequence_length + sequence_length + pred_idx
                if date_idx < len(dates):
                    anomaly_dates.append(dates[date_idx])
                    anomaly_values.append(value)
        
        ax.scatter(anomaly_dates, anomaly_values, color='purple', label='Anomalies', s=50, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(preprocessor.output_features[0])
    ax.set_title('Validation Results with Predictions and Anomalies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_interactive_validation(preprocessor, val_data, save_path=None, val_anomalies=None):
    """
    Create an interactive plot of the validation results with predictions and anomalies.
    
    Args:
        preprocessor: DataPreprocessor instance
        val_data: Validation data
        save_path: Path to save the plot
        val_anomalies: Dictionary containing validation anomalies
    """
    if val_anomalies is None:
        print("No validation anomalies provided.")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Get the dates from the validation data
    dates = val_data.index
    
    # Plot the original data
    fig.add_trace(go.Scatter(
        x=dates,
        y=val_data[preprocessor.output_features[0]],
        mode='lines',
        name='Original',
        line=dict(color='blue', width=1),
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Plot the predictions and edited data
    if 'predictions' in val_anomalies and 'edited' in val_anomalies:
        # Create a dictionary to store the predictions and edited data for each date
        predictions_dict = {}
        edited_dict = {}
        
        # Process each sequence
        for i, (pred, edit, date_range) in enumerate(zip(val_anomalies['predictions'], val_anomalies['edited'], val_anomalies['dates'])):
            # Get the sequence length and prediction window from the config
            sequence_length = preprocessor.config.get('sequence_length', 500)
            prediction_window = preprocessor.config.get('prediction_window', 15)
            
            # Calculate the start and end indices for this sequence
            start_idx = i * sequence_length
            end_idx = start_idx + sequence_length + prediction_window
            
            # Get the dates for this sequence
            if isinstance(date_range, pd.DatetimeIndex):
                sequence_dates = date_range
            else:
                # If dates are not provided, use the index from val_data
                sequence_dates = dates[start_idx:end_idx]
            
            # Get the prediction dates (the last prediction_window dates)
            pred_dates = sequence_dates[-prediction_window:]
            
            # Store the predictions and edited data for each date
            for j, date in enumerate(pred_dates):
                if date not in predictions_dict:
                    predictions_dict[date] = []
                    edited_dict[date] = []
                
                predictions_dict[date].append(pred[j, 0])
                edited_dict[date].append(edit[j, 0])
        
        # Calculate the mean predictions and edited data for each date
        pred_dates = sorted(predictions_dict.keys())
        pred_values = [np.mean(predictions_dict[date]) for date in pred_dates]
        edited_values = [np.mean(edited_dict[date]) for date in pred_dates]
        
        # Plot the predictions and edited data
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode='lines',
            name='Predictions',
            line=dict(color='red', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=edited_values,
            mode='lines',
            name='Edited',
            line=dict(color='green', width=1),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
    
    # Plot the anomalies
    if 'anomaly_points' in val_anomalies and val_anomalies['anomaly_points']:
        anomaly_dates = []
        anomaly_values = []
        
        for point in val_anomalies['anomaly_points']:
            if len(point) == 2:  # (date, value)
                date, value = point
                anomaly_dates.append(date)
                anomaly_values.append(value)
            elif len(point) == 4:  # (sequence_idx, pred_idx, target_idx, value)
                sequence_idx, pred_idx, target_idx, value = point
                # Calculate the date index
                sequence_length = preprocessor.config.get('sequence_length', 500)
                date_idx = sequence_idx * sequence_length + sequence_length + pred_idx
                if date_idx < len(dates):
                    anomaly_dates.append(dates[date_idx])
                    anomaly_values.append(value)
        
        fig.add_trace(go.Scatter(
            x=anomaly_dates,
            y=anomaly_values,
            mode='markers',
            name='Anomalies',
            marker=dict(color='purple', size=8),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Validation Results with Predictions and Anomalies',
        xaxis_title='Date',
        yaxis_title=preprocessor.output_features[0],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        width=1200,
        template='plotly_white',
        modebar=dict(
            orientation='vertical',
            bgcolor='rgba(255, 255, 255, 0.7)',
            color='#1f77b4',
            activecolor='#d62728'
        ),
        annotations=[
            dict(
                text="Use the zoom tools to explore the data",
                showarrow=False,
                xref="paper", yref="paper",
                x=0, y=1.05,
                font=dict(size=12, color="gray")
            )
        ],
        dragmode='zoom'
    )
    
    # Configure x-axis with enhanced zoom capabilities
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    # Configure axes with enhanced zoom capabilities
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    # Save the interactive plot as HTML if a path is provided
    if save_path:
        pio.write_html(fig, file=save_path, include_plotlyjs=True, full_html=True)
        print(f"Interactive plot saved to {save_path}")
    
    # Show the plot
    fig.show()



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
    # Plot the entire validation dataset using anomalies detected during training
    plot_full_validation(preprocessor, val_data, 
        val_anomalies=val_anomalies
    )
    
    # Create interactive Plotly visualization using anomalies detected during training
    plot_interactive_validation( preprocessor, val_data, 
        save_path="validation_plot_interactive.html",
        val_anomalies=val_anomalies
    )
