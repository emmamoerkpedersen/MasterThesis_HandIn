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
# Add the parent directory to the Python path to allow importing from _3_lstm_model
sys.path.append(str(Path(__file__).parent.parent.parent))
from _3_lstm_model.preprocessing_LSTM import DataPreprocessor
from config import LSTM_CONFIG
# =======================
# Model Definition
# =======================
class Iterative_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(Iterative_LSTM_Model, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)  # Predict next time step
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.LSTM(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        if self.training:
            lstm_out = self.dropout(lstm_out)
            
        # Apply fully connected layer to all time steps
        predictions = self.fc(lstm_out)  # shape: (batch_size, sequence_length, output_size)
        return predictions


# =======================
# Anomaly Detection Logic
# =======================
def compute_z_scores(pred, actual):
    """
    Compute z-scores for anomaly detection using the entire sequence.
    Handles NaN values and prevents NaN propagation.
    
    Args:
        pred: Predicted values, shape (batch_size, output_size)
        actual: Actual values, shape (batch_size, output_size)
        
    Returns:
        z_scores: Z-scores for each prediction
    """
    # Calculate residuals
    residual = actual - pred
    
    # Create mask for valid (non-NaN) residuals
    valid_mask = ~torch.isnan(residual)
    
    # Print diagnostic information
    total_points = residual.numel()
    valid_points = torch.sum(valid_mask).item()
    print(f"Computing z-scores: {valid_points}/{total_points} valid points")
    
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
    
    # Check for NaN values in z_scores
    if torch.isnan(z_scores).any():
        nan_count = torch.isnan(z_scores).sum().item()
        print(f"Warning: NaN values detected in z_scores: {nan_count}")
        # Replace NaN values with zeros
        z_scores = torch.nan_to_num(z_scores, nan=0.0)
    
    return z_scores

def replace_anomalies(pred, actual, z_scores, threshold=LSTM_CONFIG.get('z_score_threshold', 2.5)):
    """
    Replace anomalies with predicted values.
    
    Args:
        pred: Predicted values
        actual: Actual values
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
    print(f"Detected {anomaly_count} anomalies out of {anomalies.numel()} points")
    
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
        y_data: Target data, shape (num_sequences, sequence_length, 1)
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
            
            # Get model predictions for the entire sequence
            output = model(inputs)  # Shape: (1, sequence_length, 1)
            target = y_data[i]  # Shape: (sequence_length, 1)
            
            # Check for NaN values in the model output
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                print(f'NaNs detected in model output for sequence {i}, count: {nan_count}')
                # Skip this sequence and use original values
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Compute z-scores and replace anomalies for the entire sequence
            z_scores = compute_z_scores(output[0], target)
            edited, anomaly_flag = replace_anomalies(output[0], target, z_scores, threshold)
            
            # Check for NaN values in edited values or anomaly flags
            if torch.isnan(edited).any() or torch.isnan(anomaly_flag).any():
                print(f'NaNs detected in edited values or anomaly flags for sequence {i}')
                # Skip this sequence and use original values
                X_updated.append(inputs.squeeze(0))
                continue
            
            # Concatenate edited values and anomaly flags
            # Shape: (sequence_length, 2)
            new_features = torch.cat([edited, anomaly_flag], dim=1)
            
            # Create a copy of the input sequence
            new_input = inputs.clone()
            
            # Get the number of features in the input
            num_features = new_input.shape[2]
            
            # Make sure we're updating the correct features
            # The last two features should be the water level and anomaly flag
            if num_features >= 2:
                # Update the last two features for all time steps
                new_input[0, :, -2:] = new_features
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
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, config=None):
    """
    Train the RNN model.
    
    Args:
        X_train: Training input data, shape (num_sequences, sequence_length, num_features)
        y_train: Training target data, shape (num_sequences, sequence_length, 1)
        X_val: Validation input data, shape (num_sequences, sequence_length, num_features)
        y_val: Validation target data, shape (num_sequences, sequence_length, 1)
        input_size: Number of input features
        output_size: Number of output features (should be 1 for water level prediction)
        config: Configuration dictionary with model parameters
        
    Returns:
        model: Trained model
        val_anomalies: Dictionary containing validation anomalies detected during training
    """
    # Use default config if none provided
    if config is None:
        config = LSTM_CONFIG
    
    # Extract parameters from config
    hidden_size = config.get('hidden_size', 24)
    num_layers = config.get('num_layers', 1)
    dropout = config.get('dropout', 0.2)
    num_epochs = config.get('epochs', 10)
    patience = config.get('patience', 10)
    threshold = config.get('z_score_threshold', 2.5)
    learning_rate = config.get('learning_rate', 0.001)
    warmup_length = config.get('warmup_length', 100)
    max_grad_norm = config.get('max_grad_norm', 1.0)  # Add gradient clipping threshold
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Iterative_LSTM_Model(input_size, hidden_size, output_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    
    # Dictionary to store validation anomalies
    val_anomalies = {
        'dates': [],
        'original': [],
        'edited': [],
        'predictions': [],  # Add storage for raw predictions
        'anomaly_points': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_samples = 0

        for i in range(X_train.shape[0]):
            inputs = X_train[i].unsqueeze(0).to(device)  # Add batch dimension
            target = y_train[i].to(device)  # Shape: (sequence_length, 1)

            # Skip warmup period
            if warmup_length < target.shape[0]:
                valid_target = target[warmup_length:]
                output = model(inputs)  # Shape: (1, sequence_length, output_size)
                
                # Get predictions after warmup
                valid_outputs = output[0, warmup_length:]  # Remove batch dimension and warmup period
                
                # Calculate loss only on non-NaN targets
                non_nan_mask = ~torch.isnan(valid_target)
                if torch.any(non_nan_mask):
                    loss = criterion(valid_outputs[non_nan_mask], valid_target[non_nan_mask])
                    
            optimizer.zero_grad()
            loss.backward()
                    
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            train_loss += loss.item()
            valid_samples += 1

        # Calculate average loss only on valid samples
        avg_train_loss = train_loss / valid_samples if valid_samples > 0 else float('inf')

        # Validation loop
        model.eval()
        val_loss = 0.0
        valid_val_samples = 0
        
        # Initialize avg_val_loss to a high value
        avg_val_loss = float('inf')
        
        with torch.no_grad():
            for i in range(X_val.shape[0]):
                inputs = X_val[i].unsqueeze(0).to(device)
                target = y_val[i].to(device)  # Shape: (sequence_length, 1)
                
                # Skip warmup period
                if warmup_length < target.shape[0]:
                    valid_target = target[warmup_length:]
                    output = model(inputs)  # Shape: (1, sequence_length, output_size)
                    
                    # Get predictions after warmup
                    valid_outputs = output[0, warmup_length:]  # Remove batch dimension and warmup period
                    
                    # Calculate loss only on non-NaN targets
                    non_nan_mask = ~torch.isnan(valid_target)
                    if torch.any(non_nan_mask):
                        loss = criterion(valid_outputs[non_nan_mask], valid_target[non_nan_mask])
                        val_loss += loss.item()
                        valid_val_samples += 1
                    
                    # Store validation predictions and anomalies
                    if epoch == 0 or (valid_val_samples > 0 and val_loss / valid_val_samples < best_val_loss):
                        # Get full sequence predictions
                        full_outputs = output[0].cpu()  # Shape: (sequence_length, output_size)
                        
                        # Store raw predictions before anomaly detection
                        val_anomalies['predictions'].append(full_outputs.numpy())
                        
                        # Compute z-scores and replace anomalies for all predictions
                        z_scores = compute_z_scores(full_outputs, target.cpu())
                        edited, anomaly_flag = replace_anomalies(full_outputs, target.cpu(), z_scores, threshold)
                        
                        # Convert to numpy and store
                        original_np = target.cpu().numpy()
                        edited_np = edited.numpy()
                        anomaly_flag_np = anomaly_flag.numpy()
                        
                        # Find anomalies
                        anomaly_mask = anomaly_flag_np > 0
                        if np.any(anomaly_mask):
                            anomaly_indices = np.where(anomaly_mask)[0]
                            for idx in anomaly_indices:
                                val_anomalies['anomaly_points'].append((i, idx, original_np[idx][0]))
                        
                        # Store the data for this sequence
                        val_anomalies['dates'].append(i)
                        val_anomalies['original'].append(original_np)
                        val_anomalies['edited'].append(edited_np)

        # Calculate average validation loss
        avg_val_loss = val_loss / valid_val_samples if valid_val_samples > 0 else float('inf')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

        # Update training inputs with edited values and anomaly flags
        X_train = update_features(X_train, y_train, model, threshold).detach()

    print("Training complete.")
    return model, val_anomalies


# =======================
# Visualization Functions
# =======================


def plot_full_validation(preprocessor, val_data, save_path=None, val_anomalies=None):
    """
    Plot the entire validation dataset with dates on the x-axis.
    Shows four time series:
    1. Original data (blue)
    2. Predicted data (orange)
    3. Edited data (green) - combination of original and predicted where anomalies are detected
    4. Anomalies (red points)
    
    Args:
        preprocessor: DataPreprocessor instance for inverse transformation
        val_data: Original validation data DataFrame with datetime index
        save_path: Path to save the plot (optional)
        val_anomalies: Dictionary containing validation anomalies detected during training (optional)
    """
    if val_anomalies is None:
        print("No validation anomalies provided. Cannot create plot.")
        return
        
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Get sequence length and warmup length from config
    sequence_length = preprocessor.config.get('sequence_length', 100)
    warmup_length = preprocessor.config.get('warmup_length', 100)
    
    # Process each sequence separately to avoid connecting lines between sequences
    for i, seq_idx in enumerate(val_anomalies['dates']):
        # Get the original and edited data for this sequence
        original_data = val_anomalies['original'][i]
        edited_data = val_anomalies['edited'][i]
        
        # Get the raw predictions from the model output
        # The model output is stored in the 'edited' field but before anomaly replacement
        predicted_data = val_anomalies.get('predictions', [None] * len(val_anomalies['dates']))[i]
        if predicted_data is None:
            print(f"Warning: No predictions found for sequence {i}")
            continue
        
        # Get the dates for this sequence
        start_idx = seq_idx * sequence_length
        end_idx = min(start_idx + len(original_data), len(val_data.index))
        
        # Check if we have valid dates for this sequence
        if start_idx < len(val_data.index):
            sequence_dates = val_data.index[start_idx:end_idx]
            
            if len(sequence_dates) > 0:
                # Skip warmup period
                if warmup_length < len(sequence_dates):
                    valid_dates = sequence_dates[warmup_length:]
                    valid_original = original_data[warmup_length:len(sequence_dates)].reshape(-1)
                    valid_predicted = predicted_data[warmup_length:len(sequence_dates)].reshape(-1)
                    valid_edited = edited_data[warmup_length:len(sequence_dates)].reshape(-1)
                    
                    # Create mask for non-NaN values in original data
                    original_mask = ~np.isnan(valid_original)
                    
                    # Ensure all arrays have matching lengths
                    min_length = min(len(valid_dates), len(valid_original))
                    valid_dates = valid_dates[:min_length]
                    valid_original = valid_original[:min_length]
                    valid_predicted = valid_predicted[:min_length]
                    valid_edited = valid_edited[:min_length]
                    original_mask = original_mask[:min_length]
                    
                    # Plot original data with gaps (blue)
                    if np.any(original_mask):
                        plt.plot(valid_dates[original_mask], 
                               valid_original[original_mask], 
                               'b-', 
                               label='Original Data' if i == 0 else "", 
                               alpha=0.7, 
                               linewidth=1)
                    
                    # Plot predicted data (orange)
                    plt.plot(valid_dates, 
                            valid_predicted, 
                            'orange', 
                            label='Predicted Data' if i == 0 else "", 
                            alpha=0.5, 
                            linewidth=1)
                    
                    # Plot edited data (green)
                    plt.plot(valid_dates, 
                            valid_edited, 
                            'g-', 
                            label='Edited Data' if i == 0 else "", 
                            alpha=0.7, 
                            linewidth=1)
    
    # Plot anomalies (red points)
    if val_anomalies['anomaly_points']:
        anomaly_dates = []
        anomaly_values = []
        
        for seq_idx, pos_idx, value in val_anomalies['anomaly_points']:
            # Calculate the date for this anomaly
            start_idx = seq_idx * sequence_length
            date_idx = start_idx + pos_idx
            
            # Only include anomalies after warmup period and within valid date range
            if date_idx < len(val_data.index) and pos_idx >= warmup_length:
                # Only plot anomalies where we have original data
                if not np.isnan(value):
                    anomaly_dates.append(val_data.index[date_idx])
                    anomaly_values.append(value)
        
        if anomaly_dates:
            plt.scatter(anomaly_dates, anomaly_values, color='red', s=30, label='Anomalies', zorder=5)
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Water Level')
    plt.title('Water Level Data with Predictions and Anomalies (Validation Period)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    plt.gcf().autofmt_xdate()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_interactive_validation(preprocessor, val_data, save_path=None, val_anomalies=None):
    """
    Create an interactive Plotly visualization of the validation dataset with dates on the x-axis.
    Shows four time series:
    1. Original data (blue)
    2. Predicted data (orange)
    3. Edited data (green) - combination of original and predicted where anomalies are detected
    4. Anomalies (red points)
    
    Args:
        preprocessor: DataPreprocessor instance for inverse transformation
        val_data: Original validation data DataFrame with datetime index
        save_path: Path to save the HTML file (optional)
        val_anomalies: Dictionary containing validation anomalies detected during training (optional)
    """
    if val_anomalies is None:
        print("No validation anomalies provided. Cannot create plot.")
        return
        
    # Create interactive Plotly figure
    fig = go.Figure()
    
    # Get sequence length and warmup length from config
    sequence_length = preprocessor.config.get('sequence_length', 100)
    warmup_length = preprocessor.config.get('warmup_length', 100)
    
    # Process each sequence separately to avoid connecting lines between sequences
    for i, seq_idx in enumerate(val_anomalies['dates']):
        # Get the original and edited data for this sequence
        original_data = val_anomalies['original'][i]
        edited_data = val_anomalies['edited'][i]
        predicted_data = val_anomalies['edited'][i]  # Model's raw predictions
        
        # Get the dates for this sequence
        start_idx = seq_idx * sequence_length
        end_idx = min(start_idx + len(original_data), len(val_data.index))
        
        # Check if we have valid dates for this sequence
        if start_idx < len(val_data.index):
            sequence_dates = val_data.index[start_idx:end_idx]
            
            if len(sequence_dates) > 0:
                # Skip warmup period
                if warmup_length < len(sequence_dates):
                    valid_dates = sequence_dates[warmup_length:]
                    valid_original = original_data[warmup_length:len(sequence_dates)].reshape(-1)
                    valid_predicted = predicted_data[warmup_length:len(sequence_dates)].reshape(-1)
                    valid_edited = edited_data[warmup_length:len(sequence_dates)].reshape(-1)
                    
                    # Create mask for non-NaN values in original data
                    original_mask = ~np.isnan(valid_original)
                    
                    # Ensure all arrays have matching lengths
                    min_length = min(len(valid_dates), len(valid_original))
                    valid_dates = valid_dates[:min_length]
                    valid_original = valid_original[:min_length]
                    valid_predicted = valid_predicted[:min_length]
                    valid_edited = valid_edited[:min_length]
                    original_mask = original_mask[:min_length]
                    
                    # Plot original data with gaps (blue)
                    if np.any(original_mask):
                        fig.add_trace(go.Scatter(
                            x=valid_dates[original_mask],
                            y=valid_original[original_mask],
                            mode='lines',
                            name='Original Data' if i == 0 else "Original Data_" + str(i),
                            line=dict(color='blue', width=1),
                            hovertemplate='Date: %{x}<br>Water Level: %{y:.2f}<extra></extra>',
                            showlegend=i == 0
                        ))
                    
                    # Plot predicted data (orange)
                    fig.add_trace(go.Scatter(
                        x=valid_dates,
                        y=valid_predicted,
                        mode='lines',
                        name='Predicted Data' if i == 0 else "Predicted Data_" + str(i),
                        line=dict(color='orange', width=1),
                        hovertemplate='Date: %{x}<br>Water Level: %{y:.2f}<extra></extra>',
                        showlegend=i == 0
                    ))
                    
                    # Plot edited data (green)
                    fig.add_trace(go.Scatter(
                        x=valid_dates,
                        y=valid_edited,
                        mode='lines',
                        name='Edited Data' if i == 0 else "Edited Data_" + str(i),
                        line=dict(color='green', width=1),
                        hovertemplate='Date: %{x}<br>Water Level: %{y:.2f}<extra></extra>',
                        showlegend=i == 0
                    ))
    
    # Plot anomalies (red points)
    if val_anomalies['anomaly_points']:
        anomaly_dates = []
        anomaly_values = []
        
        for seq_idx, pos_idx, value in val_anomalies['anomaly_points']:
            # Calculate the date for this anomaly
            start_idx = seq_idx * sequence_length
            date_idx = start_idx + pos_idx
            
            # Only include anomalies after warmup period and within valid date range
            if date_idx < len(val_data.index) and pos_idx >= warmup_length:
                # Only plot anomalies where we have original data
                if not np.isnan(value):
                    anomaly_dates.append(val_data.index[date_idx])
                    anomaly_values.append(value)
        
        if anomaly_dates:
            fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8),
                hovertemplate='Date: %{x}<br>Water Level: %{y:.2f}<extra></extra>'
            ))
    
    # Update layout with enhanced zoom capabilities
    fig.update_layout(
        title='Water Level Data with Predictions and Anomalies (Validation Period)',
        xaxis_title='Date',
        yaxis_title='Water Level',
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
            orientation='v',
            bgcolor='rgba(255, 255, 255, 0.7)',
            color='#1f77b4',
            activecolor='#d62728'
        ),
        annotations=[
            dict(
                text="Use the zoom tools to explore both time and water level ranges",
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
        ),
        showspikes=True,
        spikemode='across',
        spikesnap='cursor'
    )
    
    # Configure y-axis with enhanced zoom capabilities
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

    # Print data shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Get input and output sizes from the data
    input_size = X_train.shape[2]  # Number of features
    output_size = 1  # We're predicting a single value (water level)

    # Train the model with configurations from LSTM_CONFIG
    trained_model, val_anomalies = train_model(
        X_train, y_train, X_val, y_val, 
        input_size, output_size,
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
