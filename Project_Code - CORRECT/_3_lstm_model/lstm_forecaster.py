"""
LSTM-based Forecasting Model for Anomaly Detection in Time Series Data.

This module implements LSTM-based forecasting models for time series anomaly detection.
Two primary model architectures are provided:

1. LSTMForecaster: A unidirectional LSTM that predicts future values based only on past data.
   - Takes a sequence of input_length timesteps
   - Predicts output_length future timesteps
   - Purely causal (no future information leakage)

2. BidirectionalForecaster: A more advanced architecture combining:
   - A bidirectional LSTM encoder to capture patterns from the input sequence
   - A unidirectional decoder that forecasts future values
   - Leverages pattern recognition while maintaining proper forecasting

Key Components:
- ForecasterWrapper: Manages data preprocessing, training, evaluation, and prediction
- train_forecaster: High-level function to create and train models
- evaluate_forecaster: Evaluates models on test data for anomaly detection

Anomaly Detection Approach:
1. Train model on normal (clean) data to learn typical patterns
2. Apply model to test data (potentially containing anomalies)
3. Compare model forecasts with actual values
4. Flag significant deviations as potential anomalies
5. Calculate performance metrics using known ground truth

Data Handling:
- Uses sliding windows to create input-output pairs
- Normalizes data using Min-Max scaling
- Handles multiple features if available
- Supports multi-station training

The forecasting approach is particularly effective for water level data, as it:
1. Preserves temporal causality (predictions only use past data)
2. Can detect both sudden spikes and gradual drift
3. Adapts to normal seasonal or daily patterns

TODO:
   # Add rate-of-change detection after calculating errors
   # Flag sudden changes that exceed physical limits
   rate_of_change = np.abs(np.diff(aligned_actual, prepend=aligned_actual[0]))
   physical_anomalies = (rate_of_change > 50).astype(int)  # 50mm per interval is suspicious
   anomaly_flags = np.logical_or(anomaly_flags, physical_anomalies).astype(int)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from tqdm.auto import tqdm  # Import tqdm for progress bars

class LSTMModel(nn.Module):
    """
    NN model for time series forecasting.
    """
    def __init__(self, input_size, sequence_length, hidden_size, output_size, num_layers, dropout):  
        super(LSTMModel, self).__init__()
        self.model_name = 'LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0
                            )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor of shape (batch_size, input_size)
        """
        # LSTM forward pass
        out, hn = self.lstm(x)  # out: [batch_size, seq_len, hidden_size], hn: [num_layers, batch_size, hidden_size]
        # Only take the output of the last time step
        last_out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        # Apply dropout
        last_out = self.dropout(last_out)
        # Pass through fully connected layer
        prediction = self.fc(last_out)  # Shape: (batch_size, input_size)

        return prediction, hn[-1] # (the prediction, the final hidden state)

class train_LSTM:
    def __init__(self, model, config):
        """
        Initialize the trainer with model and configuration.
        
        Args:
            model: The LSTMModel instance
            config: Dictionary containing training parameters
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        self.criterion = nn.MSELoss()
        
        # Initialize data scaler
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation.
        """
        # Get station data (we only have one station)
        station_id = list(data.keys())[0]
        features = pd.concat([data[station_id][col] for col in self.config['feature_cols']], axis=1)
        
        # Scale the data
        if is_training and not self.is_fitted:
            self.scaler.fit(features)
            self.is_fitted = True
        
        scaled_data = self.scaler.transform(features)
        
        # Create sequences and convert to tensors
        X, y = self._create_sequences(scaled_data)
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def _create_sequences(self, data):
        """
        Create input/output sequences for training.
        
        Args:
            data: Scaled numpy array of shape [samples, features]
            
        Returns:
        tuple: (X, y) where X is input sequences    and y is target values
        """
        X, y = [], []
        sequence_length = self.config.get('sequence_length', 1000)
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 0])
            
        return np.array(X), np.array(y).reshape(-1, 1)

    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader containing training data
            
        Returns:
            float: Average loss for this epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc="Training", leave=False):
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader containing validation data
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validating", leave=False):
                outputs, _ = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train(self, train_data, val_data, epochs=100, batch_size=1, patience=10):
        """
        Train the model with early stopping.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            dict: Training history
        """
        # Print hyperparameters
        print("\nModel Hyperparameters:")
        print(f"Input Size: {self.model.input_size}")
        print(f"Hidden Size: {self.model.hidden_size}")
        print(f"Number of Layers: {self.model.num_layers}")
        print(f"Dropout Rate: {self.config.get('dropout', 0.2)}")
        
        print("\nTraining Parameters:")
        print(f"Learning Rate: {self.config.get('learning_rate', 0.001)}")
        print(f"Batch Size: {batch_size}")
        print(f"Max Epochs: {epochs}")
        print(f"Early Stopping Patience: {patience}")
        print(f"Sequence Length: {self.config.get('sequence_length', 1000)}")
        print(f"Features Used: {self.config.get('feature_cols', ['vst_raw'])}")
        print(f"Device: {self.device}")
        print("\nStarting training...\n")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, is_training=True)
        X_val, y_val = self.prepare_data(val_data, is_training=False)
        
        # Print data shapes
        print(f"Training Data Shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Validation Data Shapes - X: {X_val.shape}, y: {y_val.shape}\n")
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )
        
        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return history

    def predict(self, data):
        """
        Make predictions on new data.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            numpy.ndarray: Predictions in original scale
        """
        self.model.eval()
        X, _ = self.prepare_data(data, is_training=False)
        
        with torch.no_grad():
            predictions, _ = self.model(X)
            
        # Convert predictions back to original scale
        predictions = predictions.cpu().numpy()
        predictions_reshaped = np.zeros((predictions.shape[0], self.scaler.n_features_in_))
        predictions_reshaped[:, 0] = predictions.flatten()  # Assuming prediction is first feature
        
        return self.scaler.inverse_transform(predictions_reshaped)[:, 0]
    
########################################################
import sys
project_root = Path(__file__).parent.parent  # Go up one level from current file
sys.path.append(str(project_root))

import pandas as pd
from pathlib import Path
from _1_preprocessing.split import split_data_rolling

from _2_synthetic.synthetic_errors import SyntheticErrorGenerator
from config import LSTM_CONFIG, SYNTHETIC_ERROR_PARAMS
lstm_config = LSTM_CONFIG.copy()


#1
station_id = ['21006846']

print(f"Loading and preprocessing station data for station {station_id}...")
preprocessed_data = pd.read_pickle('../data_utils/Sample data/preprocessed_data.pkl')

# Generate dictionary with same structure but with the specified station_id
preprocessed_data = {station_id: preprocessed_data[station_id] for station_id in station_id}


#2
print("\nSplitting data into rolling windows, 3 years training, 1 year validation, 2 years test...")
split_datasets = split_data_rolling(preprocessed_data)

#3
print("\nStep 3: Generating synthetic errors for test data only...")
# Dictionary to store results for each station/year
stations_results = {}
# Create synthetic error generator
error_generator = SyntheticErrorGenerator(SYNTHETIC_ERROR_PARAMS)

# Process only test data
if 'test' in split_datasets:
    print("\nProcessing test data...")
    for station, station_data in split_datasets['test'].items():
        try:
            print(f"Generating synthetic errors for {station} (Test)...")
            test_data = station_data['vst_raw']
            
            if test_data is None or test_data.empty:
                print(f"No test data available for station {station}")
                continue
            
            # Generate synthetic errors
            modified_data, ground_truth = error_generator.inject_all_errors(test_data)
            
            # Store results
            station_key = f"{station}_test"
            stations_results[station_key] = {
                'modified_data': modified_data,
                'ground_truth': ground_truth,
                'error_periods': error_generator.error_periods
            }
            
        except Exception as e:
            print(f"Error processing station {station}: {str(e)}")
            continue

print("\nStep 4: Training LSTM models with Station-Specific Approach...")

# Prepare train and validation data
print("\nPreparing training and validation data...")

# Get all available windows
num_windows = len(split_datasets['windows'])
print(f"Total number of windows: {num_windows}")

# Use all windows - each window has its own train/val split
print(f"Using all {num_windows} windows for training/validation")
print(f"Each window contains:")
print(f"- 3 years of training data")
print(f"- 1 year of validation data")
print(f"(Test data is stored separately in split_datasets['test'])")

# Use the LSTM configuration from config.py

print(f"Input feature size: {len(lstm_config.get('feature_cols'))}")

# Initialize model and trainer
model = LSTMModel(
    input_size=len(lstm_config['feature_cols']),
    sequence_length=lstm_config.get('sequence_length', 72),
    hidden_size=lstm_config.get('hidden_size', 64),
    output_size=1,
    num_layers=lstm_config.get('num_layers', 2),
    dropout=lstm_config.get('dropout', 0.2)
)

trainer = train_LSTM(model, lstm_config)

# Train on each window
for window_idx, window_data in split_datasets['windows'].items():
    print(f"\nProcessing window {window_idx}")
    
    # Get training and validation data for this window
    train_data = window_data['train']
    val_data = window_data['validation']
    
    # Train the model
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=lstm_config.get('epochs', 100),
        batch_size=lstm_config.get('batch_size', 32),
        patience=lstm_config.get('patience', 10)
    )
    # Optionally save the model after each window
    torch.save(model.state_dict(), f'model_window_{window_idx}.pth')

# After training, you can use the model for predictions
test_predictions = trainer.predict(split_datasets['test'])



# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.offline as pyo
# import pandas as pd
# import numpy as np
# import plotly.offline as pyo
# import webbrowser
# import os

# def create_interactive_combined_plot(all_results, station_id='21006845'):
#     """
#     Generate an interactive plot showing original data, predictions, and anomalies across all years.
    
#     Args:
#         all_results: Dictionary containing results from evaluate_forecaster
#         station_id: Base station ID without year suffix
    
#     Returns:
#         Plotly figure object that can be displayed in a browser
#     """
#     # Collect all data across years
#     all_original_timestamps = []
#     all_original_values = []
#     all_prediction_timestamps = []
#     all_prediction_values = []
#     all_anomaly_timestamps = []
#     all_anomaly_values = []
#     all_error_timestamps = []
#     all_error_values = []
    
#     # Track years found for the title
#     years_found = []
    
#     # Process each station-year key
#     for station_key in sorted(all_results.keys()):
#         # Check if this key belongs to our station
#         if not station_key.startswith(f"{station_id}_"):
#             continue
            
#         # Extract year from key
#         year = station_key.split('_')[1]
#         years_found.append(year)
        
#         # Get results for this station-year
#         station_results = all_results[station_key]
        
#         # Check if we have the necessary data
#         if ('original_data' not in station_results or 
#             station_results['original_data'] is None or 
#             'timestamps' not in station_results or 
#             'predictions' not in station_results):
#             print(f"  Missing required data for {station_key}, skipping")
#             continue
        
#         # Determine value column name
#         if 'Value' in station_results['original_data'].columns:
#             value_col = 'Value'
#         elif 'vst_raw' in station_results['original_data'].columns:
#             value_col = 'vst_raw'
#         else:
#             # Try to find a suitable column
#             numeric_cols = station_results['original_data'].select_dtypes(include=[np.number]).columns
#             if len(numeric_cols) > 0:
#                 value_col = numeric_cols[0]
#             else:
#                 print(f"  No suitable value column found for {station_key}, skipping")
#                 continue
        
#         # Collect original data
#         all_original_timestamps.extend(station_results['original_data'].index)
#         all_original_values.extend(station_results['original_data'][value_col].values)
        
#         # Collect prediction data
#         all_prediction_timestamps.extend(station_results['timestamps'])
#         all_prediction_values.extend(station_results['predictions'])
        
#         # Collect error data
#         if 'prediction_errors' in station_results and 'timestamps' in station_results:
#             all_error_timestamps.extend(station_results['timestamps'])
#             all_error_values.extend(station_results['prediction_errors'])
        
#         # Collect anomaly points
#         if 'anomaly_flags' in station_results and 'timestamps' in station_results:
#             anomaly_indices = np.where(station_results['anomaly_flags'] == 1)[0]
#             if len(anomaly_indices) > 0:
#                 anomaly_timestamps = [station_results['timestamps'][i] for i in anomaly_indices]
#                 anomaly_values = [station_results['original_values'][i] for i in anomaly_indices]
#                 all_anomaly_timestamps.extend(anomaly_timestamps)
#                 all_anomaly_values.extend(anomaly_values)
    
#     # Sort all data by timestamp to ensure proper chronological order
#     if all_original_timestamps:
#         # Create DataFrames for sorting
#         original_df = pd.DataFrame({
#             'timestamp': all_original_timestamps,
#             'value': all_original_values
#         }).sort_values('timestamp')
        
#         prediction_df = pd.DataFrame({
#             'timestamp': all_prediction_timestamps,
#             'value': all_prediction_values
#         }).sort_values('timestamp')
        
#         error_df = pd.DataFrame({
#             'timestamp': all_error_timestamps,
#             'value': all_error_values
#         }).sort_values('timestamp')
        
#         anomaly_df = pd.DataFrame({
#             'timestamp': all_anomaly_timestamps,
#             'value': all_anomaly_values
#         }).sort_values('timestamp')
        
#         # Calculate statistics for annotations
#         total_points = len(original_df)
#         anomaly_count = len(anomaly_df)
#         anomaly_percentage = (anomaly_count / total_points) * 100 if total_points > 0 else 0
#         mean_error = np.mean(error_df['value']) if not error_df.empty else 0
        
#         # Calculate average threshold
#         thresholds = [all_results[k]['threshold'] for k in all_results 
#                      if k.startswith(f"{station_id}_") and 'threshold' in all_results[k]]
#         avg_threshold = np.mean(thresholds) if thresholds else None
        
#         # Create interactive plot with two subplots
#         fig = make_subplots(
#             rows=2, cols=1, 
#             shared_xaxes=True,
#             vertical_spacing=0.1,
#             subplot_titles=("Water Level Data", "Prediction Error"),
#             row_heights=[0.7, 0.3]
#         )
        
#         # Add original data trace
#         fig.add_trace(
#             go.Scatter(
#                 x=original_df['timestamp'], 
#                 y=original_df['value'],
#                 mode='lines',
#                 name='Original Data',
#                 line=dict(color='blue', width=1.5),
#                 hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
#             ),
#             row=1, col=1
#         )
        
#         # Add prediction trace
#         fig.add_trace(
#             go.Scatter(
#                 x=prediction_df['timestamp'], 
#                 y=prediction_df['value'],
#                 mode='lines',
#                 name='Predictions',
#                 line=dict(color='green', width=1.5),
#                 hovertemplate='%{x}<br>Prediction: %{y:.2f}<extra></extra>'
#             ),
#             row=1, col=1
#         )
        
#         # Add anomalies as scatter points
#         if not anomaly_df.empty:
#             fig.add_trace(
#                 go.Scatter(
#                     x=anomaly_df['timestamp'], 
#                     y=anomaly_df['value'],
#                     mode='markers',
#                     name='Detected Anomalies',
#                     marker=dict(color='red', size=4, symbol='circle'),
#                     hovertemplate='%{x}<br>Anomaly Value: %{y:.2f}<extra></extra>'
#                 ),
#                 row=1, col=1
#             )
        
#         # Add error trace
#         if not error_df.empty:
#             fig.add_trace(
#                 go.Scatter(
#                     x=error_df['timestamp'], 
#                     y=error_df['value'],
#                     mode='lines',
#                     name='Prediction Error',
#                     line=dict(color='red', width=1),
#                     hovertemplate='%{x}<br>Error: %{y:.4f}<extra></extra>'
#                 ),
#                 row=2, col=1
#             )
            
#             # Add threshold line if available
#             if avg_threshold is not None:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=[error_df['timestamp'].min(), error_df['timestamp'].max()],
#                         y=[avg_threshold, avg_threshold],
#                         mode='lines',
#                         name=f'Threshold ({avg_threshold:.2f})',
#                         line=dict(color='black', width=1, dash='dash'),
#                         hoverinfo='name'
#                     ),
#                     row=2, col=1
#                 )
        
#         # Set title and axis labels
#         year_range = f"{min(years_found)} to {max(years_found)}" if years_found else "All Years"
#         fig.update_layout(
#             title=f'Water Level Data and Predictions for Station {station_id} ({year_range})',
#             height=800,
#             width=1200,
#             hovermode='closest',
#             legend=dict(
#                 orientation="h",
#                 yanchor="bottom",
#                 y=1.02,
#                 xanchor="right",
#                 x=1
#             ),
#             # Enable full zooming capabilities
#             dragmode='zoom',  # Default to zoom mode instead of pan
#             xaxis=dict(
#                 rangeslider=dict(visible=True),
#                 type="date"
#             ),
#             # Make sure both axes are fully interactive
#             yaxis=dict(
#                 fixedrange=False,  # Allow y-axis zooming
#                 autorange=True
#             ),
#             yaxis2=dict(
#                 fixedrange=False,  # Allow y-axis zooming in the error plot too
#                 autorange=True
#             )
#         )
        
#         # Add modebar buttons for additional interactivity
#         fig.update_layout(
#             modebar_add=[
#                 'resetScale',  # Add button to reset axes
#                 'toggleSpikelines',  # Add button for reference lines
#                 'zoomIn2d',  # Add zoom in button
#                 'zoomOut2d'  # Add zoom out button
#             ]
#         )
        
#         # Add annotation with statistics
#         stats_text = (f"Total points: {total_points}<br>"
#                      f"Anomalies detected: {anomaly_count} ({anomaly_percentage:.2f}%)<br>"
#                      f"Mean error: {mean_error:.4f}")
        
#         fig.add_annotation(
#             xref="paper", yref="paper",
#             x=0.01, y=0.01,
#             text=stats_text,
#             showarrow=False,
#             font=dict(size=12),
#             bgcolor="white",
#             bordercolor="black",
#             borderwidth=1,
#             borderpad=4,
#             align="left"
#         )
        
#         return fig
#     else:
#         print(f"No data found for station {station_id}")
#         return None

# # Create the interactive plot
# fig = create_interactive_combined_plot(all_results, station_id='21006847')

# if fig:
#     # Save the HTML file
#     html_path = 'water_level_plot.html'
#     fig.write_html(html_path, include_plotlyjs='cdn')
    
#     # Open the file in a browser
#     absolute_path = os.path.abspath(html_path)
#     print(f"Opening plot in browser: {absolute_path}")
#     webbrowser.open('file://' + absolute_path)


# To save the figure:
# fig.savefig(f'station_21006845_all_years.png', dpi=300, bbox_inches='tight')





'''
1. Architecture: Encoder-Decoder with Bidirectional Pattern Recognition
The BidirectionalForecaster model uses a sophisticated encoder-decoder architecture:
Encoder Component
Bidirectional LSTM: Processes the input sequence in both forward and backward directions
Pattern Recognition: Captures complex patterns from both past and future context within the input window
Hidden State Fusion: Combines forward and backward hidden states to create a rich representation
Decoder Component
Unidirectional LSTM: Takes the encoder's hidden states and generates future predictions
Autoregressive Generation: Each predicted value becomes input for the next prediction
Temporal Causality: Only uses information available up to the prediction point

# During forward pass:
# 1. Process input with bidirectional encoder
encoder_out, (h_n, c_n) = self.encoder(x)

# 2. Combine forward/backward states 
h_n_forward = h_n[0::2]  # Forward direction
h_n_backward = h_n[1::2]  # Backward direction
h_n_combined = torch.cat([h_n_forward, h_n_backward], dim=2)

# 3. Generate predictions autoregressively with decoder
for _ in range(self.output_length):
    out, (h_state, c_state) = self.decoder(decoder_input, (h_state, c_state))
    next_pred = self.output_fc(out).view(batch_size, 1, 1)
    outputs.append(next_pred)
    decoder_input = next_pred  # Use prediction as next input

2. Sliding Window Forecasting with Ensemble Predictions
The system uses an advanced multi-window approach to generate robust predictions:
Sliding Window Creation
Input Windows: Each window contains input_length (72) historical points
Forecast Horizons: For each window, the model predicts output_length (8) future points
Stride Parameter: Windows slide forward by stride (12) points each time
Multiple Perspectives: This creates overlapping windows that view the same future point from different past contexts
# Calculate total forecasting windows
total_windows = (len(df) - input_length - forecast_horizon + 1) // stride

# Process each window
for i in range(0, len(df) - input_length - forecast_horizon + 1, stride):
    # Extract input window (past data)
    input_window = df.iloc[i:i+input_length]
    
    # Extract target window (future data to predict)
    target_window = df.iloc[i+input_length:i+input_length+forecast_horizon]
    
    # Generate forecasts and store results
    # ...

3. Ensemble Prediction Mechanism
This is where the power of the approach becomes evident:
Multiple Forecasts Per Point: Each future point gets predicted multiple times from different windows
Ensemble Averaging: All predictions are averaged to produce the final output
This approach:
Combines diverse perspectives from multiple windows
Reduces overfitting by averaging out idiosyncrasies of individual forecasts
Provides a more stable and accurate prediction

4. Aggregation of Predictions: For each timestamp, predictions from all windows are averaged
This final step:
Ensures a single, consistent prediction for each point in time
Provides a robust and accurate forecast across the entire dataset
The ensemble effect is critical - it:
Reduces noise in individual predictions
Captures different aspects of the pattern from different contexts
Makes prediction more robust to local irregularities
Improves overall forecast accuracy

3. Context-Aware Anomaly Detection
After generating predictions, the system uses a sophisticated context-aware approach to detect anomalies:
Local Error Analysis
Error Calculation: forecast_errors = np.abs(aligned_predicted - aligned_actual)
Local Statistics: Calculate rolling mean and standard deviation of errors
  error_series = pd.Series(forecast_errors)
  window_size_local = 48  # 12 hours with 15-min data
  rolling_mean = error_series.rolling(window=window_size_local, center=True).mean()
  rolling_std = error_series.rolling(window=window_size_local, center=True).std()

Z-Score Based Flagging
Local Z-Scores: Calculate how unusual each error is in its local context
z_scores = (forecast_errors - rolling_mean) / rolling_std

Anomaly Detection Threshold
  anomaly_flags = (z_scores_local > z_threshold).astype(int)

This approach is particularly powerful because:
It adapts to different error levels in different parts of the time series
It accounts for seasonal or time-dependent variability
It can detect subtle anomalies in low-variance regions
It's less likely to flag normal variations in high-variance regions

4. The Complete Process Flow
Training Phase:
Model is trained on clean data from multiple years
Learns normal water level patterns for each station
Uses early stopping to prevent overfitting

Evaluation Phase:
Apply trained model to new data with potential anomalies
For each station:
Create sliding windows with overlap
Generate predictions from each window
Combine predictions into ensemble forecast
Smooth predictions to reduce noise
Calculate local error statistics
Flag anomalies using adaptive z-score threshold
Calculate performance metrics against ground truth
Provide detailed visualization and analysis

Memory Efficiency:
Processes data in manageable windows
Doesn't require loading entire time series into memory
Scalable to very long time series

'''