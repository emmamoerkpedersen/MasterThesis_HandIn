import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

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
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate')
        )
        
        # Use reduction='none' for element-wise weighting
        self.vst_criterion = nn.MSELoss(reduction='none')
        self.vinge_criterion = nn.MSELoss(reduction='none')
        
        # Weight for vinge loss
        self.vinge_weight = config.get('vinge_weight', 2.0)
        
        # Initialize scalers
        self.scalers = {}
        self.is_fitted = False

    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation.
        """
        # Get station data
        station_id = list(data.keys())[0]
        
        # Create DataFrame with all features
        features = pd.concat([data[station_id][col] for col in self.config['feature_cols']], axis=1)
        features.columns = self.config['feature_cols']
        
        # Get vst_raw and vinge data
        vst_raw = data[station_id]['vst_raw']
        vinge = data[station_id]['vinge']
        
        # Create vinge mask
        vinge_mask = ~vinge.isna()
        
        # Handle NaN values in features
        if features.isna().any().any():
            features = features.ffill().bfill()
            
        # Scale features and targets
        if is_training and not self.is_fitted:
            # Initialize scalers for features
            self.scalers = {
                'features': {col: MinMaxScaler(feature_range=(-1, 1)) 
                           for col in self.config['feature_cols']},
                'vst_raw': MinMaxScaler(feature_range=(-1, 1)),
                'vinge': MinMaxScaler(feature_range=(-1, 1))
            }
            
            # Fit scalers
            for col in self.config['feature_cols']:
                self.scalers['features'][col].fit(features[[col]])
            
            self.scalers['vst_raw'].fit(vst_raw.values.reshape(-1, 1))
            
            # Fit vinge scaler only on non-NaN values
            valid_vinge = vinge.dropna().values.reshape(-1, 1)
            if len(valid_vinge) > 0:
                self.scalers['vinge'].fit(valid_vinge)
            
            self.is_fitted = True
        
        # Transform features
        scaled_features = []
        for col in self.config['feature_cols']:
            scaled_col = self.scalers['features'][col].transform(features[[col]])
            scaled_features.append(scaled_col)
        
        # Combine scaled features
        scaled_data = np.hstack(scaled_features)
        
        # Scale targets
        scaled_vst = self.scalers['vst_raw'].transform(vst_raw.values.reshape(-1, 1))
        
        # Scale vinge data where available
        scaled_vinge = np.zeros_like(scaled_vst)
        scaled_vinge.fill(np.nan)
        if vinge_mask.any():
            valid_vinge = vinge[vinge_mask].values.reshape(-1, 1)
            scaled_vinge[vinge_mask] = self.scalers['vinge'].transform(valid_vinge).flatten()
        
        # Create sequences
        X, y_vst, y_vinge, seq_vinge_mask = self._create_sequences(
            scaled_data, scaled_vst, scaled_vinge, vinge_mask.values
        )
        
        return (torch.FloatTensor(X).to(self.device),
                torch.FloatTensor(y_vst).to(self.device),
                torch.FloatTensor(y_vinge).to(self.device),
                torch.BoolTensor(seq_vinge_mask).to(self.device))

    def _create_sequences(self, data, vst_targets, vinge_targets, vinge_mask):
        """
        Create sequences for training/validation.
        """
        X, y_vst, y_vinge, masks = [], [], [], []
        sequence_length = self.config.get('sequence_length')
        
        for i in range(len(data) - sequence_length):
            sequence = data[i:(i + sequence_length)]
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                continue
            
            X.append(sequence)
            y_vst.append(vst_targets[i + sequence_length])
            y_vinge.append(vinge_targets[i + sequence_length])
            masks.append(vinge_mask[i + sequence_length])
        
        return (np.array(X),
                np.array(y_vst),
                np.array(y_vinge),
                np.array(masks))

    def train_epoch(self, train_loader):
        """
        Train for one epoch using both objectives.
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_X, batch_y_vst, batch_y_vinge, batch_mask in tqdm(train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            predictions, _ = self.model(batch_X)
            
            # Calculate losses for both objectives
            vst_losses = self.vst_criterion(predictions, batch_y_vst)
            vinge_losses = self.vinge_criterion(predictions, batch_y_vinge)
            
            # Combine losses with weights
            # vst_raw always has weight 1.0
            # vinge has weight vinge_weight where available, 0 elsewhere
            batch_loss = vst_losses + (vinge_losses * self.vinge_weight * batch_mask)
            
            # Take mean for the final loss
            batch_loss = batch_loss.mean()
            
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()
            
        return total_loss / num_batches

    def validate(self, val_loader):
        """
        Validate using both objectives.
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_X, batch_y_vst, batch_y_vinge, batch_mask in tqdm(val_loader, desc="Validating", leave=False):
                predictions, _ = self.model(batch_X)
                
                # Calculate losses for both objectives
                vst_losses = self.vst_criterion(predictions, batch_y_vst)
                vinge_losses = self.vinge_criterion(predictions, batch_y_vinge)
                
                # Combine losses with weights
                batch_loss = vst_losses + (vinge_losses * self.vinge_weight * batch_mask)
                total_loss += batch_loss.mean().item()
                
        return total_loss / num_batches

    def train(self, train_data, val_data, epochs=100, batch_size=32, patience=15):
        # Prepare data with both targets
        X_train, y_train_vst, y_train_vinge, train_mask = self.prepare_data(train_data, is_training=True)
        X_val, y_val_vst, y_val_vinge, val_mask = self.prepare_data(val_data, is_training=False)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            X_train, y_train_vst, y_train_vinge, train_mask
        )
        val_dataset = torch.utils.data.TensorDataset(
            X_val, y_val_vst, y_val_vinge, val_mask
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False
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
        """
        self.model.eval()
        X, _ = self.prepare_data(data, is_training=False)
        
        with torch.no_grad():
            predictions, _ = self.model(X)
        
        # Convert predictions back to original scale
        predictions = predictions.cpu().numpy()
        # Use the vst_raw scaler to inverse transform predictions
        predictions_reshaped = np.zeros((predictions.shape[0], 1))
        predictions_reshaped[:, 0] = predictions.flatten()
        
        return self.scalers['vst_raw'].inverse_transform(predictions_reshaped)[:, 0]

def create_full_plot(test_data, test_predictions, station_id, sequence_length=72):
    """
    Create an interactive plot with aligned datetime indices.
    """
    # Get the actual test data with its datetime index
    test_actual = test_data['vst_raw']  # Fixed: access using station_id
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
    # Calculate the difference in lengths
    print(f"Sequence length: {sequence_length}")
    
    # Trim the actual data to match predictions
    if len(test_predictions) > len(test_actual):
        print("Trimming predictions to match actual data length")
        test_predictions = test_predictions[:len(test_actual)]
    else:
        print("Using full predictions")
    
    # Create a pandas Series for predictions with the matching datetime index
    predictions_series = pd.Series(
        data=test_predictions,
        index=test_actual.index[:len(test_predictions)],
        name='Predictions'
    )
    
    # Print final shapes for verification
    print(f"Final test data shape: {test_actual.shape}")
    print(f"Final predictions shape: {predictions_series.shape}")
    
    # Create figure
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(
            x=test_actual.index,
            y=test_actual.values,
            name="Actual",
            line=dict(color='blue', width=1)
        )
    )

    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=predictions_series.index,
            y=predictions_series.values,
            name="Predicted",
            line=dict(color='red', width=1)
        )
    )

    # Update layout
    fig.update_layout(
        title=f'Water Level - Actual vs Predicted (Station {station_id[0]})',
        xaxis_title='Time',
        yaxis_title='Water Level',
        width=1200,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"  # This will format the x-axis as dates
        )
    )

    # Save and open in browser
    html_path = 'predictions_with_dates.html'
    fig.write_html(html_path)
    
    # Open in browser
    absolute_path = os.path.abspath(html_path)
    print(f"Opening plot in browser: {absolute_path}")
    webbrowser.open('file://' + absolute_path)




########################################################
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent  # Go up one level from current file
    sys.path.append(str(project_root))

    import pandas as pd
    from pathlib import Path
    from _1_preprocessing.split import split_data_rolling, split_data_yearly

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
    print("\nSplitting data into yearly")
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

    #Use the LSTM configuration from config.py

    print(f"Input feature size: {len(lstm_config.get('feature_cols'))}")

    # Initialize model and trainer
    model = LSTMModel(
        input_size=len(lstm_config['feature_cols']),
        sequence_length=lstm_config.get('sequence_length'),
        hidden_size=lstm_config.get('hidden_size'),
        output_size=1,
        num_layers=lstm_config.get('num_layers'),
        dropout=lstm_config.get('dropout')
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
            epochs=lstm_config.get('epochs'),
            batch_size=lstm_config.get('batch_size'),
            patience=lstm_config.get('patience')
        )
        # Optionally save the model after each window
        torch.save(model.state_dict(), f'model_window_{window_idx}.pth')

    #After training, you can use the model for predictions
    test_predictions = trainer.predict(split_datasets['test'])
    # After making predictions, create and show the plot
    create_full_plot(test_data, test_predictions, station_id)

