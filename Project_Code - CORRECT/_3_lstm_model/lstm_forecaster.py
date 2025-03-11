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
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate')
        )
        self.criterion = nn.MSELoss()
        
        # Initialize data scaler
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
        
        # Print raw data statistics
        print("\nRaw data statistics:")
        for col in self.config['feature_cols']:
            print(f"{col}:")
            print(f"  min: {features[col].min():.4f}")
            print(f"  max: {features[col].max():.4f}")
            print(f"  mean: {features[col].mean():.4f}")
            print(f"  std: {features[col].std():.4f}")
        
        # Handle NaN values - We need to figure out how we handle this
        if features.isna().any().any():
            print("\nWarning: NaN values found in features. Filling with forward fill then backward fill.")
            features = features.ffill().bfill()
            
        # Scale each feature independently, might not be needed to scale independently
        if is_training and not self.is_fitted:
            self.scalers = {col: MinMaxScaler(feature_range=(-1, 1)) for col in self.config['feature_cols']}
            for col in self.config['feature_cols']:
                self.scalers[col].fit(features[[col]])
            self.is_fitted = True
            
        # Transform each feature
        scaled_features = []
        for col in self.config['feature_cols']:
            scaled_col = self.scalers[col].transform(features[[col]])
            scaled_features.append(scaled_col)
            
        # Combine scaled features
        scaled_data = np.hstack(scaled_features)
            
        # Create sequences
        X, y = self._create_sequences(scaled_data)
            
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def _create_sequences(self, data):
        """
        Create input/output sequences f or training, maintaining temporal order.
        """
        print(f"Data shape: {data.shape}, Data size in MB: {data.nbytes / (1024 * 1024):.2f}")
        X, y = [], []
        sequence_length = self.config.get('sequence_length')

        # Get index of target feature (vst_raw)
        target_idx = self.config['feature_cols'].index('vst_raw')

        # Create sequences in temporal order
        for i in range(len(data) - sequence_length):
            # Input sequence includes all features
            sequence = data[i:(i + sequence_length)]
            target = data[i + sequence_length, target_idx]
            # Skip if sequence contains invalid values
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                continue
            
            X.append(sequence)
            y.append(target)
        
        # Convert to numpy arrays while maintaining order
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        return X, y

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
        batch_predictions = []
        batch_targets = []
        
        for batch_X, batch_y in tqdm(train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad() # Zero the gradients
            outputs, _ = self.model(batch_X) # LSTM forward pass
            loss = self.criterion(outputs, batch_y) # MSE loss
            
            # Store predictions and targets
            batch_predictions.extend(outputs.detach().cpu().numpy()) # Detach and convert to numpy
            batch_targets.extend(batch_y.detach().cpu().numpy()) # Detach and convert to numpy
            
            loss.backward() # Backward pass
            self.optimizer.step() # Update weights
            total_loss += loss.item() # Accumulate loss
        
        # Print statistics about predictions
        predictions = np.array(batch_predictions) # Convert to numpy
        targets = np.array(batch_targets) # Convert to numpy
        print("\nTraining statistics:")
        print(f"Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}")
        print(f"Targets - min: {targets.min():.4f}, max: {targets.max():.4f}")
        
        return total_loss / len(train_loader) 

    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader containing validation data
            
        Returns:
            float: Validation loss
        """
        self.model.eval() # Set model to evaluation mode
        total_loss = 0 # Initialize total loss  
        
        with torch.no_grad(): # Disable gradient calculation
            for batch_X, batch_y in tqdm(val_loader, desc="Validating", leave=False):
                outputs, _ = self.model(batch_X) # LSTM forward pass
                loss = self.criterion(outputs, batch_y) # MSE loss
                total_loss += loss.item() # Accumulate loss
                
        return total_loss / len(val_loader)

    def train(self, train_data, val_data, epochs=100, batch_size=32, patience=15):
        # Verify data before training
        def verify_data(X, y, name):
            print(f"\nVerifying {name} data:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print(f"y unique values: {len(torch.unique(y))}")
            print(f"y range: [{y.min().item():.4f}, {y.max().item():.4f}]")
            print(f"y mean: {y.mean().item():.4f}")
            print(f"y std: {y.std().item():.4f}")
            
            if len(torch.unique(y)) < 10:
                raise ValueError(f"Too few unique values in {name} targets!")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, is_training=True)
        X_val, y_val = self.prepare_data(val_data, is_training=False)
        
        # Verify data
        verify_data(X_train, y_train, "training")
        verify_data(X_val, y_val, "validation")
        
        # Print hyperparameters
        print("\nModel Hyperparameters:")
        print(f"Input Size: {self.model.input_size}")
        print(f"Hidden Size: {self.model.hidden_size}")
        print(f"Number of Layers: {self.model.num_layers}")
        print(f"Dropout Rate: {self.config.get('dropout')}")
        
        print("\nTraining Parameters:")
        print(f"Learning Rate: {self.config.get('learning_rate')}")
        print(f"Batch Size: {batch_size}")
        print(f"Max Epochs: {epochs}")
        print(f"Early Stopping Patience: {patience}")
        print(f"Sequence Length: {self.config.get('sequence_length')}")
        print(f"Features Used: {self.config.get('feature_cols', ['vst_raw'])}")
        print(f"Device: {self.device}")
        print("\nStarting training...\n")
        
        # Print data shapes
        print(f"Training Data Shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Validation Data Shapes - X: {X_val.shape}, y: {y_val.shape}\n")

        # Create data loaders WITHOUT shuffling
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Keep temporal order
            drop_last=False  # Keep all sequences
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

