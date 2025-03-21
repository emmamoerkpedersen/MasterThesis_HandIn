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
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )

        # Dropout layer - Outcommented for the overfitting
        #self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor of shape (batch_size, sequence_length, output_size)
        """

        # LSTM forward pass
        out, _ = self.lstm(x)  # out: [batch_size, seq_len, hidden_size]
        
        #out = self.dropout(out)
        # Pass through fully connected layer
        predictions = self.fc(out)  # Shape: (batch_size, sequence_length, output_size)
       
        return predictions

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
        # Get station data and features
        feature_cols = self.config['feature_cols']
        target_col = self.config['output_features'][0]
        
        # Create features DataFrame
        features = pd.concat([data[col] for col in feature_cols], axis=1)
        features.columns = feature_cols

        # Create target DataFrame
        target = pd.DataFrame(data[target_col])
        target.columns = [target_col]

        # Initialize scalers on first training pass
        if is_training and not self.is_fitted:
            self.scalers = {
                'features': {col: MinMaxScaler() for col in feature_cols},
                'target': MinMaxScaler()
            }
            # Fit feature scalers
            for col in feature_cols:
                self.scalers['features'][col].fit(features[[col]])
            # Fit target scaler
            self.scalers['target'].fit(target)
            self.is_fitted = True
        
        # Scale features
        scaled_features = np.hstack([
            self.scalers['features'][col].transform(features[[col]]) 
            for col in feature_cols
        ])
        
        # Scale target and ensure correct shape
        scaled_target = self.scalers['target'].transform(target).flatten()
        
        print("\nDebug shapes before sequence creation:")
        print(f"scaled_features shape: {scaled_features.shape}")
        print(f"scaled_target shape: {scaled_target.shape}")
        
        # Verify no NaN values
        if np.isnan(scaled_features).any():
            print("Warning: NaN values in scaled_features")
        if np.isnan(scaled_target).any():
            print("Warning: NaN values in scaled_target")
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)
        
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def _create_sequences(self, features, targets):
        """
        Create non-overlapping sequences for sequence-to-sequence prediction.
        """
        sequence_length = self.config.get('sequence_length', 1000)
        X, y = [], []
        
        # Use non-overlapping windows
        for i in range(0, len(features), sequence_length):
            if i + sequence_length <= len(features):
                feature_seq = features[i:(i + sequence_length)]
                target_seq = targets[i:(i + sequence_length)]
                X.append(feature_seq)
                y.append(target_seq)
        
        X = np.array(X)  # Shape: (num_full_sequences, sequence_length, num_features)
        y = np.array(y)[..., np.newaxis]  # Shape: (num_full_sequences, sequence_length, 1)
        
        print(f"\nSequence creation summary:")
        print(f"Input length: {len(features)}")
        print(f"Sequence length: {sequence_length}")
        print(f"Number of sequences: {len(X)}")
        print(f"Total predicted points: {len(X) * sequence_length}")
        
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
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Store predictions and targets
            batch_predictions.extend(outputs.detach().cpu().numpy())
            batch_targets.extend(batch_y.detach().cpu().numpy())
            
            # Backward pass
            loss.backward()
        
            
            # Update weights
            self.optimizer.step()
            total_loss += loss.item()
        
        # Print statistics about predictions
        predictions = np.array(batch_predictions)
        targets = np.array(batch_targets)
        # print("\nTraining statistics:")
        # print(f"Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}")
        # print(f"Targets - min: {targets.min():.4f}, max: {targets.max():.4f}")
        
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validating", leave=False):
                outputs = self.model(batch_X) # LSTM forward pass
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
        print(f"Sequence Length: {self.model.sequence_length}")
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
        X, y = self.prepare_data(data, is_training=False)  # Changed _ to y to capture scaled targets
        
        with torch.no_grad():
            predictions = self.model(X)
            predictions = predictions.cpu().numpy()
            y = y.cpu().numpy()  # Convert scaled targets to numpy
            
            # Now predictions and y contain the scaled predictions and targets
            
            # Inverse transform predictions using target scaler
            target_col = self.config['output_features'][0]
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_original = self.scalers['target'].inverse_transform(predictions_reshaped)
            
            return predictions_original.reshape(predictions.shape), predictions, y

def create_full_plot(test_data, test_predictions, station_id):
    """
    Create an interactive plot with aligned datetime indices.
    """
    # Get the actual test data with its datetime index
    test_actual = test_data['vst_raw']
    
    # Reshape predictions from (sequences, sequence_length, 1) to 1D array
    test_predictions = test_predictions.reshape(-1)  # Flatten the predictions
    
    # Print lengths for debugging
    print(f"Length of test_actual: {len(test_actual)}")
    print(f"Length of predictions: {len(test_predictions)}")
    
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


def plot_scaled_predictions(predictions, targets, title="Scaled Predictions vs Targets"):
        """
        Plot scaled predictions and targets before inverse transformation.
        """
        # Create figure
        fig = go.Figure()
        
        # Flatten predictions and targets for plotting
        flat_predictions = predictions.reshape(-1)
        flat_targets = targets.reshape(-1)
        
        # Create x-axis points
        x_points = np.arange(len(flat_predictions))
        
        # Add targets
        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=flat_targets,
                name="Scaled Targets",
                line=dict(color='blue', width=1)
            )
        )

        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=flat_predictions,
                name="Scaled Predictions",
                line=dict(color='red', width=1)
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Timestep',
            yaxis_title='Scaled Value',
            width=1200,
            height=600,
            showlegend=True
        )
        
        # Save and open in browser
        html_path = 'scaled_predictions.html'
        fig.write_html(html_path)
        print(f"Opening scaled predictions plot in browser...")
        webbrowser.open('file://' + os.path.abspath(html_path))

def plot_convergence(history, title="Training and Validation Loss"):
    """
    Plot training and validation loss over epochs to visualize convergence.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
    """
    # Create figure
    fig = go.Figure()
    
    # Get epochs array
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Add training loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['train_loss'],
            name="Training Loss",
            line=dict(color='blue', width=1)
        )
    )

    # Add validation loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['val_loss'],
            name="Validation Loss",
            line=dict(color='red', width=1)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        width=1200,
        height=600,
        showlegend=True,
        yaxis_type="log"  # Use log scale for better visualization
    )
    
    # Save and open in browser
    html_path = 'convergence_plot.html'
    fig.write_html(html_path)
    print(f"Opening convergence plot in browser...")
    webbrowser.open('file://' + os.path.abspath(html_path))
