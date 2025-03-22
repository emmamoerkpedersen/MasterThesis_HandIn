import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from _3_lstm_model.model import LSTMModel
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = config['feature_cols']
        self.output_features = config['output_features'][0]

    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation. Scale data and create sequences.
        """
        # Get features and target
        feature_cols = self.feature_cols
        target_col = self.output_features   
        
        features = pd.concat([data[col] for col in feature_cols], axis=1)
        target = pd.DataFrame(data[target_col])

        # Scale data
        scaled_features, scaled_target = self.scale_data(features, target)

        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)

        # Convert to tensors and move to device
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def scale_data(self, features, target):
        """
        Scale features and target using MinMaxScaler.
        """
        if not self.is_fitted:
            self.scalers = {
                'features': {col: MinMaxScaler() for col in self.feature_cols},
                'target': MinMaxScaler()
            }
            for col in self.feature_cols:
                self.scalers['features'][col].fit(features[[col]])
            self.scalers['target'].fit(target)
            self.is_fitted = True

        # Scale features
        scaled_features = np.hstack([
            self.scalers['features'][col].transform(features[[col]]) 
            for col in self.feature_cols
        ])

        # Scale target and ensure correct shape
        scaled_target = self.scalers['target'].transform(target).flatten()
        
        return scaled_features, scaled_target

    def _create_sequences(self, features, targets):
        """
        Create non-overlapping sequences for sequence-to-sequence prediction.
        """
        sequence_length = self.config.get('sequence_length', 1000)
        X, y = [], []
        
        for i in range(0, len(features), sequence_length):
            if i + sequence_length <= len(features):
                feature_seq = features[i:(i + sequence_length)]
                target_seq = targets[i:(i + sequence_length)]
                X.append(feature_seq)
                y.append(target_seq)
        
        X = np.array(X)  # Shape: (num_full_sequences, sequence_length, num_features)
        y = np.array(y)[..., np.newaxis]  # Shape: (num_full_sequences, sequence_length, 1)
        
        return X, y


class LSTM_Trainer:
    def __init__(self, config, preprocessor):
        """
        Initialize the trainer and LSTM model.
        
        Args:
            config: Dictionary containing model and training parameters
            preprocessor: Instance of DataPreprocessor
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = preprocessor  # Use preprocessor for data handling

        # Initialize LSTM Model using parameters from config
        self.model = LSTMModel(
            input_size=len(config['feature_cols']),
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            output_size=len(config['output_features']),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.001))
        self.criterion = nn.MSELoss()

    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation.
        """
        self.model.train() if training else self.model.eval()
        total_loss = 0

        with torch.set_grad_enabled(training):
            for batch_X, batch_y in tqdm(data_loader, desc="Training" if training else "Validating", leave=False):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                if training:
                    loss.backward()
                    self.optimizer.step()

        return total_loss / len(data_loader)

    def train(self, train_data, val_data, epochs=100, batch_size=32, patience=15):
        """
        Train the LSTM model.
        """
        # Prepare data
        X_train, y_train = self.preprocessor.prepare_data(train_data, is_training=True)
        X_val, y_val = self.preprocessor.prepare_data(val_data, is_training=False)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
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
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return history

    def predict(self, data):
        """
        Make predictions on new data.
        """
        self.model.eval()
        X, y = self.preprocessor.prepare_data(data, is_training=False)

        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
            y = y.cpu().numpy()

            # Inverse transform predictions
            predictions_original = self.preprocessor.scalers['target'].inverse_transform(predictions.reshape(-1, 1))
            return predictions_original.reshape(predictions.shape), predictions, y

