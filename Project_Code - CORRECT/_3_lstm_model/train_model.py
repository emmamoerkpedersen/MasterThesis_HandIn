import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from _3_lstm_model.model import LSTMModel
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = config['feature_cols']+['feature_station_vst_raw']
        self.output_features = config['output_features'][0]
    
    def load_and_split_data(self, project_root, station_id, feature_station_id):
        """
        Load and Split data into features and target.
        """
        data_dir = project_root / "data_utils" / "Sample data"
        data = pd.read_pickle(data_dir / "preprocessed_data.pkl")

        # Check if station_id exists in the data dictionary, if not return empty dict
        station_data = data.get(station_id)
        feature_station_data = data.get(feature_station_id)

        if not station_data:
            raise ValueError(f"Station ID {station_id} not found in the data.")
    
        # Extract vst_raw from feature_station_data
        feature_station_data = feature_station_data['vst_raw']['vst_raw'].rename('feature_station_vst_raw')

        # Concatenate all station data columns
        df = pd.concat(station_data.values(), axis=1)
        df = pd.concat([df, feature_station_data], axis=1)

        # Start_date is first rainfall not nan, End_date is last vst_raw not nan
        start_date = df['rainfall'].first_valid_index()
        end_date = df['vst_raw'].last_valid_index()
        # Cut dataframe
        data = df[(df.index >= start_date) & (df.index <= end_date)]


        
        # Fill temperature and rainfall Nan with bfill and ffill
        data.loc[:, 'temperature'] = data['temperature'].ffill().bfill()
        data.loc[:, 'rainfall'] = data['rainfall'].fillna(-1)
        data.loc[:, 'feature_station_vst_raw'] = data['feature_station_vst_raw'].fillna(-1)
        print(f"  - Filled temperature and rainfall Nan with bfill and ffill")

        feature_cols = self.feature_cols
        target_feature = self.output_features
        all_features = list(set(feature_cols + [target_feature]))        
        #Filter data to only include the features and target feature
        data = data[all_features]

        # Split data into train, val and test
        train_data, temp = train_test_split(data, test_size=0.4, shuffle=False)
        val_data, test_data = train_test_split(temp, test_size=0.5, shuffle=False)
        print(f'Data shape: {data.shape}')
        print(f'Train data shape: {train_data.shape}')
        print(f'Val data shape: {val_data.shape}')
        print(f'Test data shape: {test_data.shape}')

        return train_data, val_data, test_data
    
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
        print(f'X shape: {X.shape}, y shape: {y.shape}')
        # Convert to tensors and move to device
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def scale_data(self, features, target):
        """
        Scale features and target using StandardScaler.
        """
        if not self.is_fitted:
            self.scalers = {
                'features': {col: StandardScaler() for col in self.feature_cols},
                'target': StandardScaler()
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
            input_size=len(config['feature_cols']+['feature_station_vst_raw']),
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            output_size=len(config['output_features']),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate'))
        self.criterion = nn.MSELoss()

    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation 
        """
        self.model.train() if training else self.model.eval()
        total_loss = 0
        
        # Lists to store validation predictions and targets
        all_predictions = []
        all_targets = []

        with torch.set_grad_enabled(training):
            for batch_X, batch_y in tqdm(data_loader, desc="Training" if training else "Validating", leave=False):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Create warm-up mask
                warmup_mask = torch.ones_like(batch_y, dtype=torch.bool)
                warmup_mask[:, :50, :] = False

                # Combine warm-up mask with NaN mask
                non_nan_mask = ~torch.isnan(batch_y)
                valid_mask = non_nan_mask & warmup_mask

                valid_outputs = outputs[valid_mask]
                valid_target = batch_y[valid_mask]
                if valid_target.size(0) == 0:
                    continue

                loss = self.criterion(valid_outputs, valid_target)
                
                if training:
                    loss.backward()

                    
                    self.optimizer.step()
                
                total_loss += loss.item()

                if not training:
                    all_predictions.append(outputs.cpu().detach())
                    all_targets.append(batch_y.cpu())

        if training:
            return total_loss / len(data_loader)
        else:
            # Concatenate all validation predictions and targets
            val_predictions = torch.cat(all_predictions, dim=0)
            val_targets = torch.cat(all_targets, dim=0)
            return total_loss / len(data_loader), val_predictions, val_targets

    def train(self, train_data, val_data, epochs=100, batch_size=32, patience=15):
        """
        Train the LSTM model.
        """
        # Prepare data
        print(f"Train data length: {len(train_data)}")
        print(f"Validation data length: {len(val_data)}")
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
            val_loss, val_predictions, val_targets = self._run_epoch(val_loader, training=False)

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

        return history, val_predictions, val_targets

    def predict(self, data):
        """
        Make predictions on new data with proper sequence handling.
        """
        self.model.eval()
        X, y = self.preprocessor.prepare_data(data, is_training=False)
        
        with torch.no_grad():
            # Make predictions in smaller chunks if needed
            predictions = self.model(X).cpu().numpy()
            y = y.cpu().numpy()
            
            # Preserve temporal order during inverse transform
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_original = self.preprocessor.scalers['target'].inverse_transform(predictions_reshaped)
            
            # Reshape back maintaining temporal order
            predictions_original = predictions_original.reshape(predictions.shape)
            
            return predictions_original, predictions, y