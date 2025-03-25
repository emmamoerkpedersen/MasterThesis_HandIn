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
        self.feature_cols = config['feature_cols'] + ['feature_station_vst_raw']
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
        val_data, test_data = train_test_split(temp, test_size=0.25, shuffle=False)
        print(f'Data shape: {data.shape}')
        print(f'Train data shape: {train_data.shape}')
        print(f'Val data shape: {val_data.shape}')
        print(f'Test data shape: {test_data.shape}')

    
        return train_data, val_data, test_data
    
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for training or validation.
        """
        feature_cols = self.feature_cols
        target_col = self.output_features   
        
        features = pd.concat([data[col] for col in feature_cols], axis=1)
        target = pd.DataFrame(data[target_col])

        # Scale data
        scaled_features, scaled_target = self.scale_data(features, target)
        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target, is_validation=not is_training)
        print(f'Prepared data - X shape: {X.shape}, y shape: {y.shape}')
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

    def _create_sequences(self, features, targets, is_validation=False):
        """
        Create sequences for training or validation.
        For validation, we want to keep the entire sequence intact to use training data as warmup.
        """
        sequence_length = self.config.get('sequence_length', 1000)
        X, y = [], []
        
        if is_validation:
            # For validation, keep the entire sequence as one piece
            # This preserves the training data for warmup
            X = [features]  # Keep all data as one sequence
            y = [targets]
            print("Validation sequence creation:")
            print(f"Total sequence length: {len(features)}")
        else:
            # For training, create normal sequences
            for i in range(0, len(features) - sequence_length + 1, sequence_length):
                feature_seq = features[i:(i + sequence_length)]
                target_seq = targets[i:(i + sequence_length)]
                X.append(feature_seq)
                y.append(target_seq)
        
        X = np.array(X)
        y = np.array(y)[..., np.newaxis]
        
        print(f"Created sequences with shape - X: {X.shape}, y: {y.shape}")
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
        self.criterion = nn.SmoothL1Loss()

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',           # Reduce LR when validation loss stops decreasing
            factor=0.1,          # Multiply LR by this factor when reducing
            patience=5,          # Number of epochs with no improvement after which LR will be reduced
            verbose=True,        # Print message when LR is reduced
            min_lr=1e-6         # Don't reduce LR below this value
        )

    def _run_epoch(self, data_loader, training=True, train_data_length=None):
        self.model.train() if training else self.model.eval()
        total_loss = 0
        num_valid_batches = 0
        
        all_predictions = []
        all_targets = []

        with torch.set_grad_enabled(training):
            for batch_idx, (batch_X, batch_y) in enumerate(tqdm(data_loader, desc="Training" if training else "Validating", leave=False)):
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                if training:
                    # During training, only mask NaN values
                    valid_mask = ~torch.isnan(batch_y)
                else:
                    # Use training data as warmup
                    # For validation, only use points training data
                    valid_mask = torch.zeros_like(batch_y, dtype=torch.bool)
                    valid_mask[:, train_data_length:, :] = True
                    non_nan_mask = ~torch.isnan(batch_y)
                    valid_mask = valid_mask & non_nan_mask
                
                # Get valid outputs and targets
                valid_outputs = outputs[valid_mask]
                valid_target = batch_y[valid_mask]
                
                if valid_target.size(0) > 0:
                    loss = self.criterion(valid_outputs, valid_target)
                    total_loss += loss.item()
                    num_valid_batches += 1
                    
                    if training:
                        loss.backward()
                        self.optimizer.step()

                if not training:
                    all_predictions.append(outputs.cpu().detach())
                    all_targets.append(batch_y.cpu())

            if training:
                return total_loss / max(num_valid_batches, 1)
            else:
                val_predictions = torch.cat(all_predictions, dim=0)
                val_targets = torch.cat(all_targets, dim=0)
                
                return total_loss / max(num_valid_batches, 1), val_predictions, val_targets

    def train(self, train_data, val_data, epochs=100, batch_size=1, patience=15):
        """
        Train the LSTM model with proper sequence handling for validation.
        """
        print(f"\nData Sizes:")
        print(f"Train data length: {len(train_data)}")
        print(f"Validation data length: {len(val_data)}")
        
        # Prepare training data normally
        X_train, y_train = self.preprocessor.prepare_data(train_data, is_training=True)
        
        # For validation, combine train and validation data to use training data as warmup
        combined_val_data = pd.concat([train_data, val_data])
        print(f"Combined validation data length: {len(combined_val_data)}")
        X_val, y_val = self.preprocessor.prepare_data(combined_val_data, is_training=False)
        
        train_data_length = len(train_data)
        print(f"Using {train_data_length} steps as warmup for validation")
        
        print(f"\nTensor Shapes:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            batch_size=batch_size, 
            shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val), 
            batch_size=1,  # Force batch_size=1 for validation
            shuffle=False
        )

        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss, val_predictions, val_targets = self._run_epoch(
                val_loader, 
                training=False, 
                train_data_length=train_data_length
            )

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

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
