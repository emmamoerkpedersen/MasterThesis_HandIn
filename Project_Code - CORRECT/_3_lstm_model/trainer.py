import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class LSTMTrainer:
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
        
        Args:
            data: DataFrame containing the time series data
            is_training: Whether this is training data (for fitting scaler)
            
        Returns:
            torch.Tensor: Prepared data ready for model
        """
        # Extract features specified in config
        feature_cols = self.config.get('feature_cols', ['vst_raw'])
        values = data[feature_cols].values
        
        # Fit scaler on training data only
        if is_training and not self.is_fitted:
            self.scaler.fit(values)
            self.is_fitted = True
        
        # Transform the data
        scaled_data = self.scaler.transform(values)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Convert to tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        return X, y

    def _create_sequences(self, data):
        """
        Create input/output sequences for training.
        
        Args:
            data: Scaled numpy array of shape [samples, features]
            
        Returns:
            tuple: (X, y) where X is input sequences and y is target values
        """
        X, y = [], []
        seq_length = self.config.get('sequence_length', 72)
        
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
            
        return np.array(X), np.array(y)

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

    def train(self, train_data, val_data, epochs=100, batch_size=32, patience=10):
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
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, is_training=True)
        X_val, y_val = self.prepare_data(val_data, is_training=False)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
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