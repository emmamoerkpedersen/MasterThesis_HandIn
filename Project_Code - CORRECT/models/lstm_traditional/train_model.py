import torch
import torch.nn as nn
import torch.optim as optim

from shared.preprocessing.preprocessing_LSTM import DataPreprocessor
from models.lstm_traditional.objective_functions import get_objective_function
from models.lstm_traditional.model import LSTMModel
from tqdm import tqdm


class LSTM_Trainer:
    def __init__(self, config, preprocessor):
        """
        Initialize the trainer and LSTM model.
        
        Args:
            config: Dictionary containing model and training parameters
            preprocessor: Instance of DataPreprocessor
        """
        self.config = config
        self.device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = preprocessor  # Use preprocessor for data handling

        # Initialize LSTM Model using parameters from config
        self.model = LSTMModel(
            input_size=len(preprocessor.feature_cols),
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            output_size=len(config['output_features']),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate'))
        self.criterion = get_objective_function(config.get('objective_function'))
        
        # # Add learning rate scheduler (removed verbose parameter)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',
        #     factor=0.8,  
        #     patience=3,  
        #     min_lr=1e-6  
        # )

    def _run_epoch(self, data_loader, training=True):
        """
        Runs an epoch for training or validation 
        """
        self.model.train() if training else self.model.eval()
        warmup_length = self.config.get('warmup_length', 100)
        total_loss = 0
        
        # Lists to store validation predictions and targets
        all_predictions = []
        all_targets = []

        with torch.set_grad_enabled(training):
            for batch_X, batch_y in tqdm(data_loader, desc="Training" if training else "Validating", leave=False):
                # Ensure batch tensors are on the correct device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Create warm-up mask
                warmup_mask = torch.ones_like(batch_y, dtype=torch.bool)
                warmup_mask[:, :warmup_length, :] = False

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

    def train(self, train_data, val_data, epochs, batch_size, patience):
        """
        Train the LSTM model with improved efficiency.
        """
        # Prepare data
        print(f"Train data length: {len(train_data)}")
        print(f"Validation data length: {len(val_data)}")
        X_train, y_train = self.preprocessor.prepare_data(train_data, is_training=True)
        X_val, y_val = self.preprocessor.prepare_data(val_data, is_training=False)
        
        # Ensure all tensors are on CPU since we're using CPU-only training
        X_train = X_train.cpu()
        y_train = y_train.cpu()
        X_val = X_val.cpu()
        y_val = y_val.cpu()

        # Create data loaders with num_workers for parallel data loading
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            batch_size=batch_size, 
            shuffle=True,  # Enable shuffling for better training
            num_workers=4  # Parallel data loading
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val), 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4
        )

        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}

        # Training loop with progress bar
        current_lr = self.optimizer.param_groups[0]['lr']
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss, val_predictions, val_targets = self._run_epoch(val_loader, training=False)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Step the scheduler based on validation loss
            #prev_lr = self.optimizer.param_groups[0]['lr']
            #self.scheduler.step(val_loss)
            #current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print learning rate change if it occurred
            # if current_lr != prev_lr:
            #     print(f'\nLearning rate updated: {prev_lr:.2e} -> {current_lr:.2e}')

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
        
        # Ensure tensors are on the correct device
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            # Make predictions in smaller chunks if needed
            predictions = self.model(X).cpu().numpy()
            y = y.cpu().numpy()
            
            # Preserve temporal order during inverse transform
            predictions_reshaped = predictions.reshape(-1, 1)
            predictions_original = self.preprocessor.feature_scaler.inverse_transform_target(predictions_reshaped)
            
            # Reshape back maintaining temporal order
            predictions_original = predictions_original.reshape(predictions.shape)
            
            return predictions_original, predictions, y