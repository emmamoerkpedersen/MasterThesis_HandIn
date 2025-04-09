import torch.nn as nn

class LSTMModelUpdate(nn.Module):
    """
    NN model for time series forecasting.
    """
    def __init__(self, input_size, sequence_length, hidden_size, output_size, num_layers, dropout):  
        super(LSTMModelUpdate, self).__init__()
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
            dropout=dropout
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization after first LSTM
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        
        
        # LSTM layer with dropout
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Layer normalization after second LSTM
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # LSTM layer with dropout
        self.lstm3 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Dropout layer
        self.dropout2 = nn.Dropout(dropout)

        # Layer normalization after third LSTM
        self.layer_norm3 = nn.LayerNorm(hidden_size)

       
        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)
   
    def update_sequence_length(self, sequence_length):
        """
        Update the sequence length after it's been calculated
        """
        self.sequence_length = sequence_length

    def forward(self, x):
        # Forward pass doesn't depend on sequence_length attribute
        
        # First LSTM
        lstm1_out, _ = self.lstm(x)
        
        if self.training:
            # First dropout
            lstm1_out = self.dropout(lstm1_out)
            
        lstm1_out = self.layer_norm1(lstm1_out)
        
        # Second LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norm2(lstm2_out)
        
        # Third LSTM
        lstm3_out, _ = self.lstm3(lstm2_out)
        

        if self.training:
            # second dropout
            lstm3_out = self.dropout2(lstm3_out)
            
        lstm3_out = self.layer_norm3(lstm3_out)
        
        # Dense layer
        predictions = self.fc(lstm3_out)
        
        return predictions

