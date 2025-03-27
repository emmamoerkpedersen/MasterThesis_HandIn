import torch.nn as nn

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
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

    def update_sequence_length(self, sequence_length):
        """
        Update the sequence length after it's been calculated
        """
        self.sequence_length = sequence_length

    def forward(self, x):
        # Forward pass doesn't depend on sequence_length attribute
        lstm_out, _ = self.lstm(x)
        
        if self.training:
            lstm_out = self.dropout(lstm_out)
            
        predictions = self.fc(lstm_out)
        
        return predictions
