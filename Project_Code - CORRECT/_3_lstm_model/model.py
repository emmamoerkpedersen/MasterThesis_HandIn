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
            dropout=dropout
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

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
        
        out = self.dropout(out)
        # Pass through fully connected layer
        predictions = self.fc(out)  # Shape: (batch_size, sequence_length, output_size)
       
        return predictions