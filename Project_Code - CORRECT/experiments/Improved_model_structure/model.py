import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Attention mechanism to focus on important parts of the sequence
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Calculate attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.bmm(attention_weights, v)
        
        # Residual connection
        return x + context

class LSTMModel(nn.Module):
    """
    Enhanced NN model for time series forecasting with attention mechanism.
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
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Using unidirectional LSTM
        )

        # Attention mechanism for focusing on important parts of the sequence
        self.attention = AttentionLayer(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        attended = self.attention(lstm_out)
        
        if self.training:
            attended = self.dropout(attended)
        
        # First dense layer with ReLU activation
        fc1_out = self.relu(self.fc1(attended))
        
        if self.training:
            fc1_out = self.dropout(fc1_out)
            
        # Final output layer
        predictions = self.fc2(fc1_out)
        
        return predictions