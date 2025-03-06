import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.svm import OneClassSVM

def smooth_hinge_loss(x, tau=10):
    return (1 / tau) * torch.log(1 + torch.exp(tau * x))

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h_mean = torch.mean(lstm_out, dim=1)  # Mean pooling over time
        return h_mean

class LSTMOcSVM(nn.Module):
    def __init__(self, input_size, hidden_size, lambda_param=0.1, tau=10):
        super(LSTMOcSVM, self).__init__()
        self.lstm = LSTMFeatureExtractor(input_size, hidden_size)
        self.w = nn.Parameter(torch.randn(hidden_size))
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.lambda_param = lambda_param
        self.tau = tau
    
    def forward(self, x):
        h = self.lstm(x)
        decision_score = torch.matmul(h, self.w) - self.rho
        return decision_score
    
    def loss(self, x):
        h = self.lstm(x)
        decision_score = torch.matmul(h, self.w) - self.rho
        slack = smooth_hinge_loss(self.rho - decision_score, self.tau)
        return 0.5 * torch.norm(self.w, p=2)**2 + (1 / (len(x) * self.lambda_param)) * torch.sum(slack) - self.rho

# Generate dummy time series data
np.random.seed(42)
torch.manual_seed(42)

num_samples = 1000
time_steps = 10
feature_dim = 5
hidden_size = 10

X_train = torch.randn((num_samples, time_steps, feature_dim))

# Initialize and train the model
model = LSTMOcSVM(input_size=feature_dim, hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    loss = model.loss(X_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Inference: Compute decision scores
X_test = torch.randn((100, time_steps, feature_dim))
decision_scores = model(X_test).detach().numpy()
anomalies = np.sign(decision_scores)  # -1 for anomalies, 1 for normal

print("Sample Anomaly Scores:", decision_scores[:10])