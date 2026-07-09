# src/dynamic/model.py

import torch
import torch.nn as nn

class ISLSequenceClassifier(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super(ISLSequenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_out_dim = hidden_size * 2
        pooling_concatenated_dim = lstm_out_dim * 2 
        
        # Fully Connected Classification Head
        self.fc1 = nn.Linear(pooling_concatenated_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Forward pass through BiLSTM
        out, _ = self.lstm(x) # Shape: [batch_size, seq_len, hidden_size * 2]
        
        # Global Temporal Pooling (Mean and Max combined)
        avg_pool = torch.mean(out, dim=1) # [batch, hidden_size * 2]
        max_pool, _ = torch.max(out, dim=1) # [batch, hidden_size * 2]
        
        combined_pooling = torch.cat([avg_pool, max_pool], dim=1) # [batch, hidden_size * 4]
        
        # Fully connected head
        out = self.fc1(combined_pooling)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    print("Testing BiLSTM Model with Pooling...")
    model = ISLSequenceClassifier(input_size=165, hidden_size=128, num_layers=1, num_classes=8)
    mock_input = torch.randn(4, 30, 165)
    output = model(mock_input)
    print(f"Output Shape: {output.shape} (Should be [4, 8])")