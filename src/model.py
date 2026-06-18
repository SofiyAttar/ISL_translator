import torch
import torch.nn as nn

class ISLClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=26):
        super(ISLClassifier, self).__init__()
        
        # Layer 1: Input -> 128 neurons
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128) # Stabilizes training
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2) # Generalization
        
        # Layer 2: 128 -> 128 neurons
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        # Layer 3: 128 -> 64 neurons
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        
        # Layer 4: 64 -> 32 neurons
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        
        # Output Layer: 32 -> 26 (Classes)
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Flow through Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Flow through Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Flow through Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        
        # Flow through Layer 4
        x = self.fc4(x)
        x = self.relu4(x)
        
        # Final Output (Logits)
        x = self.fc5(x)
        return x

# Optional: Function to count parameters (Useful for System Design)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)