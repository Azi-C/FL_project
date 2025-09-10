# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)   # 1 input channel (MNIST), 6 filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 filters
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 256 -> 120
        self.fc2 = nn.Linear(120, 84)          # 120 -> 84
        self.fc3 = nn.Linear(84, 10)           # 84 -> 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 6×12×12
        x = self.pool(F.relu(self.conv2(x)))  # -> 16×4×4
        x = torch.flatten(x, 1)               # flatten to (batch, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                       # logits
        return x

def create_model():
    """Factory function to create a Net instance."""
    return Net()
