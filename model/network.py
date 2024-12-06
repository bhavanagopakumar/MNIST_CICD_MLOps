import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simple CNN architecture with less than 25000 parameters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
         x = self.pool(self.relu(self.conv1(x)))
         x = self.pool(self.relu(self.conv2(x)))
         x = x.view(-1, 16 * 7 * 7)
         x = self.relu(self.fc1(x))
         return x