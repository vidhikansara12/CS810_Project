import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    MNIST classifier. Also returns penultimate features (for later FID-like metric).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)   # 28x28
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # 14x14 after pool
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_features: bool = False):
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        if return_features:
            return logits, feat
        return logits