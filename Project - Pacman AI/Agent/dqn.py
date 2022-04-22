import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()

        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9 * 12 * 32, 256),  # third and forth value of the tensor.size() after layer 3
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 4),
            nn.ReLU()
        )

    def forward(self, tensor):
        tensor = self.layer_1(tensor)
        tensor = self.layer_2(tensor)
        tensor = self.layer_3(tensor)
        # tensor = tensor.reshape(tensor.size(0), -1)
        tensor = torch.flatten(tensor, 1)
        tensor = self.fc1(tensor)
        return self.fc2(tensor)
