import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class CartPoleRL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

