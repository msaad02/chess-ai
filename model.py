"""
Stores model definition so we can easily save/load 
"""
import torch

class ImitationModel(torch.nn.Module):
    """Simple MLP for chess imitation model"""

    def __init__(self, in_features, out_features):
        super(ImitationModel, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features, 20_000),
            torch.nn.ReLU(),
            torch.nn.Linear(20_000, 10_000),
            torch.nn.ReLU(),
            torch.nn.Linear(10_000, 5_000),
            torch.nn.ReLU(),
            torch.nn.Linear(5_000, out_features),
        )

    def forward(self, x):
        """Forward pass on model"""
        x = self.seq(x)
        return x
