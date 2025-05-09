import torch.nn as nn
import torch

print('import')

class BiEncoder(nn.Module):
    def __init__(self, input_dim: int = 2 + 32, embed_dim: int = 128):
        super(BiEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)