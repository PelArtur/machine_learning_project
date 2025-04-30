import torch.nn as nn
import torch

print('import')

class BiEncoder(nn.Module):
    def __init__(self, input_dim=128, emb_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )

    def forward(self, desc_a, desc_b):
        z_a = self.encoder(desc_a)
        z_b = self.encoder(desc_b)
        z_a = torch.nn.functional.normalize(z_a, dim=1)
        z_b = torch.nn.functional.normalize(z_b, dim=1)
        return z_a, z_b
