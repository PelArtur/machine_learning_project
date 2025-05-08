import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PolyEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128, 
                 output_dim: int = 64,
                 num_codes: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.global_codes = nn.Parameter(torch.Tensor(num_codes, hidden_dim))
        nn.init.normal_(self.global_codes, std=0.01)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, img1_features: torch.Tensor, img2_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_keypoints, _ = img1_features.shape
        
        img1_encoded = self.feature_encoder(img1_features) 
        img2_encoded = self.feature_encoder(img2_features)

        global_codes = self.global_codes.unsqueeze(0).expand(batch_size, -1, -1) 
        
        attention_weights = torch.bmm(global_codes, img1_encoded.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=2)
       
        ctx_representations = torch.bmm(attention_weights, img1_encoded)
        
       
        img2_attention = torch.bmm(img2_encoded, ctx_representations.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        img2_attention = F.softmax(img2_attention, dim=2)
        
       
        img2_ctx = torch.bmm(img2_attention, ctx_representations)
        
        img1_emb = self.output_proj(img1_encoded)  
        img2_emb = self.output_proj(img2_encoded + img2_ctx) 
        
        return img1_emb, img2_emb