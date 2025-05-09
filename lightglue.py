import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, descriptor_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, descriptor_dim)
        )
    
    def forward(self, kpts):
        return self.mlp(kpts)

class LightGlueLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn0 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn0 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn0 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm_self0 = nn.LayerNorm(embed_dim)
        self.norm_self1 = nn.LayerNorm(embed_dim)
        self.norm_cross0 = nn.LayerNorm(embed_dim)
        self.norm_cross1 = nn.LayerNorm(embed_dim)
        self.norm_ffn0 = nn.LayerNorm(embed_dim)
        self.norm_ffn1 = nn.LayerNorm(embed_dim)
    
    def forward(self, f0, f1, mask0=None, mask1=None):
        # Self-attention
        attn0_out, _ = self.self_attn0(f0, f0, f0, key_padding_mask=~mask0)
        f0 = self.norm_self0(f0 + attn0_out)
        attn1_out, _ = self.self_attn1(f1, f1, f1, key_padding_mask=~mask1)
        f1 = self.norm_self1(f1 + attn1_out)
        
        # Cross-attention
        cross0_out, _ = self.cross_attn0(f0, f1, f1, key_padding_mask=~mask1)
        f0 = self.norm_cross0(f0 + cross0_out)
        cross1_out, _ = self.cross_attn1(f1, f0, f0, key_padding_mask=~mask0)
        f1 = self.norm_cross1(f1 + cross1_out)
        
        # FFN
        f0 = self.norm_ffn0(f0 + self.ffn0(f0))
        f1 = self.norm_ffn1(f1 + self.ffn1(f1))
        
        return f0, f1

class LightGlue(nn.Module):
    def __init__(self, descriptor_dim=32, num_layers=9, num_heads=4):
        super().__init__()
        self.pos_encoder = PositionalEncoder(descriptor_dim)
        self.layers = nn.ModuleList([LightGlueLayer(descriptor_dim, num_heads) for _ in range(num_layers)])
        self.final_proj = nn.Linear(descriptor_dim, descriptor_dim)
    
    def forward(self, desc0, desc1, kpts0, kpts1):
        mask0 = (desc0.abs().sum(dim=-1) > 0)
        mask1 = (desc1.abs().sum(dim=-1) > 0)
        
        pos0 = self.pos_encoder(kpts0)
        pos1 = self.pos_encoder(kpts1)
        f0 = desc0 + pos0
        f1 = desc1 + pos1
        
        for layer in self.layers:
            f0, f1 = layer(f0, f1, mask0, mask1)
        
        f0 = self.final_proj(f0)
        f1 = self.final_proj(f1)
        
        S = torch.bmm(f0, f1.transpose(1, 2))
        P = (F.softmax(S, dim=2) * F.softmax(S, dim=1)).sqrt()
        return P
