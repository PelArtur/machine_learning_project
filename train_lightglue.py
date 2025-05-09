import torch
import torch.nn as nn
import torch.optim as optim
from lightGlue import LightGlue
from dataset_utils import *
from dataset import *
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional


from sklearn.model_selection import KFold

def kfold_building_split(buildings, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return list(kf.split(buildings))


def create_keypoint_features(batch_data: torch.Tensor, desc_dim: int):
    kp1 = batch_data[:, :, :2]  
    desc1 = batch_data[:, :, 2:2+desc_dim]  
    
    kp2 = batch_data[:, :, 2+desc_dim:4+desc_dim]  
    desc2 = batch_data[:, :, 4+desc_dim:]  
    
    img1_features = torch.cat([kp1, desc1], dim=2)  
    img2_features = torch.cat([kp2, desc2], dim=2) 
    
    return img1_features, img2_features


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, img1_emb: torch.Tensor, img2_emb: torch.Tensor, matches: torch.Tensor) -> torch.Tensor:
        batch_size, n_keypoints, emb_dim = img1_emb.shape
        
       
        img1_emb_exp = img1_emb.unsqueeze(2) 
        img2_emb_exp = img2_emb.unsqueeze(1) 

        distances = torch.sqrt(torch.sum((img1_emb_exp - img2_emb_exp) ** 2, dim=3) + 1e-6)
        
        valid_mask = (matches != -1).float()  
        
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n_keypoints).to(matches.device)
        kp1_indices = torch.arange(n_keypoints).unsqueeze(0).expand(batch_size, -1).to(matches.device)
        
        matched_distances = torch.zeros_like(distances)
        
        valid_indices = valid_mask.bool()
        if valid_indices.sum() > 0:
            b_idx = batch_indices[valid_indices]
            kp1_idx = kp1_indices[valid_indices]
            kp2_idx = matches[valid_indices].long()
            
            pos_distances = distances[b_idx, kp1_idx, kp2_idx]
            
            matched_distances[b_idx, kp1_idx, kp2_idx] = pos_distances
        
        pos_mask = torch.zeros_like(distances)
        for b in range(batch_size):
            for i in range(n_keypoints):
                if matches[b, i] >= 0:
                    pos_mask[b, i, matches[b, i].long()] = 1.0
        
        pos_loss = (matched_distances * pos_mask).sum() / (pos_mask.sum() + 1e-6)
        
        neg_mask = 1.0 - pos_mask
        
        neg_loss = torch.clamp(self.margin - distances, min=0.0)
        neg_loss = (neg_loss * neg_mask).sum() / (neg_mask.sum() + 1e-6)
        
        return pos_loss + neg_loss


def train(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: Optional[DataLoader] = None,
          desc_dim: int = 32, 
          lr: float = 1e-3,
          num_epochs: int = 10, 
          device: str = 'cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = ContrastiveLoss(margin=0.2).to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            batch_data = batch_data.float().to(device)
            batch_labels = batch_labels.to(device)
            
            img1_features, img2_features = create_keypoint_features(batch_data, desc_dim)
            
            img1_emb, img2_emb = model(img1_features, img2_features)
            
            loss = criterion(img1_emb, img2_emb, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_data, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    batch_data = batch_data.float().to(device)
                    batch_labels = batch_labels.to(device)
                    
                    img1_features, img2_features = create_keypoint_features(batch_data, desc_dim)
                    
                    img1_emb, img2_emb = model(img1_features, img2_features)
                    
                    loss = criterion(img1_emb, img2_emb, batch_labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_poly_encoder.pth")
                
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
        scheduler.step()
    
    if val_loader is None:
        torch.save(model.state_dict(), "final_poly_encoder.pth")
    
    return model


def test_model(model: nn.Module, 
               test_loader: DataLoader, 
               desc_dim: int = 32,
               device: str = 'cuda', 
               distance_threshold: float = 0.1):

    model = model.to(device)
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_data = batch_data.float().to(device)
            batch_labels = batch_labels.to(device)
            
            img1_features, img2_features = create_keypoint_features(batch_data, desc_dim)
            
            img1_emb, img2_emb = model(img1_features, img2_features)
            
            similarity = torch.bmm(img1_emb, img2_emb.transpose(1, 2))
            
            pred_matches = torch.argmax(similarity, dim=2) 
            
            confidence = torch.gather(similarity, 2, pred_matches.unsqueeze(2)).squeeze(2)
           
            valid_pred = confidence > distance_threshold
            
            for b in range(batch_data.size(0)):
                gt_matches = batch_labels[b]  
                pred = pred_matches[b] 
                valid = valid_pred[b]  
                
                gt_valid = gt_matches >= 0
                
                for i in range(len(gt_matches)):
                    if gt_valid[i]:
                        if valid[i] and pred[i] == gt_matches[i]:
                            total_tp += 1
                        else:
                            total_fn += 1
                    elif valid[i]:
                        total_fp += 1
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    return precision, recall, f1_score

def cross_validate_model(all_buildings, detector_type='ORB', n_splits=4):
    kf_splits = kfold_building_split(all_buildings, k=n_splits)
    desc_dim = 32 if detector_type == 'ORB' else 128
    n_keypoints = 512
    best_f1 = 0.0

    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf_splits):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        train_buildings = [all_buildings[i] for i in train_idx]
        val_buildings = [all_buildings[i] for i in val_idx]

        train_dataset = MatchingDataset(
            datasets=train_buildings,
            detector_type=detector_type,
            n_keypoints=n_keypoints,
            max_pairs_per_dataset=1000
        )

        val_dataset = MatchingDataset(
            datasets=val_buildings,
            detector_type=detector_type,
            n_keypoints=n_keypoints,
            max_pairs_per_dataset=200
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        input_dim = 2 +desc_dim
        
        model = LightGlue(
            input_dim=input_dim,
            hidden_dim=256,
            gnn_layers=4
        )


        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            desc_dim=desc_dim,
            lr=1e-3,
            num_epochs=10,
            device=device
        )

        precision, recall, f1 = test_model(
            model=model,
            test_loader=val_loader, 
            desc_dim=desc_dim,
            device=device
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_lightglue_encoder.pth")

        scores.append((precision, recall, f1))

    avg_scores = np.mean(scores, axis=0)
    print(f"\n=== Average over {n_splits} folds ===")
    print(f"Precision: {avg_scores[0]:.4f}, Recall: {avg_scores[1]:.4f}, F1: {avg_scores[2]:.4f}")


def main():
    all_buildings = ["brandenburg_gate", "buckingham_palace", "sacre_coeur", "st_peters_square"]
    cross_validate_model(all_buildings, detector_type='ORB', n_splits=4)

if __name__ == "__main__":
    main()