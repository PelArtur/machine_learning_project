import torch
from torch.utils.data import DataLoader
from bi_encoder_dataset import BiEncoderSIFTDataset
from model import BiEncoder
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print("Loading dataset...")
full_dataset = BiEncoderSIFTDataset('images/images')
dataset = torch.utils.data.Subset(full_dataset, range(100))

subset = torch.utils.data.Subset(full_dataset, range(100))
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size
train_dataset, val_dataset = random_split(subset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Dataset size: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print("Dataset loaded.")

model = BiEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def contrastive_loss(z_a, z_b, labels, margin=1.0):
    cosine = F.cosine_similarity(z_a, z_b)
    loss = labels * (1 - cosine) + (1 - labels) * torch.clamp(cosine - margin, min=0)
    return loss.mean()

train_losses = []
val_losses = []

for epoch in range(50):
    model.train()
    total_loss = 0
    for a, b, label in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        a, b, label = a.to(device), b.to(device), label.to(device)
        za, zb = model(a, b)
        loss = contrastive_loss(za, zb, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    model.eval()
    val_loss = 0
    with torch.no_grad():
        for a, b, label in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            a, b, label = a.to(device), b.to(device), label.to(device)
            za, zb = model(a, b)
            loss = contrastive_loss(za, zb, label)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")


import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and validation Loss')
plt.grid(True)
plt.savefig("loss_plot.png")
print("Loss plot saved to loss_plot.png")

