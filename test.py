import cv2
import torch
import numpy as np
from model import BiEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiEncoder().to(device)
model.load_state_dict(torch.load("model.pth"))  # if saved during training
model.eval()

# Choose two images from the same scene
img1_path = 'images/images/00001.png'
img2_path = 'images/images/00002.png'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    raise ValueError("No descriptors found in one of the images.")

# Convert to tensors
des1_tensor = torch.tensor(des1, dtype=torch.float32).to(device)
des2_tensor = torch.tensor(des2, dtype=torch.float32).to(device)

# Encode descriptors
with torch.no_grad():
    z1 = model.encoder(des1_tensor)
    z2 = model.encoder(des2_tensor)

# Compute cosine similarity
sim_matrix = torch.matmul(z1, z2.T)  # shape [N1, N2]

# Get top-1 match for each descriptor in img1
top2_matches = torch.topk(sim_matrix, k=2, dim=1).indices.cpu().numpy()

# Build cv2 DMatch objects for visualization
matches = [cv2.DMatch(_queryIdx=i, _trainIdx=top2_matches[i][0], _distance=1.0 - sim_matrix[i, top2_matches[i][0]].item()) for i in range(len(top2_matches))]

# Draw matches
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
cv2.imwrite("matched_output.png", matched_img)
print("Top matches saved to matched_output.png")
