import torch
import cv2
import numpy as np
from poly_encoder import PolyEncoder
from dataset import detect_keypoints_and_descriptors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

desc_dim = 32
model_path = "/home/ironiss/Documents/ML-project/machine_learning_project/best_poly_encoder.pth"
img1_path = "/home/ironiss/Documents/ML-project/machine_learning_project/st_peters_square/dense/images/99874023_2214009602.jpg"
img2_path = "/home/ironiss/Documents/ML-project/machine_learning_project/st_peters_square/dense/images/99874023_2413009006.jpg"
detector_type = "ORB"
threshold = 0.35


model = PolyEncoder(
    input_dim=2 + desc_dim,
    hidden_dim=256,
    num_heads=8,
    dropout=0.1
).to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()


img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

data1 = detect_keypoints_and_descriptors(img1, detector_type=detector_type, n_features=512)
data2 = detect_keypoints_and_descriptors(img2, detector_type=detector_type, n_features=512)

keypoints1 = np.array([kp.pt for kp in data1["keypoints"]], dtype=np.float32)
keypoints2 = np.array([kp.pt for kp in data2["keypoints"]], dtype=np.float32)

descriptors1 = data1["descriptors"]
descriptors2 = data2["descriptors"]

feat1 = torch.tensor(np.hstack([keypoints1, descriptors1]), dtype=torch.float32).unsqueeze(0).to(device)
feat2 = torch.tensor(np.hstack([keypoints2, descriptors2]), dtype=torch.float32).unsqueeze(0).to(device)


with torch.no_grad():
    emb1, emb2 = model(feat1, feat2)

similarity = torch.bmm(emb1, emb2.transpose(1, 2))[0] 
pred_matches = torch.argmax(similarity, dim=1)         
confidences = similarity[torch.arange(similarity.size(0)), pred_matches]


kp1_cv = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints1]
kp2_cv = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints2]

fwd_match = pred_matches
rev_match = torch.argmax(similarity.transpose(0, 1), dim=1)

filtered_matches = [
    cv2.DMatch(_queryIdx=int(i), _trainIdx=int(fwd_match[i]), _distance=float(confidences[i]))
    for i in range(len(fwd_match))
    if rev_match[fwd_match[i]] == i and confidences[i] > threshold
]

k = 100
topk_indices = torch.topk(confidences, k=k).indices

filtered_matches = [
    cv2.DMatch(_queryIdx=int(i), _trainIdx=int(pred_matches[i]), _distance=float(confidences[i]))
    for i in topk_indices
]



matched_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, filtered_matches, None)
cv2.imwrite("test.png", matched_img)
print("Match image saved to cross_attention_matches.png")
