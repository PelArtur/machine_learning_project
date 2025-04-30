import cv2
import torch
import random
import os
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

print("Importing bi_encoder_dataset...")


class BiEncoderSIFTDataset(Dataset):
    def __init__(self, image_dir, num_negatives=1):
        self.pairs = []
        self.num_negatives = num_negatives
        self.image_dir = image_dir
        self.scene_groups = self.group_images(image_dir)
        self.prepare_pairs()

    def group_images(self, path):
        groups = defaultdict(list)
        for f in os.listdir(path):
            if f.endswith('.png'):
                sid = f[:3]
                groups[sid].append(os.path.join(path, f))
        return groups

    def prepare_pairs(self):
        sift = cv2.SIFT_create()
        for group in self.scene_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    img1 = cv2.imread(group[i], cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(group[j], cv2.IMREAD_GRAYSCALE)
                    kp1, des1 = sift.detectAndCompute(img1, None)
                    kp2, des2 = sift.detectAndCompute(img2, None)

                    if des1 is None or des2 is None:
                        continue

                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)

                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            pos_a = des1[m.queryIdx]
                            pos_b = des2[m.trainIdx]
                            self.pairs.append((pos_a, pos_b, 1))

                            for _ in range(self.num_negatives):
                                neg_b = des2[random.randint(0, len(des2) - 1)]
                                self.pairs.append((pos_a, neg_b, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
