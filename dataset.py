import numpy as np
from torch.utils.data import Dataset
from dataset_utils import get_image, get_dataset_data, extract_fundamental, detect_keypoints_and_descriptors, match_with_fundamental

from tqdm import tqdm
from typing import List


class MatchingDataset(Dataset):
    def __init__(self, datasets: List[str], 
                       detector_type: str = 'ORB', 
                       n_keypoints: int = 512,
                       dataset_part: float = 1.0,
                       max_pairs_per_dataset: int = -1,
                       allow_padding: bool = True):
        super().__init__()
        self.data: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.desc_dim: int = 32 if detector_type == 'ORB' else 128

        zero_desc = np.zeros(self.desc_dim)
        skipped_samples = 0
        for building in datasets:
            print(f"Extracting {building} data...")
            pairs, cameras, images = get_dataset_data(building)

            np.random.shuffle(pairs)
            n = len(pairs) if max_pairs_per_dataset == -1 else max_pairs_per_dataset
            pairs = pairs[:min(n, int(dataset_part * len(pairs)))]

            for idx1, idx2 in tqdm(pairs, desc=f"Processing {building} image pairs"):
                data1 = get_image(idx1, building, images, cameras)
                data2 = get_image(idx2, building, images, cameras)

                F = extract_fundamental(data1, data2) 
                data1 = detect_keypoints_and_descriptors(data1['image'], detector_type=detector_type, n_features=n_keypoints)
                data2 = detect_keypoints_and_descriptors(data2['image'], detector_type=detector_type, n_features=n_keypoints)
                if not allow_padding and (len(data1["keypoints"]) < n_keypoints or len(data2["keypoints"]) < n_keypoints):
                    skipped_samples += 1
                    continue
        
                matches = match_with_fundamental(data1, data2, F, threshold=1.0)

                #input data
                padded = False
                sample = np.zeros((n_keypoints, 4 + 2 * self.desc_dim))  #n_keypoints x (2 keypoints1 + 2 keypoints2 + desc_dim desc1 + desc_dim desc2)
                for i in range(n_keypoints):
                    if i < len(data1["keypoints"]):
                        sample[i][0], sample[i][1] = data1["keypoints"][i].pt
                        sample[i][2:2 + self.desc_dim] = data1["descriptors"][i]
                    else:
                        padded = True
                        sample[i][0], sample[i][1] = 0.0, 0.0
                        sample[i][2:2 + self.desc_dim] = zero_desc

                    if i < len(data2["keypoints"]):
                        sample[i][2 + self.desc_dim], sample[i][3 + self.desc_dim] = data2["keypoints"][i].pt
                        sample[i][-self.desc_dim:] = data2["descriptors"][i]
                    else:
                        padded = True
                        sample[i][2 + self.desc_dim], sample[i][3 + self.desc_dim] = 0.0, 0.0
                        sample[i][-self.desc_dim:] = zero_desc

                #matches
                matches_arr = np.zeros(n_keypoints) - 1
                for match1, match2, _ in matches:
                    matches_arr[match1] = match2

                self.data.append(sample)
                self.labels.append(matches_arr)
                if padded:
                    skipped_samples += 1

        if allow_padding:
            print(f"Samples with padding: {skipped_samples}/{len(self.data)}")
        else:
            print(f"Skipped samples: {skipped_samples}/{len(self.data) + skipped_samples}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]




class MatchingLightGlueDataset(Dataset):
    def __init__(self, datasets: List[str], 
                       detector_type: str = 'ORB', 
                       n_keypoints: int = 512,
                       dataset_part: float = 1.0,
                       max_pairs_per_dataset: int = -1,
                       allow_padding: bool = True):
        super().__init__()
        self.data: List[np.ndarray] = []
        self.desc_dim: int = 32 if detector_type == 'ORB' else 128

        skipped_samples = 0
        for building in datasets:
            print(f"Extracting {building} data...")
            pairs, cameras, images = get_dataset_data(building)

            np.random.shuffle(pairs)
            n = len(pairs) if max_pairs_per_dataset == -1 else max_pairs_per_dataset
            pairs = pairs[:min(n, int(dataset_part * len(pairs)))]

            for idx1, idx2 in tqdm(pairs, desc=f"Processing {building} image pairs"):
                data1 = get_image(idx1, building, images, cameras)
                data2 = get_image(idx2, building, images, cameras)

                F = extract_fundamental(data1, data2) 
                data1 = detect_keypoints_and_descriptors(data1['image'], detector_type=detector_type, n_features=n_keypoints)
                data2 = detect_keypoints_and_descriptors(data2['image'], detector_type=detector_type, n_features=n_keypoints)
                if not allow_padding and (len(data1["keypoints"]) < n_keypoints or len(data2["keypoints"]) < n_keypoints):
                    skipped_samples += 1
                    continue

                data1_shape = data1["image"].shape
                data2_shape = data2["image"].shape
        
                matches = match_with_fundamental(data1, data2, F, threshold=1.0)

                #input data
                padded = False
                sample = {
                    "desc0": np.array(data1["descriptors"]),
                    "desc1": np.array(data2["descriptors"])
                }

                kpts0 = np.zeros((n_keypoints, 2))
                kpts1 = np.zeros((n_keypoints, 2))
                for i in range(n_keypoints):
                    kpts0[i][0], kpts0[i][1] = data1["keypoints"][i].pt
                    kpts1[i][0], kpts1[i][1] = data2["keypoints"][i].pt

                kpts0[:, 0] /= data1_shape[1]
                kpts0[:, 1] /= data1_shape[0]
                kpts1[:, 0] /= data2_shape[1]
                kpts1[:, 1] /= data2_shape[0]
                sample["kpts0"] = kpts0
                sample["kpts1"] = kpts1

                #matches
                matches_arr = []
                for match1, match2, _ in matches:
                    matches_arr.append([match1, match2])
                sample["matches"] = matches_arr
                self.data.append(sample)
                if padded:
                    skipped_samples += 1

        if allow_padding:
            print(f"Samples with padding: {skipped_samples}/{len(self.data)}")
        else:
            print(f"Skipped samples: {skipped_samples}/{len(self.data) + skipped_samples}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        return self.data[index]
