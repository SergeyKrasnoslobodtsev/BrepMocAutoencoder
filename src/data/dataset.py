import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

def standardize_features(feature_tensor, stats):

    means = np.array([s["mean"] for s in stats])
    sds = np.array([s["standard_deviation"] for s in stats])
    eps = 1e-7
    assert np.all(sds > eps), "Feature has zero standard deviation"
    means_x = np.expand_dims(means, axis=0)
    sds_x = np.expand_dims(sds, axis=0)
    feature_tensor_zero_mean = feature_tensor - means_x
    feature_tensor_standardized = feature_tensor_zero_mean / sds_x
    return feature_tensor_standardized.astype(np.float32)

def standarize_data(data, feature_standardization):
    data["face_features"] = standardize_features(data["face_features"], feature_standardization["face_features"])
    data["edge_features"] = standardize_features(data["edge_features"], feature_standardization["edge_features"])
    data["coedge_features"] = standardize_features(data["coedge_features"], feature_standardization["coedge_features"])
    return data

class BrepNetDataset(Dataset):
    def __init__(self, json_path, feats_brep_dir, split="training_set"):
        with open(json_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.files = stats[split]
        self.feature_standardization = stats["feature_standardization"]
        self.split = split
        self.brep_dir = feats_brep_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        brep_path = os.path.join(self.brep_dir, file_name + ".npz")

        D = np.load(brep_path, allow_pickle=True)

        data_np = {
            "vertex": D["vertex"].astype(np.float32),
            "edge_features": D["edge_features"].astype(np.float32),
            "face_features": D["face_features"].astype(np.float32),
            "coedge_features": D["coedge_features"].astype(np.float32),
        }

        # Применяем стандартизацию только для обучающей выборки
        if self.split == "training_set" and self.feature_standardization is not None:
             # standarize_data ожидает словарь с определенными ключами

             data_to_standardize = {
                 "face_features": data_np["face_features"],
                 "edge_features": data_np["edge_features"],
                 "coedge_features": data_np["coedge_features"]
             }
             standardized_data = standarize_data(data_to_standardize, self.feature_standardization)
             # Обновляем стандартизированные признаки в data_np
             data_np.update(standardized_data)

        return {
            "name": file_name,
            "vertices": torch.from_numpy(data_np["vertex"]),
            "edges": torch.from_numpy(data_np["edge_features"]),
            "faces": torch.from_numpy(data_np["face_features"]),
            "edge_to_vertex": torch.from_numpy(D["edge_to_vertex"].astype(np.int64)),
            "face_to_edge": torch.from_numpy(D["face_to_edge"].astype(np.int64)),
            "face_to_face": torch.from_numpy(D["face_to_face"].astype(np.int64)),
            "sdf_uv": torch.from_numpy(D["uv_faces"].astype(np.float32)),
            "sdf_vals": torch.from_numpy(D["sdf_faces"].astype(np.float32))
        }