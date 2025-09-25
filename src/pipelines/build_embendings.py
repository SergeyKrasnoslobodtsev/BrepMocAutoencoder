from pathlib import Path
import numpy as np
import os
import json
import torch
import typer
from tqdm.auto import tqdm

from ..model.brep_autoencoder import BRepAutoEncoderModule
from ..utils.file_utils import get_files

from ..config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

def standardize_features(feature_tensor, stats):
    # num_features = len(stats)
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

# команда для запуска python -m src.pipelines.build_embendings
@app.command()
def main(
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR / 'features' / 'embeddings', help="Путь к папке для сохранения признаков энкодера."),
    features_dir: Path = typer.Option(PROCESSED_DATA_DIR / 'features' / 'brep', help="Путь к папке с признаками B-repNet (*.npz)."),
    model_path: Path = typer.Option(MODELS_DIR / 'best.ckpt', help="Путь к контрольной точке обученной модели."),
    stats_file: Path = typer.Option(PROCESSED_DATA_DIR / 'dataset_stats.json', help="Путь к выходному JSON файлу набора данных."),
):

    model = BRepAutoEncoderModule.load_from_checkpoint(model_path)
    model.eval()

    encoder = model.encoder.eval()
    brepnet_files = get_files(features_dir, ('npz',))

    with open(stats_file, encoding="utf-8") as f:
        stats = json.load(f)
    for file in tqdm(brepnet_files, desc="Извлечение признаков"):

        D = np.load(file, allow_pickle=True)

        f_e = torch.from_numpy(D["face_to_edge"].astype(np.int64))
        if f_e.dim() == 2 and f_e.shape[0] != 2 and f_e.shape[1] == 2:
            f_e = f_e.t()
        
        data = {
                "vertex": D["vertex"],
                "edge_features": D["edge_features"],
                "face_features": D["face_features"],
                "coedge_features": D["coedge_features"],
                "edge_to_vertex": D["edge_to_vertex"],
                "face_to_edge": f_e.contiguous(),
                "face_to_face": D["face_to_face"],
            }
        data = standarize_data(data, stats["feature_standardization"])
        tenzors = {
            "vertices": torch.from_numpy(data["vertex"].astype(np.float32)),
            "edges": torch.from_numpy(data["edge_features"].astype(np.float32)),
            "faces": torch.from_numpy(data["face_features"].astype(np.float32)),
            "edge_to_vertex": torch.from_numpy(data["edge_to_vertex"].astype(np.int64)),
            "face_to_edge": torch.from_numpy(D["face_to_edge"][::-1].astype(np.int64)).t(),   # [2, n_f]
            "face_to_face": torch.from_numpy(data["face_to_face"].astype(np.int64))
        }
        emb = encoder(tenzors)           # [n_faces, D]
        emb = torch.nan_to_num(emb)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        emb_global = emb.mean(dim=0)     # [D]
        emb_global = torch.nn.functional.normalize(emb_global, dim=0)
        emb_np = emb_global.detach().cpu().numpy().astype(np.float32)   # [D]
        path = output_dir / f"{file.stem}.npz"
        os.makedirs(path.parent, exist_ok=True)
        np.savez_compressed(path, embedding=emb_np)

if __name__ == "__main__":
    app()
