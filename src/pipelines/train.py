from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
import typer

from ..dataset import BrepNetDataset

from ..model.brep_autoencoder import BRepAutoEncoderModule

from ..config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

def collate_single(samples):
    assert len(samples) == 1, "Этот автоэнкодер работает с batch_size=1"
    return samples[0]

# команда для запуска python -m src.pipelines.train
@app.command()
def main(
    brepnet_dir: Path = typer.Option(PROCESSED_DATA_DIR / 'features' / 'brep', help="Путь к папке с признаками B-repNet (*.npz)."),
    stats_file: Path = typer.Option(PROCESSED_DATA_DIR / 'dataset_stats.json', help="Путь к выходному JSON файлу набора данных."),
    num_workers: int = typer.Option(0, help="Количество потоков для обработки"),
    max_epochs: int = typer.Option(100, help="Максимальное количество эпох для обучения."),
    gradient_clip_val: float = typer.Option(1.0, help="Максимальная норма градиента для обрезки."),
    # Параметры модели
    n_layers: int = typer.Option(2, help="Количество слоев в модели."),
    use_attention: bool = typer.Option(True, help="Использовать ли внимание в модели."),
    lr: float = typer.Option(1e-4, help="Начальная скорость обучения."),
    tau: float = typer.Option(0.08, help="Температура для контрастивной потери."),
    m: float = typer.Option(0.999, help="Момент для обновления целевой сети."),
    queue_size: int = typer.Option(16384, help="Размер очереди для контрастивной потери."),
    w_rec: int = typer.Option(1, help="Вес для рекурссивной потери."),
    w_con: int = typer.Option(1, help="Вес для контрастивной потери."),
    points_per_face: int = typer.Option(500, help="Количество точек на грань для SDF."),
):
    module = BRepAutoEncoderModule(
            n_layers=n_layers,
            use_attention=use_attention,
            lr=lr,
            tau=tau,
            m=m,
            queue_size=queue_size,
            w_rec=w_rec,
            w_con=w_con,
            points_per_face=points_per_face
        )

    train_dataset = BrepNetDataset(stats_file, brepnet_dir, split="training_set")
    val_dataset = BrepNetDataset(stats_file, brepnet_dir, split="validation_set")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  collate_fn=collate_single,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, collate_fn=collate_single,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    csv_logger = CSVLogger(save_dir=REPORTS_DIR, name="ssl_autoencoder_logs")
    trainer = Trainer(
        max_epochs=max_epochs, 
        logger=[csv_logger],
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        detect_anomaly=True, 
    )
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    app()
