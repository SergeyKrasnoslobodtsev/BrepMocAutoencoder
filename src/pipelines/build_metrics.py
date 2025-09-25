from pathlib import Path
import numpy as np
import typer

from ..metrics import eval_object_max

from ..config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

def load_embedding_from_npz(p: Path) -> np.ndarray:
        with np.load(p) as data:
            return data["embedding"]

# команда для запуска python -m src.pipelines.build_metrics
@app.command()
def main(
    embedding_dir: Path = typer.Option(PROCESSED_DATA_DIR / 'features' / 'embeddings', help="Путь к папке с признаками"),
    on_save: bool = typer.Option(False, help="Сохранять результаты в reports/metrics.csv"),

):
    emb_files = list(embedding_dir.glob("*.npz"))
    df = eval_object_max(emb_files, load_fn=load_embedding_from_npz)
    print(df.to_string())
    if on_save:
        df.to_excel(excel_writer=REPORTS_DIR / "metrics.xlsx", index=False)

if __name__ == "__main__":
    app()
