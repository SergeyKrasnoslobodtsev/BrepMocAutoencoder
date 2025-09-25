from pathlib import Path
import numpy as np
import typer
from loguru import logger
from ..searcher import search_top_k, search_top_k_cosine
from ..utils.file_utils import get_files

from ..config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

def load_embedding_from_npz(p: Path) -> np.ndarray:
        with np.load(p) as data:
            return data["embedding"]

# команда для запуска python -m src.pipelines.inference
@app.command()
def main(
    embedding_dir: Path = typer.Option(PROCESSED_DATA_DIR / 'features' / 'embeddings', help="Путь к папке с признаками"),
    query_model_stem: str = typer.Option("42. Ejector-01.prt", help="Имя модели для поиска (без расширения). Если пусто, берется 15-й файл в папке."),
    top_k: int = typer.Option(30, help="Количество возвращаемых результатов."),
    metric: str = typer.Option("cosine", help="Метрика для поиска: 'cosine' или 'euclidean'."),
    on_save: bool = typer.Option(True, help="Сохранять результаты в reports/query_model_stem.csv"),
):
    emb_files = get_files(embedding_dir, ('npz',))
    logger.info(f"Найдено {len(emb_files)} файлов")

    logger.info(f"Запрос: {query_model_stem}")
    if metric == "cosine":
        top_results_df = search_top_k_cosine(
                    embeddings_dir=embedding_dir,
                    query_stem=query_model_stem,
                    top_k=top_k,
                )
    elif metric == "euclidean":
        top_results_df = search_top_k(
                    embeddings_dir=embedding_dir,
                    query_stem=query_model_stem,
                    top_k=top_k,
                )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(top_results_df.to_string())
    if on_save:
        top_results_df.to_excel(excel_writer=REPORTS_DIR / f"{query_model_stem}.xlsx", index=False)

if __name__ == "__main__":
    app()
