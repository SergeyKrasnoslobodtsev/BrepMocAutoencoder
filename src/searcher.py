import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2 нормализация векторов для корректного расчета косинусного сходства."""
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def _load_embedding_from_npz(p: Path, key: str = "embedding") -> np.ndarray:
    """Загружает эмбеддинг из .npz файла."""
    try:
        with np.load(p) as data:
            E = data[key]
            # Гарантируем, что эмбеддинг двумерный [кол-во_граней, размерность]
            return E if E.ndim == 2 else E[None, :]
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл эмбеддинга не найден: {p}")
    except KeyError:
        raise KeyError(f"Ключ '{key}' не найден в файле: {p}")

def _calculate_distance_score(Q: np.ndarray, T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Вычисляет "асимметричное" расстояние между двумя наборами векторов Q и T.
    Для каждой грани в Q ищется ближайшая грань в T, и расстояния суммируются.

    :param Q: Матрица эмбеддингов запроса [Fq, D].
    :param T: Матрица эмбеддингов цели [Ft, D].
    :return: Кортеж (общий score, массив минимальных расстояний, массив индексов ближайших граней в T).
    """
    if Q.size == 0 or T.size == 0:
        return float('inf'), np.array([]), np.array([])

    # Вычисление попарных квадратов евклидовых расстояний
    # (a-b)^2 = a^2 - 2ab + b^2
    Q_sq = np.sum(Q**2, axis=1, keepdims=True)
    T_sq = np.sum(T**2, axis=1, keepdims=True)
    QT = Q @ T.T
    
    dist_sq = Q_sq - 2 * QT + T_sq.T  # [Fq, Ft]
    dist_sq = np.maximum(dist_sq, 0) # Избегаем отрицательных значений из-за ошибок округления

    # Для каждой грани в Q находим индекс и значение минимального расстояния до граней в T
    indices_min_dist = np.argmin(dist_sq, axis=1)      # [Fq]
    min_distances = np.sqrt(dist_sq[np.arange(Q.shape[0]), indices_min_dist]) # [Fq]
    
    # Итоговый score - сумма минимальных расстояний
    score = float(min_distances.sum())
    
    return score, min_distances, indices_min_dist

def search_top_k(
    embeddings_dir: Path, 
    query_stem: str, 
    top_k: int = 20, 
    embedding_key: str = "embedding"
) -> pd.DataFrame:
    """
    Выполняет поиск top-k похожих моделей для заданной модели-запроса.
    Поисковая модель всегда первая с score=0.
    """
    gallery_paths = sorted(list(embeddings_dir.glob("*.npz")))
    if not gallery_paths:
        print(f"В директории {embeddings_dir} не найдено .npz файлов.")
        return pd.DataFrame(columns=["model", "score"])

    query_path = embeddings_dir / f"{query_stem}.npz"
    if not query_path.exists():
        raise FileNotFoundError(f"Файл запроса не найден: {query_path}")

    # Загружаем эмбеддинг запроса
    Q = _load_embedding_from_npz(query_path, key=embedding_key)

    results = []
    # Добавляем поисковую модель первой с score=0
    results.append({"model": query_stem, "score": 0.0})

    for target_path in gallery_paths:
        if target_path.stem == query_stem:
            continue
        T = _load_embedding_from_npz(target_path, key=embedding_key)
        score, _, _ = _calculate_distance_score(Q, T)
        results.append({"model": target_path.stem, "score": score})

    if not results:
        return pd.DataFrame(columns=["model", "score"])

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="score", ascending=True).reset_index(drop=True)
    return df_sorted.head(top_k)


def search_top_k_cosine(
    embeddings_dir: Path,
    query_stem: str,
    top_k: int = 20,
    embedding_key: str = "embedding"
) -> pd.DataFrame:
    gallery_paths = sorted(list(embeddings_dir.glob("*.npz")))
    if not gallery_paths:
        print(f"В директории {embeddings_dir} не найдено .npz файлов.")
        return pd.DataFrame(columns=["model", "cosine"])

    query_path = embeddings_dir / f"{query_stem}.npz"
    if not query_path.exists():
        raise FileNotFoundError(f"Файл запроса не найден: {query_path}")

    # Загружаем эмбеддинг запроса и нормализуем
    Q = _load_embedding_from_npz(query_path, key=embedding_key)
    Q = l2norm(Q)
    Q_global = Q.mean(axis=0)  # [D]
    Q_global = Q_global / (np.linalg.norm(Q_global) + 1e-9)

    results = []
    # Добавляем поисковую модель первой с косинусом 1.0
    results.append({"model": query_stem, "cosine": 1.0})

    for target_path in gallery_paths:
        if target_path.stem == query_stem:
            continue
        T = _load_embedding_from_npz(target_path, key=embedding_key)
        T = l2norm(T)
        T_global = T.mean(axis=0)
        T_global = T_global / (np.linalg.norm(T_global) + 1e-9)
        cosine = float(np.dot(Q_global, T_global))
        results.append({"model": target_path.stem, "cosine": cosine})

    if not results:
        return pd.DataFrame(columns=["model", "cosine"])

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="cosine", ascending=False).reset_index(drop=True)
    return df_sorted.head(top_k)