import numpy as np
from pathlib import Path
from typing import List, Dict, Callable
from tqdm.auto import tqdm
import pandas as pd

def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2 нормализация векторов."""
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Косинусное сходство между векторами."""
    if a.size == 0 or b.size == 0:
        return np.array([])
    return a @ b.T

def recall_at_k(ranks: List[int], K: int) -> float:
    """Recall@K метрика."""
    return np.mean([1.0 if r <= K else 0.0 for r in ranks]) if ranks else 0.0

def mean_average_precision(ranks: List[int]) -> float:
    """Mean Average Precision."""
    ap = [1.0 / r for r in ranks]
    return float(np.mean(ap)) if ap else 0.0

def ndcg_at_k(ranks: List[int], K: int) -> float:
    """Normalized Discounted Cumulative Gain@K."""
    vals = [1.0 / np.log2(r + 1.0) if r <= K else 0.0 for r in ranks]
    return float(np.mean(vals)) if vals else 0.0

def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Cohen's d effect size."""
    m1, m2 = np.mean(pos), np.mean(neg)
    s1, s2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return 0.0
    s_p = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (m1 - m2) / (s_p + 1e-9)

def eval_object_max(files: List[Path], load_fn: Callable):
    """
    Оценка метрик поиска (один эмбеддинг на файл).
    
    Args:
        files: Список путей к файлам эмбеддингов
        load_fn: Функция загрузки эмбеддинга из файла
    """
    N = len(files)
    if N == 0:
        print("Warning: No embedding files found.")
        return pd.DataFrame()

    print(f"Загрузка {N} эмбеддингов...")
    
    # Загружаем все эмбеддинги
    embeddings = []
    for p in tqdm(files, desc="Loading"):
        emb = load_fn(p)
        if emb.ndim > 1:
            emb = emb.flatten()
        embeddings.append(emb)
    
    # Конвертируем в numpy массив и нормализуем
    embeddings_matrix = np.stack(embeddings, axis=0)  # [N, D]
    embeddings_matrix = l2norm(embeddings_matrix, axis=1)  # Нормализуем каждую строку
    
    print(f"Матрица эмбеддингов: {embeddings_matrix.shape}")
    
    # Вычисляем матрицу сходства (батчами, чтобы не перегружать память)
    batch_size = 32
    ranks = []
    pos_vals = []
    neg_vals = []
    
    print("Вычисление метрик...")
    for i in tqdm(range(0, N, batch_size), desc="Processing batches"):
        end_idx = min(i + batch_size, N)
        batch_queries = embeddings_matrix[i:end_idx]  # [batch, D]
        
        # Вычисляем сходство батча со всеми эмбеддингами
        sim_matrix = batch_queries @ embeddings_matrix.T  # [batch, N]
        
        # Обрабатываем каждый запрос в батче
        for local_idx in range(sim_matrix.shape[0]):
            global_idx = i + local_idx
            scores = sim_matrix[local_idx]  # [N]
            
            # Позитивное значение (сам с собой)
            pos_vals.append(float(scores[global_idx]))
            
            # Негативные значения (все остальные)
            neg_scores = np.concatenate([scores[:global_idx], scores[global_idx+1:]])
            neg_vals.extend(neg_scores.tolist())
            
            # Находим ранг правильного ответа
            # Сортируем индексы по убыванию скора
            sorted_indices = np.argsort(-scores)
            rank = int(np.where(sorted_indices == global_idx)[0][0]) + 1
            ranks.append(rank)
    
    print(f"Обработано {len(ranks)} запросов")
    
    # Вычисляем финальные метрики
    metrics = pd.DataFrame([{
        "queries": N,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mAP": mean_average_precision(ranks),
        "nDCG@5": ndcg_at_k(ranks, 5),
        "nDCG@10": ndcg_at_k(ranks, 10),
        "pos_mean": float(np.mean(pos_vals)),
        "neg_mean": float(np.mean(neg_vals)),
        "margin": float(np.mean(pos_vals) - np.mean(neg_vals)),
        "cohens_d": cohens_d(np.array(pos_vals), np.array(neg_vals)),
    }])
    
    return metrics