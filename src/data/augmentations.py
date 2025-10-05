import torch
import numpy as np

def augment_brep_data(data, 
                      feature_noise_std=0.05,
                      feature_dropout_prob=0.1,
                      feature_scale_range=(0.9, 1.1)):
    """
    Применяет СТОХАСТИЧЕСКИЕ аугментации к признакам BRep.
    
    ВАЖНО: Каждый вызов создаёт РАЗНЫЕ случайные изменения!
    """
    # Создаем новый словарь и клонируем тензоры
    augmented_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            augmented_data[k] = v.clone()
        else:
            augmented_data[k] = v

    # 1. Аугментация вершин (минимальная)
    if 'vertices' in augmented_data:
        vertices = augmented_data['vertices']
        if feature_noise_std > 0:
            # Небольшой jitter для вершин
            jitter = torch.randn_like(vertices) * feature_noise_std * 0.05
            vertices = vertices + jitter
            augmented_data['vertices'] = vertices

    # 2. Аугментация признаков рёбер
    if 'edges' in augmented_data:
        edges = augmented_data['edges']
        
        # 2.1 Gaussian noise
        if feature_noise_std > 0:
            edge_noise = torch.randn_like(edges) * feature_noise_std
            edges = edges + edge_noise
        
        # 2.2 Feature dropout (зануление случайных признаков)
        if feature_dropout_prob > 0:
            edge_mask = (torch.rand_like(edges) > feature_dropout_prob).float()
            edges = edges * edge_mask
        
        # 2.3 Масштабирование
        if feature_scale_range is not None:
            scale = np.random.uniform(feature_scale_range[0], feature_scale_range[1])
            edges = edges * scale
        
        # Важно: сохраняем изменённые рёбра
        augmented_data['edges'] = edges

    # 3. Аугментация признаков граней
    if 'faces' in augmented_data:
        faces = augmented_data['faces']
        
        # 3.1 Gaussian noise
        if feature_noise_std > 0:
            face_noise = torch.randn_like(faces) * feature_noise_std
            faces = faces + face_noise
        
        # 3.2 Feature dropout
        if feature_dropout_prob > 0:
            face_mask = (torch.rand_like(faces) > feature_dropout_prob).float()
            faces = faces * face_mask
        
        # 3.3 Масштабирование
        if feature_scale_range is not None:
            scale = np.random.uniform(feature_scale_range[0], feature_scale_range[1])
            faces = faces * scale
        
        # Важно: сохраняем изменённые грани
        augmented_data['faces'] = faces

    return augmented_data