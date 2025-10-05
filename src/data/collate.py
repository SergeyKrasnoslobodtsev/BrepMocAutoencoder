from torch_geometric.data import Batch, Data
import torch
from .augmentations import augment_brep_data


def moco_collate_fn(batch):
    """
    Функция коллации для DataLoader MoCo.
    """
    data_list_q = []
    data_list_k = []

    for idx, data_item in enumerate(batch):
        augmented_q = augment_brep_data(data_item)
        augmented_k = augment_brep_data(data_item,
                                        feature_noise_std=0.1,
                                        feature_dropout_prob=0.3,
                                        feature_scale_range=(0.7, 1.3))

        data_q = Data(
            x=augmented_q['vertices'],
            edge_attr=augmented_q['edges'],
            face_attr=augmented_q['faces'],
            edge_index=augmented_q['edge_to_vertex'],
        )

        data_k = Data(
            x=augmented_k['vertices'],
            edge_attr=augmented_k['edges'],
            face_attr=augmented_k['faces'],
            edge_index=augmented_k['edge_to_vertex'],
        )

        data_list_q.append(data_q)
        data_list_k.append(data_k)

    # Батчим только с follow_batch для face_attr
    batch_q = Batch.from_data_list(data_list_q, follow_batch=['face_attr'])
    batch_k = Batch.from_data_list(data_list_k, follow_batch=['face_attr'])

    # Вручную создаем edges_batch
    for batch_obj, data_list in [(batch_q, data_list_q), (batch_k, data_list_k)]:
        _add_edges_batch(batch_obj, data_list)
        _rename_and_add_lists(batch_obj, batch)

    return batch_q, batch_k


def simple_collate_fn(batch):
    """
    Простая функция коллации без аугментаций.
    """
    data_list = []

    for data_item in batch:
        data_obj = Data(
            x=data_item['vertices'],
            edge_attr=data_item['edges'],
            face_attr=data_item['faces'],
            edge_index=data_item['edge_to_vertex'],
        )
        data_list.append(data_obj)

    batch_data = Batch.from_data_list(data_list, follow_batch=['face_attr'])
    
    # Вручную создаем edges_batch
    _add_edges_batch(batch_data, data_list)
    _rename_and_add_lists(batch_data, batch)

    return batch_data


def _add_edges_batch(batch_obj, data_list):
    """Создает edges_batch вручную."""
    edges_batch_list = []
    for graph_idx, data in enumerate(data_list):
        num_edges = data.edge_attr.size(0)
        edges_batch_list.append(torch.full((num_edges,), graph_idx, dtype=torch.long))
    
    batch_obj.edges_batch = torch.cat(edges_batch_list)


def _rename_and_add_lists(batch_obj, original_batch):
    """Переименовывает атрибуты и добавляет списки."""
    batch_obj.vertices = batch_obj.x
    batch_obj.edges = batch_obj.edge_attr
    batch_obj.faces = batch_obj.face_attr
    batch_obj.edge_to_vertex = batch_obj.edge_index
    batch_obj.faces_batch = batch_obj.face_attr_batch
    
    # Добавляем списки из оригинального батча
    batch_obj.sdf_uv_list = [d['sdf_uv'] for d in original_batch]
    batch_obj.sdf_vals_list = [d['sdf_vals'] for d in original_batch]
    batch_obj.face_to_edge_list = [d['face_to_edge'] for d in original_batch]
    batch_obj.face_to_face_list = [d['face_to_face'] for d in original_batch]