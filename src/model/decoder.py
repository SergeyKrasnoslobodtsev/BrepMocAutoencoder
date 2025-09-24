import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_size, hidden_dims, uv_input_dim=2, output_dim=4):
        """
        Инициализация условного декодера. 
        Args:
            latent_size (int): Размерность выходного латентного вектора энкодера.
            hidden_dims (list of int): Список размеров скрытых слоев.
            uv_input_dim (int): Размерность входных (u, v) координат. По умолчанию 2.
            output_dim (int): Размерность выходных данных. По умолчанию 4 (x, y, z, d).
        """
        super().__init__()
        self.latent_size = latent_size
        self.uv_input_dim = uv_input_dim
        self.output_dim = output_dim

        # Входные данные: латентный вектор и (u, v) координаты, размер latent_size + uv_input_dim
        input_dim = latent_size + uv_input_dim

        # Построение полносвязных слоев
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Скрытые слои используют активацию ReLU
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, uv_coords, latent_vector):
        """
        Прямой проход через декодер.
        Args:
            uv_coords (torch.Tensor): Тензор (u, v) координат формы [N, 2].
            latent_vector (torch.Tensor): Латентный вектор формы [D].
        """
        # Расширение размерности latent_vector и повторение, чтобы соответствовать первому измерению uv_coords
        latent_vector = latent_vector.unsqueeze(0).repeat(uv_coords.shape[0], 1)
    
        # Объединение латентного вектора и (u, v) координат
        x = torch.cat([latent_vector, uv_coords], dim=-1)
    
        # Пропуск через полносвязную сеть
        output = self.network(x)
        return output