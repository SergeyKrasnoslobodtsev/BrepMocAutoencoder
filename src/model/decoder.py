from typing import List
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
    


class GeometryAwareDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], 
                 use_positional_encoding: bool = True,
                 use_geometry_regularization: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_pos_enc = use_positional_encoding
        self.use_geo_reg = use_geometry_regularization
        
        # Positional encoding для UV координат
        if use_positional_encoding:
            self.pos_enc_dim = 20  # L=10 частот
            uv_input_dim = 2 * self.pos_enc_dim  # sin + cos
        else:
            uv_input_dim = 2
        
        input_dim = latent_dim + uv_input_dim
        
        # Основная сеть
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Стабилизация обучения
                nn.ReLU(inplace=True),
                nn.Dropout(0.1) if i < len(hidden_dims) - 1 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Выходной слой (xyz + sdf)
        layers.append(nn.Linear(prev_dim, 4))
        
        self.main_network = nn.Sequential(*layers)
        
        # Отдельная голова для SDF с tanh активацией для стабильности
        self.sdf_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Tanh()  # Ограничиваем выходные значения
        )
        
        # Голова для XYZ координат
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[-1] // 2, 3)
        )

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Позиционное кодирование для UV координат."""
        # coords: [N, 2]
        frequencies = torch.arange(10, device=coords.device, dtype=coords.dtype)
        frequencies = 2.0 ** frequencies  # [10]
        
        # Расширяем размерности: [N, 2, 10]
        coords_freq = coords.unsqueeze(-1) * frequencies.unsqueeze(0).unsqueeze(0)
        
        # Применяем sin и cos
        sin_enc = torch.sin(coords_freq)  # [N, 2, 10]
        cos_enc = torch.cos(coords_freq)  # [N, 2, 10]
        
        # Конкатенируем и flatten
        pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)  # [N, 2, 20]
        return pos_enc.reshape(coords.shape[0], -1)  # [N, 40]

    def forward(self, uv_coords: torch.Tensor, 
                latent_vector: torch.Tensor,
                return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            uv_coords: [N, 2] UV координаты
            latent_vector: [D] или [1, D] латентный вектор
            return_features: возвращать промежуточные признаки
        """
        batch_size = uv_coords.shape[0]
        
        # Расширяем latent_vector
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)
        latent_expanded = latent_vector.repeat(batch_size, 1)
        
        # Позиционное кодирование UV
        if self.use_pos_enc:
            uv_encoded = self.positional_encoding(uv_coords)
        else:
            uv_encoded = uv_coords
        
        # Конкатенируем латентный вектор и UV
        combined_input = torch.cat([latent_expanded, uv_encoded], dim=-1)
        
        # Прогоняем через основную сеть (до последнего слоя)
        x = combined_input
        for layer in list(self.main_network.children())[:-1]: 
            x = layer(x)
        
        # Отдельные головы для xyz и sdf
        xyz_pred = self.xyz_head(x)  # [N, 3]
        sdf_pred = self.sdf_head(x).squeeze(-1)  # [N]
        # Объединяем предсказания
        output = torch.cat([xyz_pred, sdf_pred.unsqueeze(-1)], dim=-1)
        
        if return_features:
            return output, x
        return output

    def compute_eikonal_loss(self, coords: torch.Tensor, 
                           latent_vector: torch.Tensor) -> torch.Tensor:
        """Eikonal loss для соблюдения SDF свойств."""
        coords.requires_grad_(True)
        
        # Прямой проход
        output, _ = self.forward(coords, latent_vector, return_features=False)
        sdf = output[:, 3]  # SDF значения
        
        # Вычисляем градиенты
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Эикональная потеря: сигнал градиента должен быть 1
        gradient_norm = torch.norm(gradients, dim=-1)
        eikonal_loss = F.mse_loss(gradient_norm, torch.ones_like(gradient_norm))
        
        return eikonal_loss

    def compute_surface_loss(self, coords: torch.Tensor, 
                           latent_vector: torch.Tensor,
                           surface_points: torch.Tensor) -> torch.Tensor:
        """Loss для точек на поверхности (SDF = 0)."""
        if surface_points.shape[0] == 0:
            return torch.tensor(0.0, device=coords.device)

        surface_output, _ = self.forward(surface_points, latent_vector, return_features=False)
        surface_sdf = surface_output[:, 3]
        
        # На поверхности SDF должно быть 0
        surface_loss = F.mse_loss(surface_sdf, torch.zeros_like(surface_sdf))
        
        return surface_loss