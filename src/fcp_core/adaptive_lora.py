"""
AdaLoRA - Адаптивный LoRA слой с динамическим рангом

Реализация из "Последовательные решения.txt":
- Адаптивный ранг (4/8/16) для разных слоёв
- Динамическое изменение rank во время инференса
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaLoRALayer(nn.Module):
    """
    AdaLoRA - Adaptive LoRA слой с динамическим рангом.
    
    Особенности:
    - Инициализация с max_rank (например 32)
    - Динамическое изменение rank через adapt_rank()
    - Диагональная матрица для масштабирования
    """
    
    def __init__(
        self,
        hidden_dim: int,
        init_rank: int = 8,
        max_rank: int = 32,
        scaling: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_rank = max_rank
        self.scaling = scaling
        
        # LoRA matrices: P и Q
        self.P = nn.Parameter(torch.randn(hidden_dim, max_rank) * 0.02)
        self.Q = nn.Parameter(torch.randn(max_rank, hidden_dim) * 0.02)
        
        # Диагональная матрица для масштабирования
        self.diag = nn.Parameter(torch.ones(max_rank))
        
        # Текущий rank
        self.rank = init_rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass с адаптивным рангом.
        
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            x + scaling * (x @ Q[:rank] @ diag[:rank] @ P[:rank].T)
        """
        if self.rank == 0:
            return x
        
        # Обрезаем до текущего ранга
        eff_diag = self.diag[:self.rank]
        
        # Apply LoRA: h + scale * (h @ Q[:rank] @ diag[:rank] @ P[:rank].T)
        # x @ Q.T @ diag @ P.T
        lora_update = x @ self.Q[:self.rank].T @ torch.diag(eff_diag) @ self.P[:, :self.rank].T
        
        return x + self.scaling * lora_update
    
    def adapt_rank(self, new_rank: int):
        """
        Динамическое изменение rank.
        
        Args:
            new_rank: новый ранг (0 < rank <= max_rank)
        """
        self.rank = min(new_rank, self.max_rank)
    
    def get_rank(self) -> int:
        """Текущий rank."""
        return self.rank
    
    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, rank={self.rank}, max_rank={self.max_rank}"


class AdaLoRALinear(nn.Module):
    """
    AdaLoRA обёртка для nn.Linear.
    
    Применяется к linear слоям модели.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_rank: int = 8,
        max_rank: int = 32,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        
        # Базовый linear слой
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # AdaLoRA
        self.lora = AdaLoRALayer(
            hidden_dim=in_features,
            init_rank=init_rank,
            max_rank=max_rank
        )
        
        # LoRA применён (False = только base, True = base + LoRA)
        self.lora_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        
        if not self.lora_enabled:
            return base_out
        
        # Apply LoRA к входу
        lora_out = self.lora(x)
        
        # Добавляем к base выходу (нужно proj для match размеров)
        if lora_out.shape != base_out.shape:
            # Resize через linear proj
            proj = nn.Linear(
                self.lora.hidden_dim,
                self.out_features,
                device=lora_out.device,
                dtype=lora_out.dtype
            )
            lora_out = proj(lora_out)
        
        return base_out + lora_out
    
    def adapt_rank(self, new_rank: int):
        """Изменить rank."""
        self.lora.adapt_rank(new_rank)
    
    def enable_lora(self):
        """Включить LoRA."""
        self.lora_enabled = True
    
    def disable_lora(self):
        """Выключить LoRA (только base)."""
        self.lora_enabled = False


class MultiRankAdapter(nn.Module):
    """
    Мультиранговый адаптер - создаёт несколько LoRA с разными rank.
    
    Используется для:
    - r=4 для первых слоёв (facts)
    - r=8 для средних слоёв (reasoning)  
    - r=16 для последних слоёв (creative)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ranks: list = [4, 8, 16]
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ranks = ranks
        
        # Создаём адаптеры для каждого rank
        self.adapters = nn.ModuleDict()
        for i, rank in enumerate(ranks):
            self.adapters[f"r{rank}"] = AdaLoRALayer(
                hidden_dim=hidden_dim,
                init_rank=rank,
                max_rank=max(ranks)
            )
        
        # Текущий активный адаптер
        self.active_rank = ranks[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward через активный адаптер."""
        adapter = self.adapters.get(f"r{self.active_rank}")
        if adapter is None:
            return x
        return adapter(x)
    
    def set_rank(self, rank: int):
        """Установить активный rank."""
        if rank in self.ranks:
            self.active_rank = rank
    
    def get_active_rank(self) -> int:
        """Получить активный rank."""
        return self.active_rank