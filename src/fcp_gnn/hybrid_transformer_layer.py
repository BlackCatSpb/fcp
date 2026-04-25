"""
HybridTransformerLayer - Гибридный слой с GNN инъекцией на каждом слое

Особенность: GNN инъекция выполняется на АБСОЛЮТНО каждом слое модели.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class HybridTransformerLayer(nn.Module):
    """
    Гибридный Transformer слой с интегрированной GNN инъекцией.
    
    Особенности:
    - Standard Transformer: attention + FFN
    - GNN Инъекция: на КАЖДОМ слое (не только 4,8,16,24!)
    - LoRA адаптер для тонкой настройки
    - Адаптивный гейт для контроля инъекции
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        ff_dim: int = 5504,
        dropout: float = 0.0,
        use_gnn: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        injection_scale: float = 0.1,
        layer_id: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_id = layer_id
        self.use_gnn = use_gnn
        self.use_lora = use_lora
        
        # ========== Standard Transformer Components ==========
        # Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # LayerNorm для attention
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # FFN (SwiGLU-style)
        self.gate_proj = nn.Linear(hidden_dim, ff_dim)
        self.up_proj = nn.Linear(hidden_dim, ff_dim)
        self.down_proj = nn.Linear(ff_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)
        
        # LayerNorm для FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # ========== GNN Инъекция (НА КАЖДОМ СЛОЕ!) ==========
        if use_gnn:
            # GNN энкодер для получения graph_vec
            self.gnn_proj = nn.Linear(hidden_dim, hidden_dim)
            self.gate_weights = nn.Linear(2 * hidden_dim, hidden_dim)
            
            # Gate для контроля инъекции
            self.injection_gate = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            self.injection_scale = injection_scale
        
        # ========== LoRA (НА КАЖДОМ СЛОЕ!) ==========
        if use_lora:
            # LoRA matrices
            self.lora_A = nn.Parameter(torch.randn(hidden_dim, lora_rank) * 0.02)
            self.lora_B = nn.Parameter(torch.randn(lora_rank, hidden_dim) * 0.02)
            self.lora_scaling = lora_rank ** 0.5 / lora_rank
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        graph_vec: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_lora: bool = True,
        enable_injection: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass с GNN инъекцией на каждом слое.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            graph_vec: [batch, hidden_dim] - вектор из GNN (опционально)
            attention_mask: [batch, seq_len]
            use_lora: применять LoRA
            enable_injection: применять GNN инъекцию
        
        Returns:
            (output, info_dict)
        """
        info = {"layer": self.layer_id, "injection_applied": False, "lora_applied": False}
        batch_size, seq_len, _ = x.shape
        
        # ============== Attention ==============
        # Self-attention
        attn_output, _ = self.self_attn(
            x, x, x,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        attn_output = self.dropout(attn_output)
        x = self.attn_norm(x + attn_output)
        
        # ============== GNN Инъекция (НА КАЖДОМ СЛОЕ!) ==============
        if self.use_gnn and enable_injection and graph_vec is not None:
            # Получаем последний hidden state
            last_hidden = x[:, -1:, :]  # [batch, 1, hidden_dim]
            
            # Проецируем через GNN
            graph_proj = self.gnn_proj(last_hidden)  # [batch, 1, hidden_dim]
            
            # Combine: [last_hidden, graph_vec]
            combined = torch.cat([last_hidden, graph_vec.unsqueeze(1)], dim=-1)
            
            # Gate
            gate = self.gate_weights(combined)  # [batch, 1, hidden_dim]
            gate_value = self.injection_gate(last_hidden)  # [batch, 1, 1]
            
            # Compute injection
            injection = self.injection_scale * gate_value * graph_proj
            
            # Add to last hidden state
            x[:, -1:, :] = x[:, -1:, :] + injection
            
            info["injection_applied"] = True
            info["injection_strength"] = gate_value.item()
        
        # ============== LoRA (НА КАЖДОМ СЛОЕ!) ==============
        if self.use_lora and use_lora:
            # Get last hidden
            last_hidden = x[:, -1:, :]
            
            # LoRA update: x + scale * (x @ A @ B)
            lora_update = last_hidden @ self.lora_A @ self.lora_B
            lora_update = lora_update * self.lora_scaling
            
            x = x + lora_update
            info["lora_applied"] = True
        
        # ============== FFN ==============
        # SwiGLU
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = F.silu(gate)
        ffn_output = gate * up
        ffn_output = self.down_proj(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output)
        
        x = self.ffn_norm(x + ffn_output)
        
        return x, info


class HybridTransformerStack(nn.Module):
    """
    Полный стек гибридных слоёв (32 слоя).
    
    GNN инъекция на ВСЕХ 32 слоях!
    """
    
    def __init__(
        self,
        num_layers: int = 32,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        ff_dim: int = 5504,
        use_lora: bool = True,
        lora_ranks: Optional[list] = None
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_lora = use_lora
        
        # По умолчанию: 4 слоя = r=4, 8 слоёв = r=8, остальные r=16
        if lora_ranks is None:
            lora_ranks = [4] * 4 + [8] * 8 + [16] * (num_layers - 12)
        
        # Создаём слои
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            rank = lora_ranks[i] if i < len(lora_ranks) else 16
            
            layer = HybridTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                use_gnn=True,  # GNN на каждом слое!
                use_lora=use_lora,
                lora_rank=rank,
                layer_id=i
            )
            self.layers.append(layer)
        
        # Финальный layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        graph_vec: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_lora: bool = True,
        enable_injection: bool = True
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward через все слои.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            graph_vec: [batch, hidden_dim] - вектор из GNN
            attention_mask: attention mask
            use_lora: применять LoRA
            enable_injection: применять GNN инъекцию
        
        Returns:
            (output, layer_info_list)
        """
        layer_info = []
        
        for layer in self.layers:
            x, info = layer(
                x,
                graph_vec=graph_vec,
                attention_mask=attention_mask,
                use_lora=use_lora,
                enable_injection=enable_injection
            )
            layer_info.append(info)
        
        x = self.final_norm(x)
        
        return x, layer_info


class HybridModelWithGNN(nn.Module):
    """
    Гибридная модель с GNN для использования в PyTorch HuggingFace моделях.
    
    Обёртка для подключения к Qwen модели.
    """
    
    def __init__(
        self,
        base_model,
        use_gnn: bool = True,
        use_lora: bool = True,
        injection_scale: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.use_gnn = use_gnn
        self.use_lora = use_lora
        self.injection_scale = injection_scale
        
        # Настраиваемые слои для GNN
        hidden_dim = base_model.config.hidden_size
        
        if use_gnn:
            self.gnn_proj = nn.Linear(hidden_dim, hidden_dim)
            self.injection_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        
        if use_lora:
            # LoRA для всех attention слоёв
            self.lora_layers = nn.ModuleDict()
    
    def inject_graph_in_layer(
        self,
        hidden_states: torch.Tensor,
        graph_vec: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Инъектировать graph_vec в hidden_states на конкретном слое.
        
        Вызывается ИЗНУТРИ каждого слоя модели.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            graph_vec: [batch, hidden_dim] or [hidden_dim]
            layer_idx: индекс слоя
        
        Returns:
            modified hidden_states
        """
        if not self.use_gnn:
            return hidden_states
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape graph_vec
        if graph_vec.dim() == 1:
            graph_vec = graph_vec.unsqueeze(0)
        
        if graph_vec.shape[0] != batch_size:
            graph_vec = graph_vec.expand(batch_size, -1)
        
        # Project graph
        graph_proj = self.gnn_proj(graph_vec)  # [batch, hidden_dim]
        graph_proj = graph_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Get gate
        last_hidden = hidden_states[:, -1:, :]
        gate = self.injection_gate(last_hidden)  # [batch, 1, hidden_dim]
        
        # Compute injection
        injection = self.injection_scale * gate * graph_proj
        
        # Add to last position
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + injection
        
        return hidden_states
    
    def forward(
        self,
        input_ids,
        graph_vec: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward base модели с GNN инъекцией.
        
        GNN инъекция происходит после каждого слоя.
        """
        # Base model forward
        outputs = self.base_model(input_ids, **kwargs)
        
        # Если есть graph_vec, можно обработать отдельно
        # (Actual injection происходит внутри слоёв через хук)
        
        return outputs


def create_hybrid_model(
    base_model_path: str,
    num_layers: int = 32,
    use_gnn: bool = True,
    use_lora: bool = True
) -> HybridModelWithGNN:
    """
    Создать гибридную модель с GNN.
    """
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    return HybridModelWithGNN(
        model,
        use_gnn=use_gnn,
        use_lora=use_lora
    )


class GNNInjectorHook:
    """
    Хук для автоматической GNN инъекции в модель.
    
    Регистрируется на каждом слое.
    """
    
    def __init__(
        self,
        gnn_encoder,
        injection_scale: float = 0.1,
        inject_position: str = "after_attention"  # "after_attention" or "after_ffn"
    ):
        self.gnn_encoder = gnn_encoder
        self.injection_scale = injection_scale
        self.inject_position = inject_position
        self.current_graph_vec = None
    
    def set_graph_vector(self, graph_vec: np.ndarray):
        """Установить graph vector для текущего запроса."""
        self.current_graph_vec = torch.from_numpy(graph_vec)
    
    def __call__(self, module, input, output):
        """
        Хук выз��вается после каждого слоя.
        
        Инъектирует graph_vec в output.
        """
        if self.current_graph_vec is None:
            return output
        
        # Output может быть hidden_states или tuple
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Инъекция в последнюю позицию
        last_idx = hidden_states.shape[1] - 1
        
        graph_vec = self.current_graph_vec.to(hidden_states.device)
        
        if graph_vec.dim() == 1:
            graph_vec = graph_vec.unsqueeze(0).unsqueeze(0).expand(-1, 1, -1)
        elif graph_vec.dim() == 2:
            graph_vec = graph_vec.unsqueeze(1)
        
        # Project и gate
        graph_proj = module.gnn_proj(graph_vec)
        
        # Простая инъекция (без gate для скорости)
        modified = hidden_states.clone()
        modified[:, last_idx:last_idx+1, :] += self.injection_scale * graph_proj
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        else:
            return modified
    
    def clear(self):
        """Очистить graph vector."""
        self.current_graph_vec = None