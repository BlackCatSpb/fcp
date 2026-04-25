"""
Adaptive Fusion Injector - Адаптивная инъекция графового вектора

Инъектирует graph_vec в hidden_states модели.
"""
import numpy as np
from typing import Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Сигмоидная функция."""
    return 1 / (1 + np.exp(-x))


class AdaptiveFusionInjector:
    """
    Адаптивный инъектор графового вектора в скрытые состояния.
    
    Реализация из "Последовательные решения.txt":
    - Инъекция graph_vec в hidden_states
    - Gate mechanism для контроля
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        injection_scale: float = 0.1
    ):
        self.hidden_dim = hidden_dim
        self.injection_scale = injection_scale
        
        # Gate weights (2*hidden_dim -> hidden_dim)
        self.gate_weights: Optional[np.ndarray] = None
    
    def inject(
        self,
        hidden_states: np.ndarray,
        graph_vec: np.ndarray,
        gate_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Инъектировать graph_vec в hidden_states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            graph_vec: [batch, hidden_dim] or [hidden_dim]
            gate_weights: [batch, hidden_dim] or [hidden_dim]
        
        Returns:
            modified hidden_states: [batch, seq_len, hidden_dim]
        """
        # Reshape graph_vec
        if graph_vec.ndim == 1:
            graph_vec = graph_vec.reshape(1, 1, -1)
        elif graph_vec.ndim == 2:
            graph_vec = graph_vec.reshape(1, 1, -1)
        
        # Использовать предоставленные gate_weights или создать
        if gate_weights is None:
            if self.gate_weights is None:
                gate_weights = np.ones_like(graph_vec.squeeze(0))
            else:
                gate_weights = self.gate_weights
        else:
            if gate_weights.ndim == 1:
                gate_weights = gate_weights.reshape(1, -1)
        
        if gate_weights.ndim == 2:
            gate_weights = gate_weights.reshape(1, 1, -1)
        
        # Get last hidden state
        h_last = hidden_states[:, -1:, :]  # [batch, 1, hidden_dim]
        
        # Compute gate
        combined = np.concatenate([h_last, graph_vec], axis=-1)
        
        if combined.shape[-1] == 2 * self.hidden_dim:
            if gate_weights is not None and gate_weights.size > 0:
                gate = sigmoid(np.dot(combined.squeeze(1), gate_weights.T))
                gate = gate.reshape(1, 1, 1)
            else:
                gate = sigmoid(np.mean(combined, axis=-1, keepdims=True))
        else:
            gate = 0.5 * np.ones((hidden_states.shape[0], 1, 1))
        
        # Inject
        injection = self.injection_scale * gate * graph_vec
        
        # Добавляем к последнему state
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + injection
        
        return hidden_states
    
    def inject_multiple(
        self,
        hidden_states: np.ndarray,
        graph_vecs: list,
        gate_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Инъектировать несколько graph_vec.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            graph_vecs: список vectors для инъекции
            gate_weights: gate weights
        
        Returns:
            modified hidden_states
        """
        result = hidden_states.copy()
        
        for gv in graph_vecs:
            result = self.inject(result, gv, gate_weights)
        
        return result
    
    def set_gate_weights(self, weights: np.ndarray):
        """Установить фиксированные gate weights."""
        self.gate_weights = weights
    
    def reset(self):
        """Сбросить gate weights."""
        self.gate_weights = None


class TextFusionInjector:
    """
    Текстовый инъектор - преобразует подграф в текст.
    
    Альтернатива скрытой инъекции - добавляет текст к промпту.
    """
    
    def __init__(
        self,
        max_nodes: int = 5,
        format_template: str = "Контекст: {context}"
    ):
        self.max_nodes = max_nodes
        self.format_template = format_template
    
    def format_subgraph(self, subgraph: dict) -> str:
        """
        Форматировать подграф в текст.
        
        Args:
            subgraph: {
                'contents': [...],
                'node_ids': [...],
                'distances': [...]
            }
        
        Returns:
            formatted string
        """
        contents = subgraph.get('contents', [])
        
        if not contents:
            return ""
        
        # Ограничиваем кол-во узлов
        limited = contents[:self.max_nodes]
        
        # Форматируем
        lines = []
        for i, content in enumerate(limited):
            lines.append(f"{i+1}. {content}")
        
        context_str = "\n".join(lines)
        
        return self.format_template.format(context=context_str)
    
    def inject_to_prompt(
        self,
        prompt: str,
        subgraph: dict,
        position: str = "prefix"
    ) -> str:
        """
        Инъектировать подграф в промпт.
        
        Args:
            prompt: исходный промпт
            subgraph: данные подграфа
            position: 'prefix' или 'suffix'
        
        Returns:
            modified prompt
        """
        context = self.format_subgraph(subgraph)
        
        if not context:
            return prompt
        
        if position == "prefix":
            return f"{context}\n\n{prompt}"
        else:
            return f"{prompt}\n\n{context}"


class HybridFusionInjector:
    """
    Гибридный инъектор - комбинирует текстовую и скрытую инъекцию.
    """
    
    def __init__(
        self,
        hidden_dim: int = 2048,
        use_text: bool = True,
        use_hidden: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.text_injector = TextFusionInjector() if use_text else None
        self.hidden_injector = AdaptiveFusionInjector(hidden_dim) if use_hidden else None
    
    def inject(
        self,
        prompt: str,
        hidden_states: np.ndarray,
        subgraph: dict,
        gate_weights: Optional[np.ndarray] = None,
        inject_mode: str = "both"  # "text", "hidden", "both"
    ) -> tuple:
        """
        Инъектировать данные графа.
        
        Args:
            prompt: текстовый промпт
            hidden_states: скрытые состояния
            subgraph: данные подграфа
            gate_weights: gate weights
            inject_mode: режим инъекции
        
        Returns:
            (modified_prompt, modified_hidden_states)
        """
        result_prompt = prompt
        result_hidden = hidden_states
        
        if inject_mode in ("text", "both") and self.text_injector:
            result_prompt = self.text_injector.inject_to_prompt(prompt, subgraph)
        
        if inject_mode in ("hidden", "both") and self.hidden_injector:
            graph_vec = self._subgraph_to_vec(subgraph)
            result_hidden = self.hidden_injector.inject(hidden_states, graph_vec, gate_weights)
        
        return result_prompt, result_hidden
    
    def _subgraph_to_vec(self, subgraph: dict) -> np.ndarray:
        """Преобразовать подграф в вектор (mean pool)."""
        x = subgraph.get('x')
        
        if x is None or len(x) == 0:
            return np.zeros(self.hidden_dim)
        
        if isinstance(x, list):
            x = np.array(x)
        
        # Mean pool
        vec = np.mean(x, axis=0)
        
        # Resize if needed
        if vec.shape[-1] != self.hidden_dim:
            if vec.shape[-1] < self.hidden_dim:
                padding = np.zeros(self.hidden_dim - vec.shape[-1])
                vec = np.concatenate([vec, padding])
            else:
                vec = vec[:self.hidden_dim]
        
        return vec