"""
FractalGraphEncoder - GNN энкодер с SAGEConv и HNSW

Реализация из "Последовательные решения.txt":
- SAGEConv слои (conv1, conv2)
- HNSW индекс для семантического поиска
- retrieve_subgraph() для получения подграфа
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Попытка импортировать torch-geometric
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

# HNSW
try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


class FractalGraphEncoder(nn.Module):
    """
    FractalGraphEncoder - GNN энкодер для графовых структур.
    
    Особенности:
    - SAGEConv conv1: input_dim → hidden_dim
    - SAGEConv conv2: hidden_dim → output_dim
    - Linear proj для output
    - Gate projection для адаптивной инъекции
    - HNSW индекс для поиска
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 2560
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if HAS_TORCH_GEOMETRIC:
            # SAGEConv слои
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        else:
            # Fallback на обычных linear слоях
            self.conv1 = nn.Linear(input_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, output_dim)
        
        # Output projection
        self.proj = nn.Linear(output_dim, output_dim)
        
        # Gate projection (2*output_dim → output_dim)
        self.gate_proj = nn.Linear(2 * output_dim, output_dim)
        
        # HNSW индекс
        self._hnsw = None
        self._graph_nodes: List[Dict] = []
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward проход через GNN.
        
        Args:
            x: [num_nodes, input_dim] - узлы
            edge_index: [2, num_edges] - рёбра
            batch: [num_nodes] - batch индекс для pooling
        
        Returns:
            (graph_vec, gate_weights)
            - graph_vec: [batch_size, output_dim]
            - gate_weights: [batch_size, output_dim]
        """
        if not HAS_TORCH_GEOMETRIC:
            # Fallback forward
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
        else:
            # SAGEConv forward
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
        
        # Pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_vec = self.proj(global_mean_pool(h, batch))
        
        # Gate
        gate_input = torch.cat([graph_vec, graph_vec], dim=-1)
        gate_weights = self.gate_proj(gate_input)
        
        return graph_vec, gate_weights
    
    def build_hnsw_index(
        self,
        graph_nodes: List[Dict],
        dim: int = 2560,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16
    ):
        """
        Построение HNSW индекса из узлов графа.
        
        Args:
            graph_nodes: список узлов с 'embedding' полем
            dim: размерность эмбеддингов
            max_elements: макс. элементов
            ef_construction: параметр построения
            M: количество связей
        """
        if not HAS_HNSWLIB:
            raise ImportError("hnswlib не установлен")
        
        self._graph_nodes = graph_nodes
        
        # Создаём индекс
        self._hnsw = hnswlib.Index(space='cosine', dim=dim)
        self._hnsw.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )
        
        # Собираем эмбеддинги
        embeddings = []
        for node in graph_nodes:
            emb = node.get('embedding')
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            embeddings = np.array(embeddings)
            labels = np.arange(len(embeddings))
            self._hnsw.add_items(embeddings, labels)
            self._hnsw.set_ef(50)
    
    def retrieve_subgraph(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Получение подграфа по запросу.
        
        Args:
            query_embedding: [dim] - эмбеддинг запроса
            k: кол-во ближайших соседей
        
        Returns:
            {
                'x': node_embeddings,
                'edge_index': edge_index,
                'node_ids': labels,
                'contents': node_contents
            }
        """
        if self._hnsw is None:
            return {
                'x': np.array([]),
                'edge_index': np.array([[], []]),
                'node_ids': [],
                'contents': []
            }
        
        # KNN запрос
        labels, distances = self._hnsw.knn_query(query_embedding.reshape(1, -1), k=k)
        labels = labels[0]
        distances = distances[0]
        
        # Собираем эмбеддинги узлов
        node_embs = []
        node_contents = []
        node_ids = []
        
        for i, label in enumerate(labels):
            if label < len(self._graph_nodes):
                node = self._graph_nodes[label]
                node_embs.append(node.get('embedding', np.zeros(self.output_dim)))
                node_contents.append(node.get('content', ''))
                node_ids.append(node.get('id', str(label)))
        
        if not node_embs:
            return {
                'x': np.array([]),
                'edge_index': np.array([[], []]),
                'node_ids': [],
                'contents': []
            }
        
        node_embs = np.array(node_embs)
        
        # Создаём edge_index (упрощённо - полный граф между найденными узлами)
        num_nodes = len(node_embs)
        if num_nodes > 0:
            # Create edges: each node connected to next
            edge_index = np.array([
                np.arange(num_nodes - 1),
                np.arange(1, num_nodes)
            ])
        else:
            edge_index = np.array([[], []])
        
        return {
            'x': node_embs,
            'edge_index': edge_index,
            'node_ids': node_ids,
            'contents': node_contents,
            'distances': distances.tolist()
        }
    
    def search(
        self,
        query: str,
        encoder: Any,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Поиск по текстовому запросу.
        
        Args:
            query: текстовый запрос
            encoder: SentenceTransformer для кодирования
            k: кол-во результатов
        
        Returns:
            [(content, score), ...]
        """
        q_emb = encoder.encode(query, normalize_embeddings=True)
        
        result = self.retrieve_subgraph(q_emb, k=k)
        
        return list(zip(result['contents'], result['distances']))


class GraphEncoderRuntime:
    """
    Runtime обёртка для GraphEncoder (без torch-geometric зависимостей).
    
    Использует numpy вместо torch.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 2560
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights (numpy implementation)
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.02
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.02
        self.proj = np.random.randn(output_dim, output_dim).astype(np.float32) * 0.02
        self.gate_W = np.random.randn(2 * output_dim, output_dim).astype(np.float32) * 0.02
        
        self._hnsw = None
        self._graph_nodes = []
    
    def encode(
        self,
        x: np.ndarray,
        edge_index: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode узлов в graph vector.
        
        Args:
            x: [num_nodes, input_dim]
            edge_index: [2, num_edges]
        
        Returns:
            (graph_vec, gate_weights) - [1, output_dim]
        """
        # Простой forward без conv (fallback)
        h = np.tanh(x @ self.W1)
        h = np.tanh(h @ self.W2)
        
        # Mean pooling
        graph_vec = np.mean(h, axis=0, keepdims=True)
        graph_vec = graph_vec @ self.proj
        
        # Gate
        gate_input = np.concatenate([graph_vec, graph_vec], axis=-1)
        gate_weights = gate_input @ self.gate_W
        
        return graph_vec, gate_weights
    
    def build_hnsw_index(
        self,
        graph_nodes: List[Dict],
        dim: int = 2560
    ):
        """Построить HNSW индекс."""
        if not HAS_HNSWLIB:
            return
        
        self._graph_nodes = graph_nodes
        
        self._hnsw = hnswlib.Index(space='cosine', dim=dim)
        self._hnsw.init_index(max_elements=len(graph_nodes), ef_construction=200, M=16)
        
        embeddings = np.array([n.get('embedding', np.zeros(dim)) for n in graph_nodes])
        self._hnsw.add_items(embeddings, np.arange(len(graph_nodes)))
        self._hnsw.set_ef(50)
    
    def retrieve_subgraph(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Dict[str, Any]:
        """Получить подграф."""
        if self._hnsw is None:
            return {'x': np.array([]), 'edge_index': np.array([[], []]), 'node_ids': [], 'contents': []}
        
        labels, distances = self._hnsw.knn_query(query_embedding.reshape(1, -1), k=k)
        
        nodes = [self._graph_nodes[i] for i in labels[0]]
        
        return {
            'x': np.array([n.get('embedding', np.zeros(self.output_dim)) for n in nodes]),
            'edge_index': np.array([np.arange(len(nodes)), np.roll(np.arange(len(nodes)), 1)]),
            'node_ids': [n.get('id', str(i)) for i in labels[0]],
            'contents': [n.get('content', '') for n in nodes]
        }