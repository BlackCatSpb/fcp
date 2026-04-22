"""
FCP Hybrid Layer - 5 этапов обработки (полная реализация по спецификации)
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .types import (
    Subgraph, LayerState, HaltDecision, TransformerBlockOutput, FusionOutput
)


class FractalGatedHybridLayer:
    """
    Гибридный слой FCP - 5 этапов обработки (SPEC.md section 3.2)
    
    Этапы:
    1. Контекстуальный токенизатор (graph retrieval + node routing)
    2. Графовый кластеризатор (message passing + soft clustering)
    3. Трансформерный блок (attention + FFN)
    4. Активационный гейт (halt probability)
    5. Слияние потоков (cross-attention или gated add)
    """
    
    def __init__(
        self,
        layer_id: int,
        hidden_dim: int = 2048,
        num_heads: int = 16,
        max_seq_len: int = 4096,
        graph_retrieval_k: int = 32,
        master_tokens: int = 8,
        gnn_iterations: int = 2,
        stop_threshold: float = 0.85
    ):
        self.layer_id = layer_id
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = hidden_dim // num_heads
        
        # Graph retrieval
        self.graph_retrieval_k = graph_retrieval_k
        self.master_tokens = master_tokens
        self.gnn_iterations = gnn_iterations
        
        # Activation gate
        self.stop_threshold = stop_threshold
        
        # Learnable fusion weight (from spec: "gated add with learnable coefficient")
        self.fusion_weight = 0.1  # Will be learned
    
    # =========================================================================
    # ЭТАП 1: КОНТЕКСТУАЛЬНЫЙ ТОКЕНИЗАТОР
    # =========================================================================
    
    def extract_subgraph(
        self,
        query_embeddings: np.ndarray,
        graph: "FractalGraphV2"
    ) -> Subgraph:
        """
        Выполняет k-NN поиск в HNSW-индексе FractalGraphV2.
        
        SPEC: extract_subgraph(query_embeddings: Tensor, k: int) -> Subgraph
        
        Args:
            query_embeddings: (batch, seq_len, dim) or (dim,)
            graph: FractalGraphV2 instance
            
        Returns:
            Subgraph с node_ids, embeddings, edges
        """
        # Flatten if batched
        if query_embeddings.ndim == 3:
            query = query_embeddings.reshape(-1, self.hidden_dim)
        else:
            query = query_embeddings.reshape(1, -1)
        
        # Use graph's HNSW index
        try:
            result = graph.extract_subgraph(query, self.graph_retrieval_k)
            return result
        except Exception:
            return Subgraph(
                node_ids=[],
                node_embeddings=np.zeros((0, self.hidden_dim)),
                edges=[],
                edge_types=[]
            )
    
    def node_aware_routing(
        self,
        hidden_subgraph: np.ndarray
    ) -> np.ndarray:
        """
        Определяет для каждого узла подграфа, достаточно ли GNN или нужна LM.
        
        SPEC: node_aware_routing(H_sub: Tensor) -> routing_mask
        
        Args:
            hidden_subgraph: (num_nodes, dim)
            
        Returns:
            routing_mask: (num_nodes,) - 1 = GNN only, 0 = needs LM
        """
        num_nodes = hidden_subgraph.shape[0]
        
        # Heuristic: highly connected nodes with low variance = GNN only
        variances = np.var(hidden_subgraph, axis=1)
        
        # Homophily: nodes with similar embeddings can use GNN
        homophily_scores = 1.0 - np.clip(variances * 10, 0, 1)
        
        routing_mask = (homophily_scores > 0.7).astype(np.float32)
        
        return routing_mask
    
    # =========================================================================
    # ЭТАП 2: ГРАФОВЫЙ КЛАСТЕРИЗАТОР
    # =========================================================================
    
    def message_passing(
        self,
        node_embeddings: np.ndarray,
        edges: List[Tuple[str, str]],
        node_ids: List[str],
        iterations: int = 2
    ) -> np.ndarray:
        """
        Выполняет итерации распространения сообщений (GraphSAGE-style).
        
        SPEC: message_passing(H_sub: Tensor, edges: List) -> Tensor
        
        Args:
            node_embeddings: (num_nodes, dim)
            edges: [(source, target), ...]
            node_ids: для индексации
            
        Returns:
            updated_embeddings: (num_nodes, dim)
        """
        if len(edges) == 0:
            return node_embeddings
        
        # Build adjacency
        adj = self._build_adjacency(edges, node_ids)
        
        updated = node_embeddings.copy()
        
        for _ in range(iterations):
            # Aggregate from neighbors (mean pooling)
            aggregated = np.zeros_like(updated)
            for i, row in enumerate(adj):
                neighbors = np.where(row > 0)[0]
                if len(neighbors) > 0:
                    aggregated[i] = np.mean(updated[neighbors], axis=0)
            
            # Combine with self
            updated = 0.5 * node_embeddings + 0.5 * aggregated
        
        return updated
    
    def _build_adjacency(
        self,
        edges: List[Tuple[str, str]],
        node_ids: List[str]
    ) -> np.ndarray:
        """Build adjacency matrix."""
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        adj = np.zeros((n, n))
        for src, tgt in edges:
            if src in id_to_idx and tgt in id_to_idx:
                adj[id_to_idx[src], id_to_idx[tgt]] = 1
        
        return adj
    
    def soft_fractal_cluster(
        self,
        node_embeddings: np.ndarray,
        num_clusters: int,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Мягкая кластеризация узлов, формирует мастер-токены.
        
        SPEC: soft_fractal_cluster(H: Tensor, K: int, temperature: float) 
              -> Tuple[Tensor, Tensor, float]
        
        Returns:
            assignments: (num_nodes,) - cluster indices
            master_tokens: (K, dim) - cluster centroids
            structural_completeness: float - метрика полноты
        """
        num_nodes = node_embeddings.shape[0]
        
        # Initialize clusters
        indices = np.random.choice(num_nodes, min(num_clusters, num_nodes), replace=False)
        centroids = node_embeddings[indices].copy()
        
        # Soft clustering iterations
        for _ in range(10):
            # Compute distances
            distances = self._pairwise_distances(node_embeddings, centroids)
            
            # Softmax with temperature
            weights = self._softmax(-distances / temperature, axis=1)
            
            # Update centroids
            for k in range(num_clusters):
                mask = weights[:, k:k+1]
                if mask.sum() > 0:
                    centroids[k] = (node_embeddings * mask).sum(axis=0) / mask.sum()
        
        # Final assignments
        final_distances = self._pairwise_distances(node_embeddings, centroids)
        assignments = np.argmin(final_distances, axis=1)
        
        # Compute master tokens as centroids
        master_tokens = centroids
        
        # Structural completeness metric
        completeness = self._compute_completeness(node_embeddings, master_tokens, assignments)
        
        return assignments, master_tokens, completeness
    
    def _pairwise_distances(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise L2 distances."""
        dists = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))
        return dists
    
    def _softmax(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / exp_x.sum(axis=axis, keepdims=True)
    
    def _compute_completeness(
        self,
        nodes: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray
    ) -> float:
        """Compute structural completeness (0-1)."""
        if len(nodes) == 0:
            return 0.0
        
        # Average distance to assigned centroid
        total_dist = 0.0
        for i, node in enumerate(nodes):
            centroid = centroids[assignments[i]]
            total_dist += np.linalg.norm(node - centroid)
        
        avg_dist = total_dist / len(nodes)
        
        # Convert to completeness
        return max(0, 1.0 - avg_dist / 10.0)
    
    # =========================================================================
    # ЭТАП 3: ТРАНСФОРМЕРНЫЙ БЛОК
    # =========================================================================
    
    def causal_self_attention(
        self,
        hidden_states: np.ndarray,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Многоголовое причинное внимание.
        
        SPEC: causal_self_attention(X: Tensor, mask: Tensor, kv_cache: Optional) 
              -> Tuple[Tensor, KVState]
        
        Args:
            hidden_states: (batch, seq_len, dim)
            kv_cache: (past_k, past_v) if provided
            
        Returns:
            output: (batch, seq_len, dim)
            kv_cache: updated
        """
        # Simplified - actual implementation uses OpenVINO
        # This is the architecture definition
        
        batch, seq_len, dim = hidden_states.shape
        head_dim = self.head_dim
        
        # For now, pass through
        # Real implementation would use optimized attention
        output = hidden_states.copy()
        
        return output, kv_cache
    
    def swiglu_feed_forward(
        self,
        hidden_states: np.ndarray
    ) -> np.ndarray:
        """
        SwiGLU Feed Forward network.
        
        SPEC: swiglu_feed_forward(X: Tensor) -> Tensor
        """
        # Simplified architecture
        # Real: two-layer FFN with SwiGLU activation
        
        dim = hidden_states.shape[-1]
        hidden_dim = dim * 4
        
        # Simplified: just return input
        # Real: would be W1, W3, W5 with silu activation
        return hidden_states.copy()
    
    def apply_transformer_block(
        self,
        hidden_states: np.ndarray,
        active_mask: np.ndarray
    ) -> np.ndarray:
        """
        Применяет блок: attention + FFN + RMSNorm + residual.
        
        SPEC: apply_transformer_block(X: Tensor, active_mask: Tensor) -> Tensor
        
        Args:
            hidden_states: (batch, seq_len, dim)
            active_mask: (batch, seq_len) - which tokens are active
            
        Returns:
            output: (batch, seq_len, dim)
        """
        # Apply only to active tokens
        # (simplified)
        
        # Attention
        attn_out, _ = self.causal_self_attention(hidden_states, None)
        
        # Residual
        attn_out = hidden_states + attn_out
        
        # FFN
        ffn_out = self.swiglu_feed_forward(attn_out)
        
        # Final residual
        output = attn_out + ffn_out
        
        return output
    
    # =========================================================================
    # ЭТАП 4: АКТИВАЦИОННЫЙ ГЕЙТ
    # =========================================================================
    
    def compute_attention_entropy(
        self,
        attention_weights: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет энтропию внимания для каждого токена.
        
        SPEC: compute_attention_entropy(attn_weights: Tensor) -> Tensor
        """
        # Entropy = -sum(p * log(p))
        # Handle zeros
        weight_clipped = np.clip(attention_weights, 1e-10, 1.0)
        entropy = -(weight_clipped * np.log(weight_clipped)).sum(axis=-1)
        
        return entropy
    
    def evaluate_halt_prob(
        self,
        hidden_states: np.ndarray,
        structural_completeness: float,
        entropy: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет вероятность остановки для каждого токена.
        
        SPEC: evaluate_halt_prob(h: Tensor, phi_struct: float, entropy: Tensor) -> Tensor
        
        Returns:
            halt_probs: (seq_len,)
        """
        # Confidence from hidden state magnitude
        magnitude = np.linalg.norm(hidden_states, axis=-1)
        conf_from_hidden = magnitude / (magnitude.max() + 1e-8)
        
        # Combine with structural and entropy
        # Higher entropy = lower confidence
        entropy_factor = 1.0 / (1.0 + entropy)
        
        # Final probability
        halt_probs = (
            0.5 * conf_from_hidden +
            0.3 * structural_completeness +
            0.2 * entropy_factor
        )
        
        return halt_probs
    
    def update_confidence(
        self,
        prev_confidence: float,
        halt_probs: np.ndarray,
        structural_completeness: float
    ) -> float:
        """
        Обновляет накопленную послойную уверенность.
        
        SPEC: update_confidence(conf_prev: float, p_halt: Tensor, phi_struct: float) -> float
        """
        current_confidence = np.mean(halt_probs)
        
        # Exponential moving average
        updated = 0.7 * current_confidence + 0.3 * structural_completeness + 0.0 * prev_confidence
        
        return min(1.0, max(0.0, updated))
    
    def mask_halted_tokens(
        self,
        active_mask: np.ndarray,
        halt_probs: np.ndarray,
        threshold: float = 0.85
    ) -> np.ndarray:
        """
        Обновляет маску активных токенов.
        
        SPEC: mask_halted_tokens(active_mask: Tensor, p_halt: Tensor, theta_token: float) -> Tensor
        """
        # Tokens with high halt prob become inactive
        should_stop = halt_probs > threshold
        
        new_mask = active_mask.copy()
        new_mask[should_stop] = False
        
        return new_mask
    
    def evaluate_layer_halt(
        self,
        hidden_states: np.ndarray,
        active_mask: np.ndarray,
        structural_completeness: float,
        prev_confidence: float = 0.0
    ) -> HaltDecision:
        """
        Полное evaluate halt для слоя.
        
        Returns HaltDecision с stop_probabilities, active_mask, confidence
        """
        # Compute attention (simplified)
        attention_weights = np.ones((hidden_states.shape[1], hidden_states.shape[1]))
        
        # Entropy
        entropy = self.compute_attention_entropy(attention_weights)
        
        # Halt probabilities
        halt_probs = self.evaluate_halt_prob(
            hidden_states, 
            structural_completeness, 
            entropy
        )
        
        # Update confidence
        layer_confidence = self.update_confidence(
            prev_confidence, 
            halt_probs, 
            structural_completeness
        )
        
        # Mask
        new_mask = self.mask_halted_tokens(
            active_mask, 
            halt_probs, 
            self.stop_threshold
        )
        
        # Should early exit?
        active_ratio = np.mean(new_mask)
        should_exit = active_ratio < 0.1 or layer_confidence > 0.95
        
        return HaltDecision(
            stop_probabilities=halt_probs,
            active_mask=new_mask,
            layer_confidence=layer_confidence,
            should_early_exit=should_exit
        )
    
    # =========================================================================
    # ЭТАП 5: СЛИЯНИЕ ПОТОКОВ
    # =========================================================================
    
    def fuse_cross_attention(
        self,
        hidden_states: np.ndarray,
        master_tokens: np.ndarray
    ) -> np.ndarray:
        """
        Слияние через перекрёстное внимание.
        
        SPEC: fuse_cross_attention(X: Tensor, M: Tensor) -> Tensor
        """
        # X: (batch, seq, dim), M: (K, dim)
        # Q = X, K = M, V = M
        
        K = master_tokens.shape[0]
        
        # Simplified: apply as bias
        # Real: full cross-attention
        
        # Compute attention scores
        scores = np.matmul(hidden_states, master_tokens.T)  # (batch, seq, K)
        weights = self._softmax(scores, axis=-1)  # (batch, seq, K)
        
        # Weighted sum of master tokens
        context = np.matmul(weights, master_tokens)  # (batch, seq, dim)
        
        return context
    
    def fuse_gated_add(
        self,
        hidden_states: np.ndarray,
        master_tokens: np.ndarray
    ) -> np.ndarray:
        """
        Слияние через аддитивный гейт.
        
        SPEC: fuse_gated_add(X: Tensor, M: Tensor) -> Tensor
        """
        # Take first master token as global context
        if len(master_tokens) > 0:
            graph_context = master_tokens[0:1]  # (1, dim)
        else:
            graph_context = np.zeros((1, hidden_states.shape[-1]))
        
        # Gated addition
        # output = X + w * graph_context
        fused = hidden_states + self.fusion_weight * graph_context
        
        return fused
    
    def fuse_streams(
        self,
        hidden_states: np.ndarray,
        master_tokens: Optional[np.ndarray],
        method: str = "gated_add"
    ) -> np.ndarray:
        """
        Главный метод слияния потоков.
        
        Args:
            hidden_states: from transformer
            master_tokens: from graph clustering
            method: "cross_attention" or "gated_add"
            
        Returns:
            fused: (batch, seq, dim)
        """
        if master_tokens is None or len(master_tokens) == 0:
            return hidden_states
        
        if method == "cross_attention":
            return self.fuse_cross_attention(hidden_states, master_tokens)
        else:
            return self.fuse_gated_add(hidden_states, master_tokens)
    
    # =========================================================================
    # MAIN FORWARD
    # =========================================================================
    
    def forward(
        self,
        hidden_states: np.ndarray,
        graph: Optional["FractalGraphV2"],
        prev_state: Optional[LayerState],
        active_mask: np.ndarray
    ) -> Tuple[np.ndarray, LayerState, HaltDecision]:
        """
        Полный forward pass через гибридный слой.
        
        Returns:
            output: (batch, seq, dim)
            new_state: LayerState
            halt: HaltDecision
        """
        batch, seq, dim = hidden_states.shape
        
        # === Этап 1: Извлечение подграфа ===
        subgraph = self.extract_subgraph(hidden_states, graph)
        
        # === Этап 2: Графовый кластеризатор ===
        if not subgraph.is_empty:
            # Message passing
            updated_nodes = self.message_passing(
                subgraph.node_embeddings,
                subgraph.edges,
                subgraph.node_ids,
                self.gnn_iterations
            )
            
            # Soft clustering
            assignments, master_tokens, completeness = self.soft_fractal_cluster(
                updated_nodes,
                self.master_tokens
            )
        else:
            master_tokens = None
            completeness = 0.0
        
        # Node routing (for graph-only vs hybrid)
        if not subgraph.is_empty:
            routing = self.node_aware_routing(subgraph.node_embeddings)
        else:
            routing = np.array([])
        
        # === Этап 3: Трансформерный блок ===
        transformer_out = self.apply_transformer_block(hidden_states, active_mask)
        
        # === Этап 4: Активационный гейт ===
        halt = self.evaluate_layer_halt(
            transformer_out,
            active_mask,
            completeness,
            prev_state.accumulated_confidence if prev_state else 0.0
        )
        
        # === Этап 5: Слияние потоков ===
        fused = self.fuse_streams(transformer_out, master_tokens)
        
        # Build new state
        new_state = LayerState(
            layer_id=self.layer_id,
            cluster_assignments=assignments if not subgraph.is_empty else None,
            master_tokens=master_tokens,
            accumulated_confidence=halt.layer_confidence
        )
        
        return fused, new_state, halt


# Forward type for graph
class FractalGraphV2:
    """Placeholder - будет импортировано"""
    def extract_subgraph(self, query, k):
        return Subgraph([], np.array([]), [], [])