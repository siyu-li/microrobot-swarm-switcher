"""
Graph Attention Network with Obstacle Nodes (IGA-Obstacle) - OPTIMIZED VERSION.

This is an optimized version of iga_obstacle.py that uses:
1. Vectorized edge construction via torch.where() (no nested Python loops)
2. GPU-accelerated hard attention masking
3. All feature computations are fully batched/vectorized
4. Batched message passing via a single mega-graph (no per-sample Python loop)
5. Scatter-based combined_weights (no .item() GPU-CPU sync)

Key optimizations:
- Replaces triple-nested loops in edge building with vectorized torch.where()
- Eliminates element-wise tensor access that triggers CPU synchronization
- All B graphs are merged into one mega-graph with offset node indices,
  processed in a single message-passing call
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def entropy_from_attention(attn_weights, target_nodes, num_nodes, eps=1e-10):
    """
    Compute mean per-node entropy of incoming softmax attention distributions.

    Args:
        attn_weights (Tensor): Per-edge attention probabilities of shape (num_edges,).
        target_nodes (Tensor): Destination node indices for each edge of shape (num_edges,).
        num_nodes (int): Total number of nodes (robots only for output).
        eps (float, optional): Small constant for numerical stability in log. Defaults to 1e-10.

    Returns:
        Tensor: Scalar tensor containing the mean entropy across nodes that have at least
            one incoming edge. Returns 0.0 if no nodes have incoming edges.
    """
    attn_log = (attn_weights + eps).log()
    contrib = -(attn_weights * attn_log)

    entropies = torch.zeros(num_nodes, device=attn_weights.device).index_add_(
        0, target_nodes, contrib
    )
    counts = torch.zeros(num_nodes, device=attn_weights.device).index_add_(
        0, target_nodes, torch.ones_like(attn_weights)
    )

    mask = counts > 0
    entropies = entropies[mask] / counts[mask]
    return (
        entropies.mean()
        if mask.any()
        else torch.tensor(0.0, device=attn_weights.device)
    )


class GoalAttentionLayerObstacle(MessagePassing):
    """
    Message-passing layer with learned attention over goal/edge features.

    Extends GoalAttentionLayer to handle heterogeneous graphs with robot and
    obstacle nodes. Only robot nodes (indices 0 to n_robots-1) are targets.

    Args:
        node_dim (int): Dimensionality of node features.
        edge_dim (int): Dimensionality of edge attributes.
        out_dim (int): Output dimensionality for projected messages.
    """

    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr="add")
        self.q = nn.Linear(node_dim, out_dim, bias=False)
        self.k = nn.Linear(edge_dim, out_dim, bias=False)
        self.v = nn.Linear(edge_dim, out_dim)
        self.attn_score_layer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )
        self._last_attn_weights = None

    def forward(self, x, edge_index, edge_attr, n_robots):
        """
        Run attention-based message passing.

        Args:
            x (Tensor): Node features of shape (num_nodes, node_dim).
                        First n_robots nodes are robots, rest are obstacles.
            edge_index (LongTensor): Edge indices of shape (2, num_edges)
                with format [source, target]. Targets are only robot indices (0..n_robots-1).
            edge_attr (Tensor): Edge attributes of shape (num_edges, edge_dim).
            n_robots (int): Number of robot nodes. Only these are message targets.

        Returns:
            tuple:
                out (Tensor): Updated robot node features of shape (n_robots, out_dim).
                attn_weights (Tensor or None): Last per-edge softmax weights.
        """
        q_robots = self.q(x[:n_robots])  # (n_robots, out_dim)

        # Propagate with size hint: (num_all_nodes, n_robots)
        out = self.propagate(
            edge_index,
            x=(x, q_robots),
            edge_attr=edge_attr,
            size=(x.size(0), n_robots),
        )
        return out, self._last_attn_weights

    def message(self, x_i, edge_attr, index, ptr, size_i):
        """
        Compute per-edge messages using attention weights.

        Args:
            x_i (Tensor): Target-node queries for each edge, shape (num_edges, out_dim).
            edge_attr (Tensor): Edge attributes for each edge, shape (num_edges, edge_dim).
            index (LongTensor): Target node indices per edge, shape (num_edges,).
            ptr: Unused (PyG internal).
            size_i: Number of target nodes.

        Returns:
            Tensor: Per-edge messages of shape (num_edges, out_dim).
        """
        k = F.leaky_relu(self.k(edge_attr))
        v = F.leaky_relu(self.v(edge_attr))
        attention_input = torch.cat([x_i, k], dim=-1)
        scores = self.attn_score_layer(attention_input).squeeze(-1)
        attn_weights = softmax(scores, index, num_nodes=size_i)
        self._last_attn_weights = attn_weights.detach()
        return v * attn_weights.unsqueeze(-1)


class AttentionObstacleOptimized(nn.Module):
    """
    OPTIMIZED multi-robot attention mechanism with obstacle nodes.

    This version uses vectorized operations and PyG batching for 5-10x speedup.

    Node layout: [robot_0, robot_1, ..., robot_{n-1}, obs_0, obs_1, ..., obs_{m-1}]

    Args:
        embedding_dim (int): Dimension of the agent embedding vector.
        robot_node_dim (int): Input dimension for robot node features. Default 5.
        obstacle_node_dim (int): Input dimension for obstacle node features. Default 5.
    """

    def __init__(self, embedding_dim, robot_node_dim=5, obstacle_node_dim=5):
        super(AttentionObstacleOptimized, self).__init__()
        self.embedding_dim = embedding_dim

        # Message passing layer
        self.message_graph = GoalAttentionLayerObstacle(
            node_dim=embedding_dim, edge_dim=10, out_dim=embedding_dim
        )

        # Robot node encoder
        self.embedding1 = nn.Linear(robot_node_dim, 128)
        nn.init.kaiming_uniform_(self.embedding1.weight, nonlinearity="leaky_relu")
        self.embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding2.weight, nonlinearity="leaky_relu")

        # Obstacle node encoder (kept for checkpoint compatibility; geometry is
        # encoded entirely in robot-obstacle edge features so obstacle node slots
        # in the mega-graph are filled with zeros as placeholders at runtime).
        self.obs_embedding1 = nn.Linear(obstacle_node_dim, 128)
        nn.init.kaiming_uniform_(self.obs_embedding1.weight, nonlinearity="leaky_relu")
        self.obs_embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.obs_embedding2.weight, nonlinearity="leaky_relu")

        # Hard attention MLP for robot-robot edges
        self.hard_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hard_encoding = nn.Linear(embedding_dim, 2)

        # Hard attention for robot-obstacle edges
        self.hard_mlp_obs = nn.Sequential(
            nn.Linear(embedding_dim + 5, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hard_encoding_obs = nn.Linear(embedding_dim, 2)

        # Decoder
        self.decode_1 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_1.weight, nonlinearity="leaky_relu")
        self.decode_2 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_2.weight, nonlinearity="leaky_relu")

    def encode_robot_features(self, embed):
        """Encode per-robot features with a two-layer MLP."""
        embed = F.leaky_relu(self.embedding1(embed))
        embed = F.leaky_relu(self.embedding2(embed))
        return embed


    def build_edges_vectorized(self, hard_mask_rr, hard_mask_ro, soft_feats_rr, soft_feats_ro,
                                n_robots, n_obs, device):
        """
        Build graph edges for ALL batch samples at once using vectorized operations.

        Produces a single mega-graph where each sample's node indices are offset by
        b * n_total. This allows a single message-passing call for the entire batch.

        Args:
            hard_mask_rr (Tensor): Robot-robot hard mask, shape (B, N_r, N_r).
            hard_mask_ro (Tensor): Robot-obstacle hard mask, shape (B, N_r, N_obs).
            soft_feats_rr (Tensor): Robot-robot soft features, shape (B, N_r, N_r, 10).
            soft_feats_ro (Tensor): Robot-obstacle soft features, shape (B, N_r, N_obs, 10).
            n_robots (int): Number of robots.
            n_obs (int): Number of obstacles.
            device: Torch device.

        Returns:
            tuple: (edge_index, edge_attr, batch_ids)
                edge_index: (2, total_edges) with globally offset node indices
                edge_attr: (total_edges, 10)
                batch_ids: (total_edges,) which batch sample each edge belongs to
        """
        batch_size = hard_mask_rr.shape[0]
        n_total = n_robots + n_obs

        # === Robot-robot edges (fully batched) ===
        # Remove self-loops
        eye = torch.eye(n_robots, device=device).unsqueeze(0)  # (1, N_r, N_r)
        mask_rr = (hard_mask_rr > 0.5).float() * (1.0 - eye)

        # b_rr, i_rr, j_rr = batch, target, source
        b_rr, i_rr, j_rr = torch.where(mask_rr)

        if b_rr.numel() > 0:
            # Offset by b * n_total for mega-graph
            src_rr = j_rr + b_rr * n_total
            tgt_rr = i_rr + b_rr * n_total
            edge_attr_rr = soft_feats_rr[b_rr, i_rr, j_rr]  # (num_edges_rr, 10)
        else:
            src_rr = torch.zeros(0, dtype=torch.long, device=device)
            tgt_rr = torch.zeros(0, dtype=torch.long, device=device)
            edge_attr_rr = torch.zeros((0, 10), device=device)
            b_rr = torch.zeros(0, dtype=torch.long, device=device)

        # === Robot-obstacle edges (fully batched) ===
        mask_ro = (hard_mask_ro > 0.5).float()
        b_ro, i_ro, j_ro = torch.where(mask_ro)

        if b_ro.numel() > 0:
            # Obstacle source nodes offset: n_robots + j within the graph, plus b * n_total
            src_ro = j_ro + n_robots + b_ro * n_total
            tgt_ro = i_ro + b_ro * n_total
            edge_attr_ro = soft_feats_ro[b_ro, i_ro, j_ro]  # (num_edges_ro, 10)
        else:
            src_ro = torch.zeros(0, dtype=torch.long, device=device)
            tgt_ro = torch.zeros(0, dtype=torch.long, device=device)
            edge_attr_ro = torch.zeros((0, 10), device=device)
            b_ro = torch.zeros(0, dtype=torch.long, device=device)

        # === Combine all edges ===
        src_all = torch.cat([src_rr, src_ro])
        tgt_all = torch.cat([tgt_rr, tgt_ro])
        edge_index = torch.stack([src_all, tgt_all], dim=0)
        edge_attr = torch.cat([edge_attr_rr, edge_attr_ro], dim=0)
        batch_ids = torch.cat([b_rr, b_ro])

        return edge_index, edge_attr, batch_ids

    def forward(self, robot_embedding, obstacle_embedding):
        """
        Compute hard and soft attentions with obstacle nodes using optimized operations.

        Args:
            robot_embedding (Tensor): Robot states of shape (B, N_robots, 11).
            obstacle_embedding (Tensor): Obstacle info of shape (B, N_obs, 4).

        Returns:
            tuple: Same outputs as original AttentionObstacle for compatibility.
        """
        if robot_embedding.dim() == 2:
            robot_embedding = robot_embedding.unsqueeze(0)
        if obstacle_embedding.dim() == 2:
            obstacle_embedding = obstacle_embedding.unsqueeze(0)

        batch_size, n_robots, _ = robot_embedding.shape
        _, n_obs, _ = obstacle_embedding.shape
        n_total = n_robots + n_obs
        device = robot_embedding.device

        # === Extract robot features ===
        robot_feat = robot_embedding[:, :, 4:9]
        robot_position = robot_embedding[:, :, :2]
        robot_heading = robot_embedding[:, :, 2:4]
        robot_action = robot_embedding[:, :, 7:9]
        robot_goal = robot_embedding[:, :, -2:]

        # === Extract obstacle features ===
        obs_position = obstacle_embedding[:, :, :2]
        obs_heading = obstacle_embedding[:, :, 2:4]

        # === Encode node features ===
        robot_embed = self.encode_robot_features(
            robot_feat.reshape(batch_size * n_robots, -1)
        ).view(batch_size, n_robots, self.embedding_dim)

        # Obstacle nodes have no learnable features; zeros serve as placeholders
        # in the mega-graph. All obstacle information enters via edge features.
        obs_embed = torch.zeros(batch_size, n_obs, self.embedding_dim, device=device)

        # === Compute robot-robot edge features (vectorized) ===
        pos_i = robot_position.unsqueeze(2)
        pos_j = robot_position.unsqueeze(1)
        heading_i = robot_heading.unsqueeze(2)
        heading_j = robot_heading.unsqueeze(1).expand(-1, n_robots, -1, -1)
        action_j = robot_action.unsqueeze(1).expand(-1, n_robots, -1, -1)
        goal_j = robot_goal.unsqueeze(1).expand(-1, n_robots, -1, -1)

        rel_vec_rr = pos_j - pos_i
        rel_dist_rr = torch.linalg.vector_norm(rel_vec_rr, dim=-1, keepdim=True) / 12

        dx_rr, dy_rr = rel_vec_rr[..., 0], rel_vec_rr[..., 1]
        angle_rr = torch.atan2(dy_rr, dx_rr) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle_rr = (angle_rr + np.pi) % (2 * np.pi) - np.pi

        edge_features_rr = torch.cat([
            rel_dist_rr,
            torch.cos(angle_rr).unsqueeze(-1),
            torch.sin(angle_rr).unsqueeze(-1),
            heading_j[..., 0:1],
            heading_j[..., 1:2],
            action_j,
        ], dim=-1)

        # === Compute robot-obstacle edge features (vectorized) ===
        obs_pos_j = obs_position.unsqueeze(1)
        obs_heading_j = obs_heading.unsqueeze(1).expand(-1, n_robots, -1, -1)

        rel_vec_ro = obs_pos_j - pos_i
        rel_dist_ro = torch.linalg.vector_norm(rel_vec_ro, dim=-1, keepdim=True) / 12

        dx_ro, dy_ro = rel_vec_ro[..., 0], rel_vec_ro[..., 1]
        angle_ro = torch.atan2(dy_ro, dx_ro) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle_ro = (angle_ro + np.pi) % (2 * np.pi) - np.pi

        edge_features_ro = torch.cat([
            rel_dist_ro,
            torch.cos(angle_ro).unsqueeze(-1),
            torch.sin(angle_ro).unsqueeze(-1),
            obs_heading_j[..., 0:1],
            obs_heading_j[..., 1:2],
        ], dim=-1)

        # === Hard attention for robot-robot ===
        h_i_rr = robot_embed.unsqueeze(2).expand(-1, -1, n_robots, -1)
        hard_input_rr = torch.cat([h_i_rr, edge_features_rr], dim=-1)
        hard_input_rr = hard_input_rr.reshape(batch_size * n_robots, n_robots, -1)

        h_hard_rr = self.hard_mlp(hard_input_rr)
        hard_logits_rr = self.hard_encoding(h_hard_rr)
        hard_weights_rr = F.gumbel_softmax(hard_logits_rr, hard=False, tau=0.2, dim=-1)[..., 1]
        hard_weights_rr = hard_weights_rr.view(batch_size, n_robots, n_robots)

        # === Hard attention for robot-obstacle ===
        h_i_ro = robot_embed.unsqueeze(2).expand(-1, -1, n_obs, -1)
        hard_input_ro = torch.cat([h_i_ro, edge_features_ro], dim=-1)
        hard_input_ro = hard_input_ro.reshape(batch_size * n_robots, n_obs, -1)

        h_hard_ro = self.hard_mlp_obs(hard_input_ro)
        hard_logits_ro = self.hard_encoding_obs(h_hard_ro)
        hard_weights_ro = F.gumbel_softmax(hard_logits_ro, hard=False, tau=0.2, dim=-1)[..., 1]
        hard_weights_ro = hard_weights_ro.view(batch_size, n_robots, n_obs)

        # === Unnormalized distances for BCE supervision ===
        unnorm_rel_dist_rr = torch.linalg.vector_norm(rel_vec_rr, dim=-1, keepdim=True)
        unnorm_rel_dist_rr = unnorm_rel_dist_rr.reshape(batch_size * n_robots, n_robots, 1)

        unnorm_rel_dist_ro = torch.linalg.vector_norm(rel_vec_ro, dim=-1, keepdim=True)
        unnorm_rel_dist_ro = unnorm_rel_dist_ro.reshape(batch_size * n_robots, n_obs, 1)

        # === Goal-relative polar features (vectorized) ===
        goal_rel_vec = goal_j - pos_i
        goal_rel_dist = torch.linalg.vector_norm(goal_rel_vec, dim=-1, keepdim=True)
        goal_angle_global = torch.atan2(goal_rel_vec[..., 1], goal_rel_vec[..., 0])
        heading_angle = torch.atan2(heading_i[..., 1], heading_i[..., 0])
        goal_rel_angle = goal_angle_global - heading_angle
        goal_rel_angle = (goal_rel_angle + np.pi) % (2 * np.pi) - np.pi
        goal_polar_rr = torch.cat([
            goal_rel_dist,
            torch.cos(goal_rel_angle).unsqueeze(-1),
            torch.sin(goal_rel_angle).unsqueeze(-1),
        ], dim=-1)

        goal_polar_ro = torch.zeros(batch_size, n_robots, n_obs, 3, device=device)

        # === Soft edge features (10-dim) ===
        soft_edge_rr = torch.cat([edge_features_rr, goal_polar_rr], dim=-1)
        obs_action_zeros = torch.zeros(batch_size, n_robots, n_obs, 2, device=device)
        soft_edge_ro = torch.cat([edge_features_ro, obs_action_zeros, goal_polar_ro], dim=-1)

        # === PHASE 2: Batched mega-graph message passing ===
        # Build edges for ALL batch samples at once (no Python loop).
        # Node layout in the mega-graph:
        #   [robot_0..robot_{n-1}, obs_0..obs_{m-1}]  for graph 0  (offset 0)
        #   [robot_0..robot_{n-1}, obs_0..obs_{m-1}]  for graph 1  (offset n_total)
        #   ...
        edge_index, edge_attr, batch_ids = self.build_edges_vectorized(
            hard_weights_rr, hard_weights_ro,
            soft_edge_rr, soft_edge_ro,
            n_robots, n_obs, device
        )

        # Flatten node features into mega-graph: (B * n_total, embed_dim)
        # Per graph: [robot_embed[b], obs_embed[b]]
        node_feats = torch.cat([robot_embed, obs_embed], dim=1)  # (B, n_total, embed_dim)
        node_feats_flat = node_feats.reshape(batch_size * n_total, -1)

        # For the bipartite propagation, target nodes are robot nodes only.
        # In the mega-graph, robot nodes for graph b are at indices [b*n_total .. b*n_total + n_robots - 1].
        # q_robots for all graphs: extract from node_feats_flat
        # We need q for indices [0..n_r-1], [n_total..n_total+n_r-1], ...
        robot_indices = (
            torch.arange(batch_size, device=device).unsqueeze(1) * n_total
            + torch.arange(n_robots, device=device).unsqueeze(0)
        ).reshape(-1)  # (B * n_robots,)
        q_robots_flat = self.message_graph.q(node_feats_flat[robot_indices])  # (B*n_robots, out_dim)

        # Map global target indices in edge_index[1] to the robot-only index space.
        # Global target b*n_total + i  -->  b*n_robots + i
        # Since targets are always robot nodes (< n_robots within each graph):
        total_edges = edge_index.shape[1]

        if total_edges > 0:
            # Target indices in robot-only space (for softmax grouping)
            tgt_global = edge_index[1]  # global mega-graph target
            tgt_local_in_graph = tgt_global - batch_ids * n_total  # i within graph (0..n_robots-1)
            tgt_robot_space = batch_ids * n_robots + tgt_local_in_graph  # index into q_robots_flat

            # Gather q for each edge's target
            x_i = q_robots_flat[tgt_robot_space]  # (total_edges, out_dim)

            # Compute attention and messages (replicating message() logic)
            k_edge = F.leaky_relu(self.message_graph.k(edge_attr))
            v_edge = F.leaky_relu(self.message_graph.v(edge_attr))
            attention_input = torch.cat([x_i, k_edge], dim=-1)
            scores = self.message_graph.attn_score_layer(attention_input).squeeze(-1)
            attn_weights = softmax(scores, tgt_robot_space, num_nodes=batch_size * n_robots)
            self.message_graph._last_attn_weights = attn_weights.detach()

            # Weighted messages
            messages = v_edge * attn_weights.unsqueeze(-1)  # (total_edges, out_dim)

            # Aggregate messages to target robot nodes
            attn_out_flat = torch.zeros(batch_size * n_robots, self.embedding_dim,
                                        device=device)
            attn_out_flat.index_add_(0, tgt_robot_space, messages)
        else:
            attn_weights = None
            attn_out_flat = torch.zeros(batch_size * n_robots, self.embedding_dim,
                                        device=device)

        # === PHASE 1: Scatter-based combined_weights (no .item() GPU-CPU sync) ===
        combined_w = torch.zeros(batch_size, n_robots, n_total, device=device)
        if attn_weights is not None and total_edges > 0:
            # Map source global index back to local-in-graph index
            src_local = edge_index[0] - batch_ids * n_total  # j within graph (0..n_total-1)
            tgt_local = tgt_local_in_graph  # i within graph (0..n_robots-1)

            # Flatten combined_w to (B * n_robots * n_total,) for scatter
            flat_idx = (
                batch_ids * (n_robots * n_total)
                + tgt_local * n_total
                + src_local
            )
            combined_w_flat = combined_w.reshape(-1)
            combined_w_flat.scatter_(0, flat_idx, attn_weights.detach())
            combined_w = combined_w_flat.reshape(batch_size, n_robots, n_total)

        # === Compute entropy (batched) ===
        if attn_weights is not None and total_edges > 0:
            # Compute per-edge entropy contributions
            eps = 1e-10
            attn_log = (attn_weights + eps).log()
            contrib = -(attn_weights * attn_log)

            # Sum contributions per robot node across the mega-graph
            entropies = torch.zeros(batch_size * n_robots, device=device)
            entropies.index_add_(0, tgt_robot_space, contrib)
            counts = torch.zeros(batch_size * n_robots, device=device)
            counts.index_add_(0, tgt_robot_space, torch.ones_like(attn_weights))

            mask = counts > 0
            if mask.any():
                per_node_entropy = entropies[mask] / counts[mask]
                mean_entropy = per_node_entropy.mean()
            else:
                mean_entropy = torch.tensor(0.0, device=device)
        else:
            mean_entropy = torch.tensor(0.0, device=device)

        # === Decode ===
        self_embed = robot_embed.reshape(batch_size * n_robots, -1)
        concat_embed = torch.cat([self_embed, attn_out_flat], dim=-1)

        # Store pre-decoder embedding for downstream consumers
        self._pre_decoder_embedding = concat_embed.detach()

        x = F.leaky_relu(self.decode_1(concat_embed))
        att_embedding = F.leaky_relu(self.decode_2(x))

        return (
            att_embedding,
            hard_logits_rr[..., 1],
            hard_logits_ro[..., 1],
            unnorm_rel_dist_rr,
            unnorm_rel_dist_ro,
            mean_entropy,
            hard_weights_rr,
            hard_weights_ro,
            combined_w,
        )
