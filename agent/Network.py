import numpy as np
import pandas as pd
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = 'cuda'
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

class ConvLayer(nn.Module):
    def __init__(self, node_fea_len, edge_fea_len):
        super(ConvLayer, self).__init__()
        self.node_fea_len = node_fea_len      # Length of node feature vector
        self.edge_fea_len = edge_fea_len      # Length of edge feature vector

        # Fully connected layer for message passing (gate + core)
        self.fc_full = nn.Linear(2 * self.node_fea_len + self.edge_fea_len,
                                 2 * self.node_fea_len).to(device)

        self.sigmoid = nn.Sigmoid()           # Used for gating function
        self.softplus = nn.Softplus()         # Activation for positive output

        # Learnable scaling parameter (for skip connection)
        self.alpha = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for linear and batchnorm layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He (Kaiming) initialization for linear layers
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        """
        Args:
            node_in_fea (Tensor): Node features, shape (N, F_node)
            edge_fea (Tensor): Edge features, shape (N, M, F_edge)
            edge_fea_idx (Tensor): Index of destination nodes, shape (N, M)

        Returns:
            out (Tensor): Updated node features, shape (N, F_node)
        """
        N, M = edge_fea_idx.shape

        # 1. Gather neighbor node features (destination node for each edge)
        node_edge_fea = node_in_fea[edge_fea_idx, :]  # shape (N, M, F_node)

        # 2. Concatenate: [central node | neighbor node | edge feature]
        total_nbr_fea = torch.cat([
            node_in_fea.unsqueeze(1).expand(N, M, self.node_fea_len),
            node_edge_fea,
            edge_fea
        ], dim=2)  # shape (N, M, 2*F_node + F_edge)

        # 3. Apply shared linear layer to compute gated message
        total_gated_fea = self.fc_full(total_nbr_fea)  # shape (N, M, 2*F_node)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # split into gate & core

        nbr_filter = self.sigmoid(nbr_filter)          # Gate values in [0, 1]
        nbr_core = self.softplus(nbr_core)             # Non-negative transformed message

        # 4. Apply mask to remove invalid edges (edge_fea_idx == -1)
        mask = torch.where(edge_fea_idx < 0, torch.tensor(0), torch.tensor(1))  # shape (N, M)
        mask = mask.unsqueeze(2)  # shape (N, M, 1)

        nbr_filter = nbr_filter * mask
        nbr_core = nbr_core * mask

        # 5. Aggregate messages from neighbors
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # shape (N, F_node)

        # 6. Combine with residual connection (scaled input + sum)
        out = self.softplus(self.alpha * node_in_fea + nbr_sumed)  # shape (N, F_node)
        return out


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim // num_heads
        # Linear transformation for input feature
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.zeros(size=(num_heads, self.out_dim)))
        self.a_dst = nn.Parameter(torch.zeros(size=(num_heads, self.out_dim)))
        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, X,A):
        """
        A: Adjacency matrix (b, n, n)
        X: Node features (b, n, f)
        """
        b, n, _ = X.shape  # Batch size, number of nodes
        # Apply linear transformation
        X_trans = self.W(X)  # (b, n, out_dim)
        # Multi-head attention
        X_split = X_trans.view(b, n, self.num_heads, self.out_dim)  # (b, n, heads, out_dim)
        # Compute attention coefficients
        attn_src = torch.einsum("bnhd,hd->bnh", X_split, self.a_src)  # (b, n, heads)
        attn_dst = torch.einsum("bnhd,hd->bnh", X_split, self.a_dst)  # (b, n, heads)
        attn_matrix = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (b, n, n, heads)
        attn_matrix = F.leaky_relu(attn_matrix, negative_slope=0.2)
        # Mask out non-existing edges (use adjacency matrix)
        attn_matrix = attn_matrix.masked_fill(A.unsqueeze(-1) == 0, float("-inf"))
        # Apply softmax normalization
        attn_matrix = F.softmax(attn_matrix, dim=2)
        # attn_matrix = self.dropout(attn_matrix)  # (b, n, n, heads)
        # Apply attention mechanism
        out = torch.einsum("bnnk,bnkd->bnkd", attn_matrix, X_split)  # (b, n, heads, out_dim)
        # Concatenate multi-head results
        out = out.reshape(b, n, -1)  # (b, n, out_dim * heads)
        return out

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        # Linear transformation applied after neighbor aggregation
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        Args:
            x (Tensor): Node features, shape (N, F_in)
            adj (Tensor): Normalized adjacency matrix with self-loops, shape (N, N)

        Returns:
            out (Tensor): Updated node features, shape (N, F_out)
        """
        # 1. Aggregate neighbor features using adjacency matrix
        out = torch.matmul(adj, x)  # shape (N, F_in)

        # 2. Apply shared linear transformation
        out = self.linear(out)      # shape (N, F_out)

        return out


def normalize_adjacency(adj):
    """
    Computes the symmetric normalized adjacency matrix:  D^{-1/2} A D^{-1/2}

    Args:
        adj (Tensor): Raw adjacency matrix (N x N)

    Returns:
        Tensor: Symmetrically normalized adjacency matrix (N x N)
    """
    # Add self-loops to each node (A ← A + I)
    adj = adj + torch.eye(adj.size(0)).to(adj.device)

    # Compute degree vector (D)
    deg = torch.sum(adj, dim=1)  # shape: (N,)

    # Compute D^{-1/2}
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0  # handle divide-by-zero
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    # Return D^{-1/2} * A * D^{-1/2}
    return torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)

def build_adjacency_from_neighbors(neighbors, num_nodes):
    """
    Builds an unnormalized adjacency matrix from a list of neighbor indices.

    Args:
        neighbors (list of lists): neighbors[i] contains a list of neighbor indices for node i.
                                   Invalid neighbors should be marked as -1.
        num_nodes (int): Total number of nodes

    Returns:
        Tensor: Adjacency matrix (num_nodes x num_nodes), with 1s where edges exist.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for i in range(num_nodes):
        for j in neighbors[i]:
            if j != -1:         # Ignore invalid or padded neighbors
                adj[i, j] += 1  # Add edge (i → j)

    return adj

def build_node_edge_adjacency_and_features(neighbors, edge_features, node_features):
    """
    neighbors: (N, M) tensor
    edge_features: (N, M, F) tensor
    node_features: (N, F) tensor

    Returns:
    - adj: (N+E, N+E) adjacency matrix
    - features: (N+E, F) feature matrix
    """
    num_nodes, max_neighbors = neighbors.shape
    feature_dim = edge_features.shape[2]

    edge_list = []
    edge_feat_list = []

    for i in range(num_nodes):
        for m in range(max_neighbors):
            j = neighbors[i, m].item()
            if j != -1:
                edge_list.append((i, j))
                edge_feat_list.append(edge_features[i, m])  # shape: (F,)

    num_edges = len(edge_list)
    total_nodes = num_nodes + num_edges

    # 1. Adjacency matrix (N+E, N+E)
    adj = torch.zeros((total_nodes, total_nodes), dtype=torch.float32).to(device)

    for edge_id, (i, j) in enumerate(edge_list):
        e = num_nodes + edge_id
        adj[i, e] = 1
        adj[e, i] = 1
        adj[j, e] = 1
        adj[e, j] = 1

    # 2. Feature matrix (N+E, F)
    combined_features = torch.zeros((total_nodes, feature_dim), dtype=torch.float32).to(device)
    combined_features[:num_nodes] = node_features
    if edge_feat_list:
        combined_features[num_nodes:] = torch.stack(edge_feat_list)

    return adj, combined_features

class GAT(nn.Module):
    def __init__(self, original_node_fea_len, original_edge_fea_len, hidden_dim, N):
        super(GAT, self).__init__()

        # Node and edge feature projection to hidden dimension
        self.embedding_n = nn.Linear(original_node_fea_len, hidden_dim)
        self.embedding_e = nn.Linear(original_edge_fea_len, hidden_dim)

        # Stacked GAT layers (with 1 attention head each)
        self.gcn1 = GATLayer(hidden_dim, hidden_dim, num_heads=1)
        self.gcn2 = GATLayer(hidden_dim, hidden_dim, num_heads=1)
        self.gcn3 = GATLayer(hidden_dim, hidden_dim, num_heads=1)

        # Readout MLP after GAT layers
        self.conv_to_fc = nn.Linear(hidden_dim * N, 256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)

        self.act_fun = nn.ELU()  # Activation function for readout

    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        """
        Forward pass for GAT graph encoder.

        Args:
            node_in_fea (Tensor): Node input features, shape (N, F_node)
            edge_fea (Tensor): Edge input features, shape (N, M, F_edge)
            edge_fea_idx (Tensor): Destination indices for each edge, shape (N, M)

        Returns:
            Tensor: Updated node embeddings after GAT, shape (N, hidden_dim)
        """
        N, M = edge_fea_idx.shape

        # 1. Project node and edge features
        n = self.embedding_n(node_in_fea)      # shape: (N, hidden_dim)
        e = self.embedding_e(edge_fea)         # shape: (N, M, hidden_dim)

        # 2. Construct graph adjacency and combined features
        adj, x = build_node_edge_adjacency_and_features(edge_fea_idx, e, n)
        adj = normalize_adjacency(adj).unsqueeze(0)  # shape: (1, N, N)
        x = x.unsqueeze(0)                           # shape: (1, N, hidden_dim)

        # 3. Apply GAT layers
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        x = self.gcn3(x, adj).squeeze(0)  # shape: (N, hidden_dim)

        return x[:N]  # Return node embeddings only

    def readout(self, node_fea):
        """
        Readout function to map node embeddings to scalar output (e.g., for scoring or classification).

        Args:
            node_fea (Tensor): Node embeddings, shape (B, N, hidden_dim)

        Returns:
            Tensor: Scalar prediction per batch, shape (B, 1)
        """
        B, N, M = node_fea.shape

        # Flatten node features and pass through feedforward layers
        node_fea = self.conv_to_fc(node_fea.view(B, -1))
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)
        out = self.fc_out(node_fea)

        return out

class GCN1(nn.Module):
    def __init__(self, original_node_fea_len, hidden_dim, N):
        super(GCN1, self).__init__()

        # Linear embedding from input dimension to hidden dimension
        self.embedding = nn.Linear(original_node_fea_len, hidden_dim)

        # 3-layer Graph Convolutional Network (GCN)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)

        # Readout layers after GCN
        self.conv_to_fc = nn.Linear(hidden_dim * N, 256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)

        self.act_fun = nn.ELU()

    def forward(self, node_in_fea, edge_fea_idx):
        """
        Forward pass through GCN layers.

        Args:
            node_in_fea (Tensor): Node features, shape (N, F_node)
            edge_fea_idx (Tensor): Neighbor index list, shape (N, M)

        Returns:
            Tensor: Node embeddings, shape (N, hidden_dim)
        """
        N, M = edge_fea_idx.shape

        # Build adjacency matrix from neighbor index list
        adj = build_adjacency_from_neighbors(edge_fea_idx, N)
        adj = normalize_adjacency(adj).to(device)  # shape (N, N)

        # Apply input embedding
        x = self.embedding(node_in_fea)  # shape (N, hidden_dim)

        # Apply stacked GCN layers
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        x = self.gcn3(x, adj)

        return x  # shape (N, hidden_dim)

    def readout(self, node_fea):
        """
        Readout: aggregates all node embeddings and maps to scalar prediction.

        Args:
            node_fea (Tensor): Node embeddings, shape (B, N, hidden_dim)

        Returns:
            Tensor: Output per batch, shape (B, 1)
        """
        B, N, M = node_fea.shape

        # Flatten and pass through MLP
        node_fea = self.conv_to_fc(node_fea.view(B, -1))
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)
        out = self.fc_out(node_fea)

        return out

class GCN2(nn.Module):
    def __init__(self, original_node_fea_len, original_edge_fea_len, hidden_dim, N):
        super(GCN2, self).__init__()

        # Linear projection of input features to hidden space
        self.embedding_n = nn.Linear(original_node_fea_len, hidden_dim)
        self.embedding_e = nn.Linear(original_edge_fea_len, hidden_dim)

        # Stacked GCN layers (3 layers)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)

        # Fully connected readout layers
        self.conv_to_fc = nn.Linear(hidden_dim * N, 256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)

        self.act_fun = nn.ELU()

    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        """
        Forward pass through edge-enhanced GCN.

        Args:
            node_in_fea (Tensor): Node features, shape (N, F_node)
            edge_fea (Tensor): Edge features, shape (N, M, F_edge)
            edge_fea_idx (Tensor): Neighbor indices, shape (N, M)

        Returns:
            Tensor: Node embeddings, shape (N, hidden_dim)
        """
        N, M = edge_fea_idx.shape

        # Project node and edge features
        n = self.embedding_n(node_in_fea)      # (N, hidden_dim)
        e = self.embedding_e(edge_fea)         # (N, M, hidden_dim)

        # Build adjacency matrix and combined node feature matrix
        adj, x = build_node_edge_adjacency_and_features(edge_fea_idx, e, n)
        adj = normalize_adjacency(adj)

        # Apply GCN layers
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        x = self.gcn3(x, adj)

        # Return first N node embeddings (excluding edge nodes)
        return x[:N]

    def readout(self, node_fea):
        """
        Readout function that aggregates node embeddings into a scalar output.

        Args:
            node_fea (Tensor): Node embeddings, shape (B, N, hidden_dim)

        Returns:
            Tensor: Output per batch, shape (B, 1)
        """
        B, N, M = node_fea.shape

        # Flatten node features and apply MLP
        node_fea = self.conv_to_fc(node_fea.view(B, -1))
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)
        out = self.fc_out(node_fea)

        return out

class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_node_fea_len, orig_edge_fea_len, edge_fea_len, node_fea_len,
                 final_node_len, dis):
        super(CrystalGraphConvNet, self).__init__()

        # Input projection for node and edge features
        self.embedding_n = nn.Linear(orig_node_fea_len, node_fea_len).to(device)
        self.embedding_e = nn.Linear(orig_edge_fea_len, edge_fea_len).to(device)

        self.dis = dis  # distance matrix, possibly for future use
        N = dis.shape[0]

        # Stacked graph convolution layers (message passing layers)
        self.convs1 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs2 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs3 = ConvLayer(node_fea_len, edge_fea_len)

        # Linear transformation after convolution
        self.final_layer = nn.Linear(node_fea_len, int(final_node_len / 2)).to(device)

        # Fully-connected layers for graph-level prediction
        self.conv_to_fc = nn.Linear(final_node_len * N, 256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)

        # Distance-aware attention (optional, learnable modulation)
        self.DA_weight = nn.Parameter(torch.tensor(48 / 5, dtype=torch.float32))  # learnable scalar
        self.DA_bias = nn.Parameter(torch.tensor(-28 / 5, dtype=torch.float32))   # learnable bias
        self.DA_act = nn.Sigmoid()

        self.act_fun = nn.ELU()

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Apply He initialization to all Linear and BatchNorm layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, node_fea, edge_fea, edge_fea_idx):
        """
        Forward pass through graph convolution layers.

        Args:
            node_fea (Tensor): Input node features, shape (N, F_node)
            edge_fea (Tensor): Input edge features, shape (N, M, F_edge)
            edge_fea_idx (Tensor): Edge index tensor, shape (N, M)

        Returns:
            Tensor: Node embeddings, shape (N, F_hidden)
        """
        # Project node and edge features to hidden dimensions
        node_fea = self.embedding_n(node_fea)      # shape: (N, node_fea_len)
        edge_fea = self.embedding_e(edge_fea)      # shape: (N, M, edge_fea_len)

        # Apply 3 stacked message-passing layers
        node_fea = self.convs1(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs2(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs3(node_fea, edge_fea, edge_fea_idx)

        return node_fea  # shape: (N, node_fea_len)

    def readout(self, node_fea):
        """
        Readout function to aggregate node embeddings into scalar graph-level output.

        Args:
            node_fea (Tensor): Node embeddings, shape (B, N, node_fea_len)

        Returns:
            Tensor: Output value per graph, shape (B, 1)
        """
        B, N, M = node_fea.shape

        # Flatten all node embeddings and pass through MLP
        node_fea = self.conv_to_fc(node_fea.view(B, -1))
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)

        out = self.fc_out(node_fea)
        return out



class MLP(nn.Module):
    def __init__(self, state_size, output_size):
        super(MLP, self).__init__()

        # Fully-connected feedforward network layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

        # Apply weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all Linear and BatchNorm layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He (Kaiming) initialization for better ReLU convergence
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # Optional BatchNorm weight initialization
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch_size, state_size)

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))  # Layer 1 + ReLU
        x = F.relu(self.fc2(x))  # Layer 2 + ReLU
        x = F.relu(self.fc3(x))  # Layer 3 + ReLU
        x = self.fc4(x)          # Final output layer (no activation)
        return x
class PPO(nn.Module):
    def __init__(self, learning_rate, lmbda, gamma, alpha, beta, epsilon, discount_factor,
                 location_num, transporter_type, dis, gnn_mode):
        super(PPO, self).__init__()
        """
                Initialize PPO agent with selected GNN encoder and hyperparameters.

                Args:
                    learning_rate (float): Learning rate for the optimizer.
                    lmbda (float): Lambda for GAE (Generalized Advantage Estimation).
                    gamma (float): Discount factor.
                    alpha (float): Value loss coefficient.
                    beta (float): Entropy bonus coefficient (unused).
                    epsilon (float): Clipping parameter for PPO.
                    discount_factor (float): General reward discount factor.
                    location_num (int): Not used directly here.
                    transporter_type (int): Number of transporter types.
                    dis (Tensor): Precomputed distance matrix for node embedding.
                    gnn_mode (str): GNN backbone type ('CGCNN', 'GCN1', 'GCN2', 'GAT').
        """
        # Hyperparameters
        self.transporter_type = transporter_type
        self.node_fea_len = 32
        self.final_node_len = 32
        self.edge_fea_len = 32
        self.gnn_mode = gnn_mode

        # Select GNN model
        if gnn_mode == 'CGCNN':
            self.gnn = CrystalGraphConvNet(
                orig_node_fea_len=int(2 * self.transporter_type),
                orig_edge_fea_len=int(3 + self.transporter_type),
                edge_fea_len=self.edge_fea_len,
                node_fea_len=self.node_fea_len,
                final_node_len=32,
                dis=dis
            )
        elif gnn_mode == 'GCN1':
            self.gnn = GCN1(int(2 * self.transporter_type), 32, dis.shape[0])
        elif gnn_mode == 'GCN2':
            self.gnn = GCN2(int(2 * self.transporter_type), int(3 + self.transporter_type), 32, dis.shape[0])
        elif gnn_mode == 'GAT':
            self.gnn = GAT(int(2 * self.transporter_type), int(3 + self.transporter_type), 32, dis.shape[0])

        # Policy network
        self.pi = MLP(32 + 2 * int(3 + self.transporter_type) + 5 + 5, 1).to(device)
        self.temperature = 1.0
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # PPO parameters
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def calculate_GNN(self, node_fea, edge_fea, edge_fea_idx):
        """Forward node and edge features through selected GNN encoder"""
        # Forward through the GNN model
        if self.gnn_mode == 'CGCNN':
            return self.gnn(node_fea, edge_fea, edge_fea_idx)
        if self.gnn_mode == 'GCN1':
            return self.gnn(node_fea, edge_fea_idx)
        if self.gnn_mode in ['GCN2', 'GAT']:
            return self.gnn(node_fea, edge_fea, edge_fea_idx)

    def calculate_pi(self, state_gnn, node_fea, edge_fea, edge_fea_idx, distance, tp_type):
        """
                Compute action logits for each edge candidate.

                Args:
                    state_gnn: Output of GNN for each node.
                    node_fea, edge_fea, edge_fea_idx: Graph inputs.
                    distance: Precomputed distance matrix.
                    tp_type: Transporter type ID.

                Returns:
                    action_probability: Output logits for each edge candidate.
        """
        # Compute action logits using the policy MLP
        action_variable = state_gnn[edge_fea_idx, :]
        edge_fea_tensor = edge_fea.repeat(1, 1, 2)
        distance_tensor = distance.unsqueeze(2).repeat(1, 1, 5)
        tp_tensor = torch.full((edge_fea_idx.shape[0], edge_fea_idx.shape[1], 5),
                               float(tp_type) / self.transporter_type).to(device)

        action_variable = torch.cat([action_variable, edge_fea_tensor, distance_tensor, tp_tensor], dim=2)
        action_probability = self.pi(action_variable)
        return action_probability

    def get_action(self, node_fea, edge_fea, edge_fea_idx, mask, distance, tp_type):
        """
               Sample an action from the policy with masking invalid entries.

               Returns:
                   action (int), i (int), j (int), selected probability (Tensor)
        """
        # Select action based on current policy
        with torch.no_grad():
            N, M = edge_fea_idx.shape
            state = self.calculate_GNN(node_fea, edge_fea, edge_fea_idx)
            probs = self.calculate_pi(state, node_fea, edge_fea, edge_fea_idx, distance, tp_type)
            logits_masked = probs - 1e8 * mask
            prob = torch.softmax((logits_masked.flatten() - torch.max(logits_masked.flatten())) / self.temperature, dim=-1)
            m = Categorical(prob)
            action = m.sample().item()
            i = int(action / M)
            j = int(action % M)
            while edge_fea_idx[i][j] < 0:
                action = m.sample().item()
                i = int(action / M)
                j = int(action % M)
            return action, i, j, prob[action]

    def calculate_v(self, x):
        """Estimate value function from GNN output"""
        # Value estimation using GNN readout
        return self.gnn.readout(x)

    def update(self, data, probs, rewards, action, dones, step1, validation_step, model_dir):
        """
                Perform PPO update using collected episode data.

                Args:
                    data: List of episodes (list of state tuples).
                    probs: Old action probabilities.
                    rewards: Collected rewards.
                    action: Taken actions.
                    dones: Done signals.
                    step1: Global training step.
                    validation_step: Model validation & saving step
                    model_dir: Path for saving models.

                Returns:
                    ave_loss, v_loss, p_loss (float): Averaged total, value, and policy losses.
        """
        # Perform PPO update with collected episode data
        probs = torch.tensor(probs, dtype=torch.float32).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        step = 0
        sw = 0
        tr = 0
        pi_a_total = None
        state_GNN = None
        next_state_GNN = None

        # Compute log π(a|s) for all time steps
        for episode in data:
            for state in episode:
                state_gnn = self.calculate_GNN(state[0], state[1], state[2])
                if step < len(episode) - 1:
                    prob_a = self.calculate_pi(state_gnn, state[0], state[1], state[2], state[3], state[4])
                    mask = state[5]
                    logits_maksed = prob_a - 1e8 * mask
                    prob = torch.softmax(
                        (logits_maksed.flatten() - torch.max(logits_maksed.flatten())) / self.temperature, dim=-1)
                    pi_a = prob[int(action[sw])]
                    sw += 1
                    if tr == 0:
                        pi_a_total = pi_a.unsqueeze(0)
                    else:
                        pi_a_total = torch.cat([pi_a_total, pi_a.unsqueeze(0)])
                state_gnn = state_gnn.unsqueeze(0)
                if tr == 0:
                    state_GNN = state_gnn
                elif tr == 1:
                    next_state_GNN = state_gnn
                    state_GNN = torch.cat([state_GNN, state_gnn])
                elif step == 0:
                    state_GNN = torch.cat([state_GNN, state_gnn])
                elif step == len(episode) - 1:
                    next_state_GNN = torch.cat([next_state_GNN, state_gnn])
                else:
                    state_GNN = torch.cat([state_GNN, state_gnn])
                    next_state_GNN = torch.cat([next_state_GNN, state_gnn])
                tr += 1
                step += 1
            step = 0

        total_time_step = sw
        state_v = self.calculate_v(state_GNN)
        state_next_v = self.calculate_v(next_state_GNN)
        td_target = rewards + self.gamma * state_next_v * dones
        delta = td_target - state_v

        # Compute advantages using GAE
        advantage_lst = torch.zeros(total_time_step, 1).to(device)
        i = 0
        for e, episode in enumerate(data):
            if e < 1:
                advantage = 0.0
                for t in reversed(range(i, i + len(episode) - 1)):
                    advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                    advantage_lst[t][0] = advantage
                i += len(episode) - 1

        # PPO loss
        ratio = torch.exp(torch.log(pi_a_total.unsqueeze(1)) - torch.log(probs))
        surr1 = ratio * advantage_lst
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_v, td_target.detach()) * self.alpha

        ave_loss = loss.mean().item()
        v_loss = (self.alpha * F.smooth_l1_loss(state_v, td_target.detach())).item()
        p_loss = -torch.min(surr1, surr2).mean().item()

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # Save model periodically
        if step1 % validation_step == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_dir + 'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss



