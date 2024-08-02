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
        """
        Initialize the ConvLayer class.

        Parameters:
        - node_fea_len (int): Length of the node feature vector.
        - edge_fea_len (int): Length of the edge feature vector.
        """
        super(ConvLayer, self).__init__()
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len

        # Fully connected layer for feature transformation
        self.fc_full = nn.Linear(2 * self.node_fea_len + self.edge_fea_len, 2 * self.node_fea_len).to(device)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.alpha = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the network using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        """
        Forward pass through the convolution layer.

        Parameters:
        - node_in_fea (torch.Tensor): Input node features.
        - edge_fea (torch.Tensor): Input edge features.
        - edge_fea_idx (torch.Tensor): Edge feature indices.

        Returns:
        - out (torch.Tensor): Output node features after convolution.
        """
        N, M = edge_fea_idx.shape
        # Convolution operation
        node_edge_fea = node_in_fea[edge_fea_idx, :]  # Edge feature indices for start and end nodes
        total_nbr_fea = torch.cat([node_in_fea.unsqueeze(1).expand(N, M, self.node_fea_len), node_edge_fea, edge_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        mask = torch.where(edge_fea_idx < 0, torch.tensor(0), torch.tensor(1))
        nbr_filter = nbr_filter * mask.unsqueeze(2)
        nbr_core = nbr_filter * mask.unsqueeze(2)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        out = self.softplus(self.alpha * node_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_node_fea_len, orig_edge_fea_len, edge_fea_len, node_fea_len, final_node_len, dis):
        """
        Initialize the CrystalGraphConvNet class.

        Parameters:
        - orig_node_fea_len (int): Original length of the node feature vector.
        - orig_edge_fea_len (int): Original length of the edge feature vector.
        - edge_fea_len (int): Length of the edge feature vector.
        - node_fea_len (int): Length of the node feature vector.
        - final_node_len (int): Final length of the node feature vector.
        - dis (torch.Tensor): Distance matrix.
        """
        super(CrystalGraphConvNet, self).__init__()
        self.embedding_n = nn.Linear(orig_node_fea_len, node_fea_len).to(device)
        self.embedding_e = nn.Linear(orig_edge_fea_len, edge_fea_len).to(device)
        self.dis = dis
        N = dis.shape[0]
        self.convs1 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs2 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs3 = ConvLayer(node_fea_len, edge_fea_len)
        self.final_layer = nn.Linear(node_fea_len, int(final_node_len / 2)).to(device)
        self.conv_to_fc = nn.Linear(final_node_len * N, 256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)
        self.DA_weight = nn.Parameter(torch.tensor(48 / 5, dtype=torch.float32))
        self.DA_bias = nn.Parameter(torch.tensor(-28 / 5, dtype=torch.float32))
        self.DA_act = nn.Sigmoid()
        self.act_fun = nn.ELU()

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the network using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, node_fea, edge_fea, edge_fea_idx):
        """
        Forward pass through the CrystalGraphConvNet.

        Parameters:
        - node_fea (torch.Tensor): Input node features.
        - edge_fea (torch.Tensor): Input edge features.
        - edge_fea_idx (torch.Tensor): Edge feature indices.

        Returns:
        - node_fea (torch.Tensor): Output node features after convolution layers.
        """
        node_fea = self.embedding_n(node_fea)
        edge_fea = self.embedding_e(edge_fea)
        node_fea = self.convs1(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs2(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs3(node_fea, edge_fea, edge_fea_idx)
        return node_fea

    def readout(self, node_fea):
        """
        Readout function to aggregate node features.

        Parameters:
        - node_fea (torch.Tensor): Node features.

        Returns:
        - out (torch.Tensor): Aggregated node features.
        """
        B, N, M = node_fea.shape
        node_fea = self.conv_to_fc(node_fea.view(B, -1))  # Flatten and pass through FC layer
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)
        out = self.fc_out(node_fea)
        return out


class MLP(nn.Module):
    def __init__(self, state_size, output_size):
        """
        Initialize the MLP class.

        Parameters:
        - state_size (int): Size of the input state vector.
        - output_size (int): Size of the output vector.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the network using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - x (torch.Tensor): Output tensor after passing through the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PPO(nn.Module):
    def __init__(self, learning_rate, lmbda, gamma, alpha, beta, epsilon, discount_factor, location_num, dis):
        """
        Initialize the PPO class.

        Parameters:
        - learning_rate (float): Learning rate for the optimizer.
        - lmbda (float): Lambda for GAE.
        - gamma (float): Discount factor.
        - alpha (float): Weight for the value loss.
        - beta (float): Weight for the entropy loss.
        - epsilon (float): Clipping parameter for PPO.
        - discount_factor (float): Discount factor for rewards.
        - location_num (int): Number of locations.
        - dis (torch.Tensor): Distance matrix.
        """
        super(PPO, self).__init__()
        self.node_fea_len = 32
        self.final_node_len = 32
        self.edge_fea_len = 32
        self.gnn = CrystalGraphConvNet(orig_node_fea_len=4, orig_edge_fea_len=5, edge_fea_len=self.edge_fea_len, node_fea_len=self.node_fea_len, final_node_len=32, dis=dis)
        self.pi = MLP(32 + 10 + 5 + 5, 1).to(device)
        self.temperature = 1
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.beta = beta
        self.epsilon = epsilon

    def calculate_GNN(self, node_fea, edge_fea, edge_fea_idx):
        """
        Calculate the output of the GNN.

        Parameters:
        - node_fea (torch.Tensor): Node features.
        - edge_fea (torch.Tensor): Edge features.
        - edge_fea_idx (torch.Tensor): Edge feature indices.

        Returns:
        - torch.Tensor: Output of the GNN.
        """
        return self.gnn(node_fea, edge_fea, edge_fea_idx)

    def calculate_pi(self, state_gnn, node_fea, edge_fea, edge_fea_idx, distance, tp_type):
        """
        Calculate the policy.

        Parameters:
        - state_gnn (torch.Tensor): GNN state.
        - node_fea (torch.Tensor): Node features.
        - edge_fea (torch.Tensor): Edge features.
        - edge_fea_idx (torch.Tensor): Edge feature indices.
        - distance (torch.Tensor): Distance matrix.
        - tp_type (float): Transporter type.

        Returns:
        - torch.Tensor: Action probabilities.
        """
        action_variable = state_gnn[edge_fea_idx, :]
        edge_fea_tensor = edge_fea.repeat(1, 1, 2)
        distance_tensor = distance.unsqueeze(2).repeat(1, 1, 5)
        action_variable = torch.cat([action_variable, edge_fea_tensor], 2)
        action_variable = torch.cat([action_variable, distance_tensor], 2)
        action_variable = torch.cat((action_variable, torch.full((edge_fea_idx.shape[0], edge_fea_idx.shape[1], 5), tp_type).to(device)), dim=2)
        action_probability = self.pi(action_variable)
        return action_probability

    def get_action(self, node_fea, edge_fea, edge_fea_idx, mask, distance, tp_type):
        """
        Get an action from the policy.

        Parameters:
        - node_fea (torch.Tensor): Node features.
        - edge_fea (torch.Tensor): Edge features.
        - edge_fea_idx (torch.Tensor): Edge feature indices.
        - mask (torch.Tensor): Mask for invalid actions.
        - distance (torch.Tensor): Distance matrix.
        - tp_type (float): Transporter type.

        Returns:
        - action (int): Selected action.
        - i (int): Start node index.
        - j (int): End node index.
        - prob (torch.Tensor): Probability of the selected action.
        """
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
        """
        Calculate the value function.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Value function output.
        """
        return self.gnn.readout(x)

    def update(self, data, probs, rewards, action, dones, step1, model_dir):
        """
        Update the policy and value function.

        Parameters:
        - data (list): List of episodes.
        - probs (np.ndarray): Action probabilities.
        - rewards (np.ndarray): Rewards.
        - action (np.ndarray): Actions.
        - dones (np.ndarray): Done flags.
        - step1 (int): Current step.
        - model_dir (str): Directory to save the model.

        Returns:
        - ave_loss (float): Average loss.
        - v_loss (float): Value loss.
        - p_loss (float): Policy loss.
        """
        num = 0
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        p_loss = 0
        probs = torch.tensor(probs, dtype=torch.float32).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        tr = 0
        step = 0
        sw = 0
        for episode in data:
            for state in episode:
                state_gnn = self.calculate_GNN(state[0], state[1], state[2])

                if step < len(episode) - 1:
                    prob_a = self.calculate_pi(state_gnn, state[0], state[1], state[2], state[3], state[4])
                    mask = state[5]
                    logits_maksed = prob_a - 1e8 * mask
                    prob = torch.softmax((logits_maksed.flatten() - torch.max(logits_maksed.flatten())) / self.temperature, dim=-1)
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

        advantage_lst = np.zeros(total_time_step)
        advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).unsqueeze(1).to(device)
        for episode in data:
            advantage = 0.0
            i = 0
            for t in reversed(range(i, i + len(episode))):
                advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                advantage_lst[t][0] = advantage
            i += len(episode)
        ratio = torch.exp(torch.log(pi_a_total.unsqueeze(1)) - torch.log(probs))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage_lst
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_v, td_target.detach()) * self.alpha

        ave_loss = loss.mean().item()
        v_loss = (self.alpha * F.smooth_l1_loss(state_v, td_target.detach())).item()
        p_loss = -torch.min(surr1, surr2).mean().item()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        if step1 % 10 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_dir + 'trained_model' + str(step1) + '.pth')

        return ave_loss, v_loss, p_loss
