from Network_DAN import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import random
import matplotlib.pyplot as plt
device='cuda'
import numpy as np
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
import math
class Problem_sampling:
    def __init__(self, block_number, location_number, transporter_type, transporter_number,
                 dis_high, dis_low, ready_high, tardy_high, gap):
        """
        Generates a synthetic block transport scheduling problem instance.

        This class creates:
            - A symmetric distance matrix between locations.
            - Block information including origin, destination, processing time, ready time, due time, and transporter type.
            - Transporter state initialization.
            - Feature tensors for graph-based learning or heuristic scheduling.

        Attributes:
            Block_Number (int): Total number of blocks.
            Location_Number (int): Number of unique locations.
            Transporter_type (int): Number of transporter types.
            Transporter_Number (int): Total number of transporters.
            Dis (ndarray): Symmetric distance matrix between locations.
            ready_high (int): Maximum ready time (used for sampling).
            tardy_high (int): Maximum due time (used for normalization).
            gap (int): Minimum gap between ready and due time.

        Methods:
            sample():
                Samples block and transporter information and constructs:
                    - Block matrix with processing time, ready/due time, and transporter type encoding.
                    - Transporter initial states.
                    - Edge and node features for graph input.
                Returns all required data structures for simulation or learning.
        """

        # Problem parameters
        self.Block_Number = block_number                      # Number of blocks
        self.Location_Number = location_number                # Number of locations
        self.Transporter_type = transporter_type              # Number of transporter types
        self.Transporter_Number = transporter_number          # Total number of transporters

        # Generate a symmetric distance matrix
        upper_tri = np.random.uniform(dis_low, dis_high, (location_number, location_number))
        upper_tri = np.triu(upper_tri, 1)
        symmetric_matrix = upper_tri + upper_tri.T
        np.fill_diagonal(symmetric_matrix, 0)
        self.Dis = symmetric_matrix.copy()                    # Distance matrix between locations

        # Timing parameters
        self.ready_high = ready_high                          # Upper bound for ready time
        self.tardy_high = tardy_high                          # Upper bound for due time
        self.dis_high = dis_high                              # Max distance (used elsewhere if needed)
        self.gap = gap                                        # Minimum time gap between ready and due

    def sample(self):
        # Initialize block and transporter data structures
        Block = np.zeros((self.Block_Number, 5 + self.Transporter_type))  # Block info array
        transporter = np.zeros((self.Transporter_Number, 6))              # Transporter info array

        # Sample block information
        for i in range(self.Block_Number):
            v = np.random.choice(self.Location_Number, 2, False)
            Block[i, 0], Block[i, 1] = v[0], v[1]  # Origin and destination
            Block[i, 2] = self.Dis[int(Block[i, 0]), int(Block[i, 1])] / 80 / self.tardy_high  # Processing time
            Block[i, 3] = np.random.randint(0, self.ready_high) / self.tardy_high             # Ready time
            Block[i, 4] = (np.random.randint(self.gap, self.tardy_high) / self.tardy_high) - Block[i, 2]  # Due time

            # Randomly assign weight and set transporter type one-hot encoding
            weight = np.random.uniform(1, 50 * self.Transporter_type)
            temp_type = int(weight / 50)
            Block[i, 5:5 + temp_type] += 1  # One-hot encoding for transporter type requirement

        # Sort blocks by origin location
        Block = Block[Block[:, 0].argsort()]

        # Create edge features: block assignments per origin
        unique_values, counts = np.unique(Block[:, 0], return_counts=True)
        max_count = np.max(counts)
        edge_fea_idx = -np.ones((self.Location_Number, max_count))  # Destination index per origin
        edge_fea = np.zeros((self.Location_Number, max_count, 3 + self.Transporter_type))  # Features: [proc_time, ready, due, one-hot type]
        step = 0
        node_in_fea = np.zeros((self.Location_Number, 2 * self.Transporter_type))  # Initial transporter availability at each node
        step_to_ij = np.zeros((self.Location_Number, max_count))  # Mapping from (i,j) to block index

        # Populate edge feature arrays
        for i in range(len(counts)):
            for j in range(max_count):
                if j < counts[i]:
                    edge_fea_idx[int(unique_values[i])][j] = int(Block[step, 1])
                    edge_fea[int(unique_values[i])][j] = Block[step, 2:]  # [proc, ready, due, one-hot type]
                    step_to_ij[int(unique_values[i])][j] = step
                    step += 1

        # Set initial transporter count at location 0 (depot)
        for i in range(self.Transporter_type):
            node_in_fea[0, i * 2] = int(self.Transporter_Number / self.Transporter_type)

        # Initialize transporter states
        for i in range(self.Transporter_Number):
            transporter[i, 0] = int((i * self.Transporter_type) / self.Transporter_Number)  # Transporter type
            transporter[i, 1] = 0     # Heading location
            transporter[i, 2] = 0     # Time left to arrive
            transporter[i, 3] = 0     # Empty travel time
            transporter[i, 4] = -1    # Last action: source
            transporter[i, 5] = -1    # Last action: destination

        return self.Block_Number, self.Transporter_Number, Block, transporter,edge_fea_idx, node_in_fea, edge_fea, self.Dis, step_to_ij


def simulation(B, T, transporter, block, edge_fea_idx, node_fea, edge_fea,
               dis, step_to_ij, tardy_high, mode, ppo):
    """
        Simulates the block transport scheduling environment
        using either a heuristic or reinforcement learning (PPO) policy.

        Parameters:
            B (int): Number of blocks to transport.
            T (int): Number of transporters.
            transporter (ndarray): Initial states of the transporters (T x 6).
            block (ndarray): Block information matrix (B x [features]).
            edge_fea_idx (ndarray): Destination indices for each origin node.
            node_fea (ndarray): Transporter availability at each node.
            edge_fea (ndarray): Edge features (processing time, ready time, due time, one-hot transporter type).
            dis (ndarray): Distance matrix between locations.
            step_to_ij (ndarray): Mapping from (i, j) to block index.
            tardy_high (float): Normalization factor for time.
            mode (str): Dispatching strategy ('RL_full', 'Random', 'SSPT', 'SET', 'SRT', 'ATCS', 'MDD', 'COVERT').
            ppo (object): PPO agent used in RL mode.

        Returns:
            reward_sum (float): Total reward (travel time + tardiness).
            tardy_sum (float): Total accumulated tardiness.
            ett_sum (float): Total empty travel time.
            event (list): Event records for Gantt chart visualization.
            episode (list): Saved states for RL training.
            actions (ndarray): Chosen actions per step.
            probs (ndarray): Action probabilities from PPO (only in RL).
            rewards (ndarray): Immediate reward per step.
            dones (ndarray): Done flags (1 if step ends episode, else 0).
    """

    # Deep copy to avoid modifying input data
    transporter = transporter.copy()
    block = block.copy()
    edge_fea_idx = edge_fea_idx.copy()
    node_fea = node_fea.copy()
    edge_fea = edge_fea.copy()

    # Initialize environment
    event = []
    unvisited_num = B  # Number of unvisited blocks
    node_fea = torch.tensor(node_fea, dtype=torch.float32).to(device)
    edge_fea = torch.tensor(edge_fea, dtype=torch.float32).to(device)
    edge_fea_idx = torch.tensor(edge_fea_idx, dtype=torch.long).to(device)

    N = edge_fea_idx.shape[0]  # Number of nodes
    M = edge_fea_idx.shape[1]  # Max number of edges per node
    episode = []  # To store state transitions (for PPO training)
    probs = np.zeros(B)
    rewards = np.zeros(B)
    dones = np.ones(B)
    actions = np.zeros(B)

    # Performance metrics
    tardiness = 0
    reward_sum = 0
    tardy_sum = 0
    ett_sum = 0
    step = 0
    time = 0
    prob = 0
    num_valid_coords = 10

    # Select initial transporter agent
    agent = np.random.randint(0, int(T / 2))
    node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] -= 1  # Remove agent from node

    while True:
        start_location = transporter[agent][1]

        # Calculate empty travel time to all nodes
        distance = torch.tensor(dis[int(start_location)] / 120 / tardy_high,
                                dtype=torch.float32).unsqueeze(1).repeat(1, M).to(device)

        # Mode-based action selection
        if mode == 'RL_full':
            # Mask invalid edges (already visited or mismatched transporter type)
            valid_coords = ((edge_fea_idx >= 0) & (edge_fea[:, :, 3 + int(transporter[agent][0])] == 0)).nonzero()
            mask = np.ones((N, M, 1))
            for i in range(valid_coords.shape[0]):
                n, e = valid_coords[i]
                mask[n, e, 0] = 0
            mask = torch.tensor(mask).to(device)

            # Save current state and get action from PPO
            episode.append([node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(),
                            distance.clone(), transporter[agent][0], mask])
            action, i, j, prob = ppo.get_action(node_fea, edge_fea, edge_fea_idx, mask,
                                                distance, transporter[agent][0])

        # Heuristic modes
        else:
            valid_coords = ((edge_fea_idx >= 0) &
                            (edge_fea[:, :, 3 + int(transporter[agent][0])] == 0)).nonzero()

            if mode == 'Random':
                # Randomly choose from valid actions
                action = random.randint(0, valid_coords.shape[0] - 1)
                i, j = valid_coords[action]

            elif mode == 'SSPT':
                # Shortest Setup + Processing Time
                pt = np.array([
                    max(dis[int(start_location)][n] / 120 / tardy_high, edge_fea[n, e, 1].item()) +
                    edge_fea[n, e, 0].item()
                    for n, e in valid_coords
                ])
                action = np.random.choice(np.where(pt == pt.min())[0])
                i, j = valid_coords[action]

            elif mode == 'SET':
                # Shortest Empty Travel
                pt = np.array([dis[int(start_location)][n] / 120 / tardy_high for n, e in valid_coords])
                action = np.random.choice(np.where(pt == pt.min())[0])
                i, j = valid_coords[action]

            elif mode == 'SRT':
                # Shortest Ready Time
                pt = np.array([edge_fea[n, e, 1].item() for n, e in valid_coords])
                action = np.random.choice(np.where(pt == pt.min())[0])
                i, j = valid_coords[action]

            elif mode == 'ATCS':
                # Apparent Tardiness Cost with Setup
                pt_average = np.array([edge_fea[n, e, 0].cpu().item() for n, e in valid_coords ])
                st_average = np.array([dis[int(start_location)][n] / 120 / tardy_high for n, e in valid_coords])
                pt_a, st_a = pt_average.mean(), st_average.mean()

                pt = np.array([
                    (1 / edge_fea[n, e, 0] *
                     math.exp(-max(edge_fea[n, e, 2], 0) / pt_a) *
                     math.exp(-dis[int(start_location)][n] / 120 / tardy_high / st_a)).item()
                    for n, e in valid_coords
                ])
                action = np.random.choice(np.where(pt == pt.max())[0])
                i, j = valid_coords[action]

            elif mode == 'MDD':
                # Minimum Due Date
                pt = np.array([edge_fea[n, e, 2].item() for n, e in valid_coords])
                action = np.random.choice(np.where(pt == pt.min())[0])
                i, j = valid_coords[action]

            elif mode == 'COVERT':
                # Covert rule
                pt = np.array([
                    -(1 / edge_fea[n, e, 0] * (1 - edge_fea[n, e, 2] / edge_fea[n, e, 0])).item()
                    for n, e in valid_coords
                ])
                action = np.random.choice(np.where(pt == pt.min())[0])
                i, j = valid_coords[action]

        # Execute action and update system
        transporter, edge_fea_idx, node_fea, edge_fea, event_list, ett, td = do_action(
            transporter, edge_fea_idx.clone(), node_fea.clone(), edge_fea.clone(),
            agent, i, j, dis, time, step_to_ij, tardy_high)

        # Final block handling
        if unvisited_num == 1:
            event_list.extend([round(td, 3), round(ett, 3), round(td + ett, 3)])
            event.append(event_list)
            tardy_sum += td
            ett_sum += ett
            reward = td + ett
            reward_sum += reward
            actions[step] = action
            probs[step] = prob
            dones[step] = 0
            rewards[step] = reward
            if mode=='RL_full':
                episode.append([node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(),
                                distance.clone(), transporter[agent][0], mask])
            break

        # Select next agent if no valid action
        sw, temp_tardy = 0, 0
        while num_valid_coords <= 0 or sw == 0:
            sw = 1
            next_agent, mintime = select_agent(transporter)
            transporter, edge_fea_idx, node_fea, edge_fea, tardiness, tardy = next_state(
                transporter, edge_fea_idx, node_fea, edge_fea, tardiness, mintime, next_agent)
            agent = next_agent
            temp_tardy += tardy
            time += mintime
            valid_coords = ((edge_fea_idx >= 0) &
                            (edge_fea[:, :, 3 + int(transporter[agent][0])] == 0)).nonzero()
            num_valid_coords = valid_coords.shape[0]
            if num_valid_coords == 0:
                transporter[agent][2] = float("inf")

        # Record current step's result
        tardy_sum += td + temp_tardy
        ett_sum += ett
        reward = temp_tardy + ett + td
        reward_sum += reward
        event_list.extend([round(temp_tardy + td, 3), round(ett, 3), round(reward, 3)])
        event.append(event_list)
        actions[step] = action
        probs[step] = prob
        rewards[step] = reward
        unvisited_num -= 1
        step += 1

    return reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones


def do_action(transporter, edge_fea_idx, node_fea, edge_fea, agent, i, j, dis, time, step_to_ij, tardy_high):
    """
    Executes the selected action (i, j) for the given transporter (agent), updating the system state.

    Parameters:
        transporter (ndarray): Array of transporter states.
        edge_fea_idx (Tensor): Destination index tensor (N x M), updated after execution.
        node_fea (Tensor): Node features, includes transporter availability and expected completion time.
        edge_fea (Tensor): Edge features for each node pair.
        agent (int): Index of the transporter executing the action.
        i (int): Origin node index.
        j (int): Index of the edge (from node i) selected by the agent.
        dis (ndarray): Symmetric distance matrix between locations.
        time (float): Current global simulation time.
        step_to_ij (ndarray): Mapping from (i, j) to original block index.
        tardy_high (float): Normalization factor for time.

    Returns:
        transporter (ndarray): Updated transporter states.
        edge_fea_idx (Tensor): Updated edge feature index (marked -1 after visitation).
        node_fea (Tensor): Updated node features after transporter arrival.
        edge_fea (Tensor): (Unchanged) edge features.
        event_list (list): [start_time, ett_end_time, due_end_time, finish_time, agent_id, block_id].
        ett (float): Empty travel time (negative).
        td (float): Tardiness (non-negative if due is missed).
    """

    # Store previous transporter location
    past_location = int(transporter[agent][1])

    # Compute empty travel time and store in transporter
    transporter[agent][3] = dis[past_location, i] / 120 / tardy_high
    ett = -dis[past_location, i] / 120 / tardy_high  # negative for reward shaping

    # Calculate tardiness: how late it will be after due time
    td = min(edge_fea[i, j, 2].item() - dis[past_location, i] / 120 / tardy_high, 0) - min(edge_fea[i, j, 2].item(), 0)

    # Set transporter arrival time to new task's ready or travel time + processing time
    transporter[agent][2] = (max(dis[past_location, i] / 120 / tardy_high, edge_fea[i][j][1].item()) + edge_fea[i][j][0].item())

    # Update transporter destination and action memory
    transporter[agent][1] = edge_fea_idx[i][j].item()  # move to destination node
    transporter[agent][4] = i  # store origin node of this action
    transporter[agent][5] = j  # store edge index (destination info)

    # Create event log for Gantt chart visualization
    event_list = [
        round(time, 3),  # start time
        round(transporter[agent][3] + time, 3),  # empty travel finish time
        round(edge_fea[i][j][2].item() + time + edge_fea[i][j][0].item(), 3),  # due time end
        round(transporter[agent][2] + time, 3),  # processing finish time
        agent,  # transporter index
        step_to_ij[i][j]  # block ID
    ]

    # Update node feature at arrival location:
    # Update average expected completion time (running mean)
    dest_node = int(transporter[agent][1])
    tp_type = int(transporter[agent][0])
    old_count = node_fea[dest_node][tp_type * 2]
    old_mean = node_fea[dest_node][tp_type * 2 + 1]
    new_time = transporter[agent][2]
    node_fea[dest_node][tp_type * 2 + 1] = (old_mean * old_count + new_time) / (old_count + 1)
    node_fea[dest_node][tp_type * 2] += 1  # Increment available transporter count

    # Mark this edge as visited
    edge_fea_idx[i][j] = -1

    return transporter, edge_fea_idx, node_fea, edge_fea, event_list, ett, td



def next_state(transporter, edge_fea_idx, node_fea, edge_fea, tardiness, min_time, next_agent):
    """
    Advances the system state by `min_time` and updates all relevant tensors.

    Parameters:
        transporter (ndarray): Transporter state array (T x 6), including time left to arrive.
        edge_fea_idx (Tensor): Index tensor of destination nodes for each origin.
        node_fea (Tensor): Transporter availability and expected completion time at each node.
        edge_fea (Tensor): Edge features including ready time and due time.
        tardiness (float): Previous total tardiness value.
        min_time (float): Minimum time step to advance (i.e., time until next transporter becomes available).
        next_agent (int): Index of the next transporter becoming available.

    Returns:
        transporter (ndarray): Updated transporter state (time remaining reduced).
        edge_fea_idx (Tensor): Unchanged.
        node_fea (Tensor): Updated transporter availability and completion estimates.
        edge_fea (Tensor): Updated edge features (ready time, due time decremented).
        tardiness_next (float): Total new tardiness in the system.
        tardy (float): Incremental tardiness (tardiness_next - previous tardiness).
    """

    # Step 1: Decrease arrival time for all transporters
    transporter[:, 2] -= min_time  # time left to arrive

    # Step 2: Decrease expected completion time for all transporter types at all nodes
    TP_capacity_type_length = int(np.max(transporter[:, 0]) + 1)  # Number of transporter types
    idxs = [2 * i + 1 for i in range(TP_capacity_type_length)]
    node_fea[:, idxs] -= min_time

    # Step 3: Clip negative values in node features (for safety)
    node_fea[node_fea < 0] = 0

    # Step 4: One transporter arrives at a node (remove from available count)
    node = int(transporter[next_agent][1])
    tp_type = int(transporter[next_agent][0])
    node_fea[node, tp_type * 2] -= 1

    # Step 5: Decrease ready time and due time for all unvisited edges
    mask = torch.where(edge_fea_idx >= 0, torch.tensor(1.0), torch.tensor(0.0))  # valid edge mask
    edge_fea[:, :, [1, 2]] -= mask.unsqueeze(2).repeat(1, 1, 2) * min_time

    # Step 6: Clip negative ready times to 0
    edge_fea[:, :, 1][edge_fea[:, :, 1] < 0] = 0

    # Step 7: Calculate updated total tardiness
    tardiness_next = edge_fea[:, :, 2][edge_fea[:, :, 2] < 0].sum().item()
    tardy = tardiness_next - tardiness  # increase in tardiness during this time step

    return transporter, edge_fea_idx, node_fea, edge_fea, tardiness_next, tardy


def select_agent(transporter):
    """
    Selects the next available transporter agent based on the minimum remaining arrival time.

    Parameters:
        transporter (ndarray): Transporter state array (T x 6).
            Column 2 represents the remaining time until the transporter becomes available.

    Returns:
        agent (int): Index of the selected transporter with the minimum arrival time.
        min_time (float): The minimum time until the next event (agent becomes available).
    """

    # Get remaining arrival times for all transporters
    event = transporter[:, 2]  # column 2: time left to arrive

    # Find the minimum remaining time
    min_time = event.min()

    # Identify all transporters that share the minimum time (tie case)
    argmin = np.where(min_time == event)[0]

    # Randomly select one among those (to break ties fairly)
    agent = int(random.choice(argmin))

    return agent, min_time


def plot_gantt_chart(events, B, T):
    """
    Plots a Gantt chart visualizing transporter schedules with empty travel and job execution times.

    Parameters:
        events (list of lists): Each event is a list with the following format:
            [start_time, empty_travel_end_time, due_end_time, job_end_time, transporter_id, block_id]
        B (int): Number of blocks (used for color coding).
        T (int): Number of transporters.

    Returns:
        None (displays the Gantt chart using matplotlib).
    """

    # Generate distinct colors for each block
    colorset = plt.cm.rainbow(np.linspace(0, 1, B))

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot each event as two stacked bars: empty travel and job execution
    for event in events:
        job_start = event[0]
        empty_travel_end = event[1]
        job_end = event[3]
        transporter_id = int(event[4])
        block_id = int(event[5])

        # Plot empty travel time as a grey bar
        ax.barh(
            y=transporter_id,
            width=empty_travel_end - job_start,
            left=job_start,
            height=0.6,
            label=f'transporter {transporter_id + 1}',
            color='grey'
        )

        # Plot block processing time with a unique color per block
        ax.barh(
            y=transporter_id,
            width=job_end - empty_travel_end,
            left=empty_travel_end,
            height=0.6,
            label=f'transporter {transporter_id + 1}',
            color=colorset[block_id]
        )

        # Label the block on the job bar
        ax.text(
            (empty_travel_end + job_end) / 2,
            transporter_id,
            f'Block{block_id}',
            ha='center',
            va='center',
            fontsize=6
        )

    # Configure axis
    ax.set_xlabel('Time')
    ax.set_yticks(range(T))
    ax.set_yticklabels([f'transporter {i + 1}' for i in range(T)])

    # Display the chart
    plt.show()

