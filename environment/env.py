from agent.network import *
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

class ProblemSampling:
    def __init__(self,block_number,location_number,transporter_type,transporter_number,dis_high,dis_low,ready_high,tardy_high,gap):
        """
        Initialize the ProblemSampling class.

        Parameters:
        - block_number (int): Number of blocks.
        - location_number (int): Number of locations.
        - transporter_type (int): Number of transporter types.
        - transporter_number (int): Number of transporters.
        - dis_high (float): Upper limit for distance sampling.
        - dis_low (float): Lower limit for distance sampling.
        - ready_high (int): Maximum ready time.
        - tardy_high (int): Maximum tardy time.
        - gap (int): Minimum gap between ready and tardy times.
        """
        self.block_number = block_number #30
        self.location_number = location_number #10
        self.transporter_type = transporter_type # 2
        self.transporter_number = transporter_number  # 6 3,3
        upper_tri = np.random.uniform(dis_low, dis_high, (location_number, location_number))
        upper_tri = np.triu(upper_tri, 1)  # 대각선 아래 제거
        symmetric_matrix = upper_tri + upper_tri.T
        np.fill_diagonal(symmetric_matrix, 0)
        self.dis = symmetric_matrix.copy()
        self.ready_high=ready_high #250
        self.tardy_high=tardy_high #300
        self.dis_high=dis_high
        self.gap=gap #100
    def sample(self):
        """
        Sample a new problem instance.

        Returns:
        - block_number (int): Number of blocks.
        - transporter_number (int): Number of transporters.
        - block (np.ndarray): Array of block attributes.
        - transporter (np.ndarray): Array of transporter attributes.
        - edge_fea_idx (np.ndarray): Edge feature indices.
        - node_in_fea (np.ndarray): Node input features.
        - edge_fea (np.ndarray): Edge features.
        - dis (np.ndarray): Distance matrix.
        - step_to_ij (np.ndarray): Step indices for edges.
        """

        block = np.zeros((self.block_number, 7))
        transporter = np.zeros((self.transporter_number, 6))
        for i in range(self.block_number):
            v = np.random.choice(self.location_number, 2, False)
            block[i, 0], block[i, 1] = v[0], v[1]
            block[i, 2] = self.dis[int(block[i, 0]), int(block[i, 1])] / 80 / self.tardy_high   #processing time
            block[i, 3] = np.random.randint(0, self.ready_high) / self.tardy_high   # ready time
            block[i, 4] = np.random.randint(block[i, 3] +self.gap,self.tardy_high ) / self.tardy_high - Block[i, 2]  # tardy time

            weight = np.random.uniform(0, 100)
            if weight > 50:
                block[i, 5] = 0
                block[i, 6] = 1
            else:
                block[i, 5] = 1
                block[i, 6] = 0

        block = block[block[:,0].argsort()]
        unique_values, counts = np.unique(block[:, 0], return_counts=True)
        max_count = np.max(counts)
        edge_fea_idx = -np.ones((self.location_number, max_count))
        edge_fea = np.zeros((self.location_number, max_count, 5))
        step = 0
        node_in_fea = np.zeros((self.location_number, 4))
        step_to_ij = np.zeros((self.location_number, max_count))
        for i in range(len(counts)):
            for j in range(max_count):
                if j < counts[i]:
                    edge_fea_idx[int(unique_values[i])][j] = int(block[step, 1])
                    edge_fea[int(unique_values[i])][j] = block[step, 2:7]
                    step_to_ij[int(unique_values[i])][j] = step
                    step += 1

        node_in_fea[0, 0] =  int(self.transporter_number / 2)
        node_in_fea[0, 2] = self.transporter_number-int(self.transporter_number / 2)
        

        for i in range(self.transporter_number):
            if i < int(self.transporter_number / 2):
                transporter[i, 0] = 0  # TP type
                transporter[i, 1] = 0  # TP heading point
                transporter[i, 2] = 0  # TP arrival left time
                transporter[i, 3] = 0  # empty travel time
                transporter[i, 4] = -1  # action i
                transporter[i, 5] = -1  # action j

            if i >= int(self.transporter_number / 2):
                transporter[i, 0] = 1  # TP type
                transporter[i, 1] = 0  # TP heading point
                transporter[i, 2] = 0  # TP arrival left time
                transporter[i, 3] = 0  # empty travel time
                transporter[i, 4] = -1  # action i
                transporter[i, 5] = -1  # action j

        return self.block_number, self.transporter_number, block, transporter, edge_fea_idx, node_in_fea, edge_fea, self.dis , step_to_ij


def simulation(B, T, transporter, block, edge_fea_idx, node_fea, edge_fea, dis, step_to_ij, tardy_high, mode,ppo):
    """
    Simulate the transportation problem.

    Parameters:
    - B (int): Number of blocks.
    - T (int): Number of transporters.
    - transporter (np.ndarray): Array of transporter attributes.
    - block (np.ndarray): Array of block attributes.
    - edge_fea_idx (np.ndarray): Edge feature indices.
    - node_fea (np.ndarray): Node input features.
    - edge_fea (np.ndarray): Edge features.
    - dis (np.ndarray): Distance matrix.
    - step_to_ij (np.ndarray): Step indices for edges.
    - tardy_high (int): Maximum tardy time.
    - mode (str): Simulation mode.
    - ppo (PPO): PPO agent.

    Returns:
    - reward_sum (float): Total reward.
    - tardy_sum (float): Total tardiness.
    - ett_sum (float): Total empty travel time.
    - event (list): List of events during the simulation.
    - episode (list): List of episodes during the simulation.
    - actions (np.ndarray): Actions taken during the simulation.
    - probs (np.ndarray): Probabilities of actions taken.
    - rewards (np.ndarray): Rewards received.
    - dones (np.ndarray): Done flags for episodes.
    """
    transporter = transporter.copy()
    block = block.copy()
    edge_fea_idx = edge_fea_idx.copy()
    node_fea = node_fea.copy()
    edge_fea = edge_fea.copy()
    event = []
    unvisited_num = B
    node_fea = torch.tensor(node_fea, dtype=torch.float32).to(device)
    edge_fea = torch.tensor(edge_fea, dtype=torch.float32).to(device)
    edge_fea_idx = torch.tensor(edge_fea_idx, dtype=torch.long).to(device)

    N=edge_fea_idx.shape[0]
    M=edge_fea_idx.shape[1]
    episode = []  # torch node_fea (9,13), edge_fea (9,3,5), edge_fea_idx(9,3), distance (9,3)
    probs = np.zeros(B)
    rewards = np.zeros(B)
    dones = np.ones(B)
    actions = np.zeros(B)
    tardiness = 0
    reward_sum = 0
    tardy_sum = 0
    ett_sum = 0
    step = 0
    time = 0
    prob = 0
    num_valid_coords = 10
    mask=np.ones((N,M,1))
    agent = np.random.randint(0, int(T/2)) #랜덤 트랜스포터 부터 지정
    node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] -= 1
    while unvisited_num > 0:
        # transporter (T,3) TP type / heading point / TP arrival time
        start_location = transporter[agent][1]
        distance = torch.tensor(dis[int(start_location)]/120/tardy_high, dtype=torch.float32).unsqueeze(1).repeat(1,edge_fea_idx.shape[1]).to(device)  #(n, e)
        if mode=='RL_full':
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            mask = np.ones((N, M, 1))
            for i in range(valid_coords.shape[0]):
                n=valid_coords[i][0].item()
                e=valid_coords[i][1].item()
                mask[n, e, 0] = 0
            mask = torch.tensor(mask).to(device)
            episode.append(
                [node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(), distance.clone(), transporter[agent][0],
                 mask])
            action, i, j, prob = ppo.get_action(node_fea, edge_fea, edge_fea_idx, mask, distance, transporter[agent][0])

        elif mode == 'RL_RHR':
            #masking action
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pri=np.zeros((6,valid_coords.shape[0]))
            mask=np.ones((N,M,1))
            action_list=[]
            for i in range(valid_coords.shape[0]):
                n=valid_coords[i][0]
                e=valid_coords[i][1]
                pri[0][i]=max(dis[int(start_location)][n]/120/tardy_high,edge_fea[n,e,1].item())+edge_fea[n,e,0].item()
                pri[1][i]=dis[int(start_location)][n] / 120 / tardy_high
                pri[2][i]=edge_fea[n,e,1].item()
                pri[3][i]=-(1/edge_fea[n,e,0]*torch.exp(-(edge_fea[n,e,2])/(torch.sum(edge_fea[:,:,0])/valid_coords.shape[0]))).item()
                pri[4][i]=edge_fea[n,e,2].item()
                pri[5][i]=-(1/edge_fea[n,e,0]*(1-(edge_fea[n,e,2]/edge_fea[n,e,0]))).item()
            for i in range(6):
                value=np.unique(pri[i])
                value1=value[0]
                for j in np.where(value1==pri[i])[0]:
                    n = valid_coords[j][0].item()
                    e = valid_coords[j][1].item()
                    mask[n, e, 0] = 0
                if len(value)>1:
                    value2 = value[1]
                    for j in np.where(value2 == pri[i])[0]:
                        n = valid_coords[j][0].item()
                        e = valid_coords[j][1].item()
                        mask[n, e, 0] = 0

            mask=torch.tensor(mask).to(device)
            episode.append(
            [node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(), distance.clone(), transporter[agent][0],mask])

            action, i, j, prob = ppo.get_action(node_fea, edge_fea, edge_fea_idx, mask,distance, transporter[agent][0])
            
        elif mode == 'RL_HR':
            #masking action
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()

            pri=np.zeros((6,valid_coords.shape[0]))
            mask=np.ones((N,M,1))
            action_list=[]
            for i in range(valid_coords.shape[0]):
                n=valid_coords[i][0]
                e=valid_coords[i][1]
                pri[0][i]=max(dis[int(start_location)][n]/120/tardy_high,edge_fea[n,e,1].item())+edge_fea[n,e,0].item()
                pri[1][i]=dis[int(start_location)][n] / 120 / tardy_high
                pri[2][i]=edge_fea[n,e,1].item()
                pri[3][i]=-(1/edge_fea[n,e,0]*torch.exp(-(edge_fea[n,e,2])/(torch.sum(edge_fea[:,:,0])/valid_coords.shape[0]))).item()
                pri[4][i]=edge_fea[n,e,2].item()
                pri[5][i]=-(1/edge_fea[n,e,0]*(1-(edge_fea[n,e,2]/edge_fea[n,e,0]))).item()
            for i in range(6):
                value=np.unique(pri[i])
                value1=value[0]
                for j in np.where(value1==pri[i])[0]:
                    n = valid_coords[j][0].item()
                    e = valid_coords[j][1].item()
                    mask[n, e, 0] = 0
            mask=torch.tensor(mask).to(device)
            episode.append(
            [node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(), distance.clone(), transporter[agent][0],mask])

            action, i, j, prob = ppo.get_action(node_fea, edge_fea, edge_fea_idx, mask,distance, transporter[agent][0])

        elif mode == 'Random':
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            num_valid_coords = valid_coords.shape[0]
            action = random.randint(0, num_valid_coords - 1)
            i = valid_coords[action][0].item()
            j = valid_coords[action][1].item()
        elif mode=='SPT': #PDR
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pt=np.zeros(valid_coords.shape[0])
            for i in range(valid_coords.shape[0]):
                n=valid_coords[i][0]
                e=valid_coords[i][1]
                pt[i]=max(dis[int(start_location)][n]/120/tardy_high,edge_fea[n,e,1].item())+edge_fea[n,e,0].item()
            min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

            # 같은 값이 여러 개인 경우 처리
            min_value = pt[min_index]
            same_value_indices = np.where(pt == min_value)[0]

            # 같은 값이 하나 이상인 경우
            if len(same_value_indices) > 1:
                min_index = np.random.choice(same_value_indices)
            action=min_index
            i= valid_coords[action][0].item()
            j= valid_coords[action][1].item()

        elif mode=='SET':

            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pt = np.zeros(valid_coords.shape[0])
            for i in range(valid_coords.shape[0]):
                n = valid_coords[i][0]
                e = valid_coords[i][1]
                pt[i] = dis[int(start_location)][n] / 120 / tardy_high
            min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

            # 같은 값이 여러 개인 경우 처리
            min_value = pt[min_index]
            same_value_indices = np.where(pt == min_value)[0]

            # 같은 값이 하나 이상인 경우
            if len(same_value_indices) > 1:
                min_index = np.random.choice(same_value_indices)
            action = min_index
            i = valid_coords[action][0].item()
            j = valid_coords[action][1].item()

        elif mode == 'SRT':
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pt = np.zeros(valid_coords.shape[0])
            for i in range(valid_coords.shape[0]):
                n = valid_coords[i][0]
                e = valid_coords[i][1]
                pt[i] = edge_fea[n,e,1].item()
            min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

            # 같은 값이 여러 개인 경우 처리
            min_value = pt[min_index]
            same_value_indices = np.where(pt == min_value)[0]

            # 같은 값이 하나 이상인 경우
            if len(same_value_indices) > 1:
                min_index = np.random.choice(same_value_indices)
            action = min_index
            i = valid_coords[action][0].item()
            j = valid_coords[action][1].item()

        elif mode=='ATC':
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pt = np.zeros(valid_coords.shape[0])
            for i in range(valid_coords.shape[0]):
                n = valid_coords[i][0]
                e = valid_coords[i][1]
                pt[i] = -(1/edge_fea[n,e,0]*torch.exp(-(edge_fea[n,e,2])/(torch.sum(edge_fea[:,:,0])/valid_coords.shape[0]))).item()
            min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

            # 같은 값이 여러 개인 경우 처리
            min_value = pt[min_index]
            same_value_indices = np.where(pt == min_value)[0]

            # 같은 값이 하나 이상인 경우
            if len(same_value_indices) > 1:
                min_index = np.random.choice(same_value_indices)
            action = min_index
            i = valid_coords[action][0].item()
            j = valid_coords[action][1].item()


        elif mode=='EDD':
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pt = np.zeros(valid_coords.shape[0])
            for i in range(valid_coords.shape[0]):
                n = valid_coords[i][0]
                e = valid_coords[i][1]
                pt[i] = edge_fea[n,e,2].item()
            min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

            # 같은 값이 여러 개인 경우 처리
            min_value = pt[min_index]
            same_value_indices = np.where(pt == min_value)[0]

            # 같은 값이 하나 이상인 경우
            if len(same_value_indices) > 1:
                min_index = np.random.choice(same_value_indices)
            action = min_index
            i = valid_coords[action][0].item()
            j = valid_coords[action][1].item()

        elif mode=='COVERT':
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            pt = np.zeros(valid_coords.shape[0])
            for i in range(valid_coords.shape[0]):
                n = valid_coords[i][0]
                e = valid_coords[i][1]
                pt[i] =-(1/edge_fea[n,e,0]*(1-(edge_fea[n,e,2]/edge_fea[n,e,0]))).item()
            min_index = np.argmin(pt)  # 가장 작은 값의 인덱스 찾기

            # 같은 값이 여러 개인 경우 처리
            min_value = pt[min_index]
            same_value_indices = np.where(pt == min_value)[0]

            # 같은 값이 하나 이상인 경우
            if len(same_value_indices) > 1:
                min_index = np.random.choice(same_value_indices)
            action = min_index
            i = valid_coords[action][0].item()
            j = valid_coords[action][1].item()

        transporter,  edge_fea_idx, node_fea, edge_fea, event_list,ett,td =do_action(transporter,
                                                                                                 edge_fea_idx.clone(),
                                                                                                 node_fea.clone(),
                                                                                                 edge_fea.clone(),
                                                                                                 agent, i, j, dis, time,
                                                                                                 step_to_ij, tardy_high)
        if unvisited_num == 1:
            event_list.append(round(td, 3))
            event_list.append(round(ett, 3))
            event_list.append(round(td+ett, 3))
            event.append(event_list)
            tardy_sum +=td
            ett_sum += ett
            reward =  ett +td
            reward_sum += reward
            actions[step] = action
            probs[step] = prob
            dones[step] = 0
            rewards[step] = reward
            episode.append(
                [node_fea.clone(), edge_fea.clone(), edge_fea_idx.clone(), distance.clone(), transporter[agent][0],
                 mask])
            break
        sw = 0  # do while

        temp_tardy = 0
        
        while (((num_valid_coords <= 0) | (sw == 0))):
            sw = 1

            next_agent, mintime = select_agent(transporter)
            
            transporter, edge_fea_idx, node_fea, edge_fea, tardiness, tardy = next_state(
                transporter,  edge_fea_idx, node_fea, edge_fea, tardiness,  mintime, next_agent)
            agent = next_agent
            temp_tardy += tardy
            time += mintime
            
            valid_coords = ((edge_fea_idx >= 0) & (transporter[agent][0] >= edge_fea[:, :, 4])).nonzero()
            num_valid_coords = valid_coords.shape[0]
            if num_valid_coords == 0:
                transporter[agent][2] = float("inf")
        tardy_sum +=td
        tardy_sum += temp_tardy
        ett_sum += ett
        reward = temp_tardy + ett +td
        event_list.append(round(temp_tardy+td, 3))
        event_list.append(round(ett, 3))
        event_list.append(round(reward, 3))

        # event_list 현재 시간, ett,tardy,완료시간,tp,몇번,tardy,ett,reward

        event.append(event_list)
        actions[step] = action
        probs[step] = prob
        rewards[step] = reward
        unvisited_num -= 1

        reward_sum += reward
        step += 1

        # edge fea는 시간 /220 , 속도 100/80
        # dis 거리/4000
        # ready time이 0보다 작으면 0으로
        # tardiness는 그떄 발생한 정도 case 1,2,3 0보다 작으면

    
      # event_list 현재 시간, ett,tardy,완료시간,tp,몇번,tardy,ett,reward

    return reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones


# 각각 action과 next_state로 분리하자
def do_action(transporter, edge_fea_idx, node_fea, edge_fea, agent, i, j, dis, time, step_to_ij,
              tardy_high):
    """
    Perform the action of a transporter agent.

    Parameters:
    - transporter (np.ndarray): Array of transporter attributes.
    - edge_fea_idx (np.ndarray): Edge feature indices.
    - node_fea (np.ndarray): Node input features.
    - edge_fea (np.ndarray): Edge features.
    - agent (int): Index of the transporter agent.
    - i (int): Index of the start node.
    - j (int): Index of the end node.
    - dis (np.ndarray): Distance matrix.
    - time (float): Current simulation time.
    - step_to_ij (np.ndarray): Step indices for edges.
    - tardy_high (int): Maximum tardy time.

    Returns:
    - transporter (np.ndarray): Updated transporter attributes.
    - edge_fea_idx (np.ndarray): Updated edge feature indices.
    - node_fea (np.ndarray): Updated node input features.
    - edge_fea (np.ndarray): Updated edge features.
    - event_list (list): List of event details.
    - ett (float): Empty travel time.
    - td (float): Tardy time.
    """
    past_location = int(transporter[agent][1])
    transporter[agent][3] = dis[int(transporter[agent][1]), i] /120 / tardy_high
    ett=-dis[int(transporter[agent][1]), i] /120 / tardy_high
    td=min(edge_fea[i,j,2].item()-dis[int(transporter[agent][1]), i] /120 / tardy_high,0)-min(edge_fea[i,j,2].item(),0)
    transporter[agent][2] = (max(dis[int(transporter[agent][1]), i] /120 / tardy_high, edge_fea[i][j][1].item()) + edge_fea[i][j][0].item())
    transporter[agent][1] = edge_fea_idx[i][j].item()
    transporter[agent][4] = i
    transporter[agent][5] = j
    event_list = [round(time, 3), round(transporter[agent][3] + time, 3), round(edge_fea[i][j][2].item() + time+ edge_fea[i][j][0].item(), 3),
                  round(transporter[agent][2] + time, 3), agent,
                  step_to_ij[i][j]]  # event_list 현재 시간, ett 끝 시간 ,tardy 끝 시간 ,완료 시간,tp, 몇번

    # 1 TP heading point
    # 2 TP arrival left time
    # 3 empty travel time
    # 4 action i
    # 5 action j

    node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2 + 1] = (node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2 + 1] *node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] +transporter[agent][2]) / (node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] + 1)
    node_fea[int(transporter[agent][1])][int(transporter[agent][0]) * 2] += 1
    edge_fea_idx[i][j] = -1
    return transporter, edge_fea_idx, node_fea, edge_fea, event_list,ett,td


def next_state(transporter, edge_fea_idx, node_fea, edge_fea,  tardiness, min_time,
               next_agent):
    
    """
    Update the state of the system after a time step.

    Parameters:
    - transporter (np.ndarray): Array of transporter attributes.
    - edge_fea_idx (np.ndarray): Edge feature indices.
    - node_fea (np.ndarray): Node input features.
    - edge_fea (np.ndarray): Edge features.
    - tardiness (float): Current total tardiness.
    - min_time (float): Minimum time step for the next event.
    - next_agent (int): Index of the next agent to act.

    Returns:
    - transporter (np.ndarray): Updated transporter attributes.
    - edge_fea_idx (np.ndarray): Updated edge feature indices.
    - node_fea (np.ndarray): Updated node input features.
    - edge_fea (np.ndarray): Updated edge features.
    - tardiness_next (float): Updated total tardiness.
    - tardy (float): Incremental tardiness.
    """
    transporter[:,2] -= min_time
    # node_fea

    node_fea[:, [1, 3]] = node_fea[:, [1, 3]] - min_time
    node_fea[node_fea < 0] = 0
    node_fea[int(transporter[next_agent][1]), int(transporter[next_agent][0]) * 2] -= 1

    # edge_fea
    mask=torch.where(edge_fea_idx >= 0, torch.tensor(1.0), torch.tensor(0.0))
    edge_fea[:, :, [1, 2]] = edge_fea[:, :, [1, 2]] - mask.unsqueeze(2).repeat(1, 1, 2) * min_time
    edge_fea[:, :, 1][edge_fea[:, :, 1] < 0] = 0
    tardiness_next = edge_fea[:, :, 2][edge_fea[:, :, 2] < 0].sum().item()
    tardy = tardiness_next - tardiness
    

    # tardiness 수정, weight constraint 고려 ready time
    return transporter,  edge_fea_idx, node_fea, edge_fea, tardiness_next, tardy


def select_agent(transporter):
    """
    Select the next agent based on the minimum arrival time.

    Parameters:
    - transporter (np.ndarray): Array of transporter attributes.

    Returns:
    - agent (int): Index of the selected agent.
    - min_time (float): Minimum time step for the next event.
    """
    event=transporter[:,2]
    min_time=event.min()
    argmin = np.where( (min_time == transporter[:,2]) & (transporter[:,0]==0))[0]

    if len(argmin)>0:
        agent = int(random.choice(argmin))
    else:
        argmin = np.where((min_time == transporter[:, 2]) )[0]
        agent = int(random.choice(argmin))
    return agent, min_time


def plot_gantt_chart(events, B, T):
    """
    Plot a Gantt chart for the transporter events.

    Parameters:
    - events (list): List of event details.
    - B (int): Number of blocks.
    - T (int): Number of transporters.
    """


    # version 1:
    colorset = plt.cm.rainbow(np.linspace(0, 1, B))

    # Set up figure and axis
    fig, ax = plt.subplots()

    # Plot Gantt chart bars
    for event in events:
        job_start = event[0]
        empty_travel_end = event[1]
        ax.barh(y=event[4], width=empty_travel_end - job_start, left=job_start, height=0.6,
                label=f'transporter {event[4] + 1}', color='grey')
        job_end = event[3]
        ax.barh(y=event[4], width=job_end - empty_travel_end, left=empty_travel_end, height=0.6,
                label=f'transporter {event[4] + 1}', color=colorset[int(event[5])])
        # ax.text((job_start+empty_travel_end)/2, event[3], 'empty travel time',ha='center',fontsize=7,va='center')
        ax.text((empty_travel_end + job_end) / 2, event[4], 'Block' + str(int(event[5])), ha='center', fontsize=6,
                va='center')

    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_yticks(range(T))
    ax.set_yticklabels([f'transporter {i + 1}' for i in range(T)])

    # Show the plot
    plt.show()
