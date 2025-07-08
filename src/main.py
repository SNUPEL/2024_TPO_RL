from Simulation import *
from Network import *
import torch
#import vessl
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

"""
Main script for training and evaluating a PPO-based transport scheduling policy.

This script:
- Generates synthetic block transport scheduling problems.
- Trains a PPO agent using graph-based state representations.
- Compares against multiple heuristic baselines (Random, SSPT, SET, etc.).
- Logs training and validation metrics via Vessl.
- Saves training history and validation results to Excel files.

Structure:
1. Generate validation problems and evaluate heuristics.
2. Iteratively train the PPO agent.
3. Periodically evaluate PPO on validation set.
4. Save training and validation results.

Requirements:
- Simulation environment implemented in `simulation()`
- PPO agent implemented in `PPO` class (in `Network.py`)
- Problem sampler implemented in `Problem_sampling` (in `Simulation.py`)
"""

if __name__ == "__main__":
    problem_dir = '/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir = '/output/model/ppo/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir = '/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    device = 'cuda'

    # Problem configuration
    block_number = 40
    location_number = 20
    transporter_type = 2
    transporter_number = 8
    dis_high = 3000
    dis_low = 500
    ready_high = 100
    tardy_high = 300
    gap = 100
    K_epoch = 2

    # Instantiate problem sampler and PPO agent
    Pr_sampler = Problem_sampling(block_number, location_number, transporter_type,
                                  transporter_number, dis_high, dis_low, ready_high, tardy_high, gap)
    dis = torch.tensor(Pr_sampler.Dis, dtype=torch.float32).to(device)

    ppo = PPO(learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01,
              epsilon=0.2, discount_factor=1, location_num=location_number,
              transporter_type=transporter_type, dis=dis, gnn_mode='CGCNN').to(device)

    # Validation configuration
    number_of_validation = 20
    number_of_validation_batch = 50

    # Training configuration
    number_of_problem = 10        # Problems per batch
    number_of_batch = 30          # Episodes per problem
    number_of_trial = 1           # How often to regenerate training problems
    number_of_iteration = int(1001 / number_of_trial)

    validation = []
    validation_step = 10          # Perform validation every N steps
    Heuristic_result = np.zeros((20, 7, 6))  # [problem, mode, metrics]
    history = np.zeros((number_of_iteration * number_of_trial, 2))
    validation_history = np.zeros((int(1001 / validation_step) + 10, 4))

    step = 0
    mode_list = ['Random', 'SSPT', 'SET', 'SRT', 'ATCS', 'MDD', 'COVERT']
    temp_step = 0
    past_time_step = 0

    # === Run heuristic baselines for validation ===
    for j in range(number_of_validation):
        B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
        efi = efi.astype('int')
        validation.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])

        for nu, mod in enumerate(mode_list):
            rs = np.zeros(number_of_validation_batch)
            es = np.zeros(number_of_validation_batch)
            ts = np.zeros(number_of_validation_batch)
            for k in range(number_of_validation_batch):
                reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                    validation[j][0], validation[j][1], validation[j][2], validation[j][3],
                    validation[j][4], validation[j][5], validation[j][6], validation[j][7],
                    validation[j][8], validation[j][9], mod, ppo)
                rs[k] = reward_sum
                es[k] = ett_sum
                ts[k] = tardy_sum
            # Store heuristic statistics
            Heuristic_result[temp_step, nu, 0] = rs.mean()
            Heuristic_result[temp_step, nu, 1] = rs.var()
            Heuristic_result[temp_step, nu, 2] = es.mean()
            Heuristic_result[temp_step, nu, 3] = es.var()
            Heuristic_result[temp_step, nu, 4] = ts.mean()
            Heuristic_result[temp_step, nu, 5] = ts.var()
            print(Heuristic_result[0,nu,0])
        temp_step += 1

    # === PPO Training loop ===
    for i in range(number_of_iteration):
        problem = []
        temp_step = 0
        # Generate training problems
        for j in range(number_of_problem):
            B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
            efi = efi.astype('int')
            problem.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])

        for k in range(number_of_trial):
            ave_reward = 0
            ave_tardy = 0
            ave_ett = 0
            loss_temp = 0

            # Initialize episode buffer
            data = []
            action_list = np.array([])
            prob_list = np.array([])
            reward_list = np.array([])
            done_list = np.array([])

            # Run simulation and collect trajectories
            for j in range(number_of_problem):
                for l in range(number_of_batch):
                    reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                        problem[j][0], problem[j][1], problem[j][2], problem[j][3],
                        problem[j][4], problem[j][5], problem[j][6], problem[j][7],
                        problem[j][8], problem[j][9], 'RL_full', ppo)
                    ave_reward += reward_sum.item()
                    ave_ett += ett_sum
                    ave_tardy += tardy_sum
                    data.append(episode)
                    action_list = np.concatenate((action_list, actions))
                    prob_list = np.concatenate((prob_list, probs))
                    reward_list = np.concatenate((reward_list, rewards))
                    done_list = np.concatenate((done_list, dones))
            print(ave_reward)
            # Update PPO agent
            for m in range(K_epoch):
                ave_loss, v_loss, p_loss = ppo.update(
                    data, prob_list, reward_list, action_list, done_list, step, validation_step, model_dir)
                print(ave_loss)
                loss_temp += ave_loss

            # Average metrics
            ave_reward /= number_of_problem * number_of_batch
            ave_ett /= number_of_problem * number_of_batch
            ave_tardy /= number_of_problem * number_of_batch

            history[step, 0] = ave_reward
            #Suggest to write print code or any visualization code for current step observation
            #vessl.log(step=step, payload={'train_average_reward': ave_reward})
            step += 1

            # Perform validation every `validation_step`, after warm-up
            if step % validation_step == 1 and step > 500:
                valid_reward_full = 0
                valid_ett_full = 0
                valid_tardy_full = 0
                best_reward = 0
                best_ett = 0
                best_tardy = 0
                for j in range(number_of_validation):
                    temp_best_reward = -100
                    for l in range(number_of_validation_batch):
                        reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                            validation[j][0], validation[j][1], validation[j][2], validation[j][3],
                            validation[j][4], validation[j][5], validation[j][6], validation[j][7],
                            validation[j][8], validation[j][9], 'RL_full', ppo)
                        valid_reward_full += reward_sum.item()
                        valid_ett_full += ett_sum
                        valid_tardy_full += tardy_sum
                        temp_best_reward = max(reward_sum.item(), temp_best_reward)
                    best_reward += temp_best_reward

                valid_reward_full /= (number_of_validation * number_of_validation_batch)
                valid_ett_full /= (number_of_validation * number_of_validation_batch)
                valid_tardy_full /= (number_of_validation * number_of_validation_batch)
                best_reward /= number_of_validation

                valid_step = int(step / validation_step)
                validation_history[valid_step, 0] = valid_reward_full
                validation_history[valid_step, 1] = valid_ett_full
                validation_history[valid_step, 2] = valid_tardy_full
                validation_history[valid_step, 3] = best_reward
                #vessl.log(step=step, payload={'valid_average_reward_full': valid_reward_full})

    # Save history results to Excel
    history = pd.DataFrame(history)
    validation_history = pd.DataFrame(validation_history)
    history.to_excel(history_dir + 'history.xlsx', sheet_name='Sheet', index=False)
    validation_history.to_excel(history_dir + 'valid_history.xlsx', sheet_name='Sheet', index=False)


