import argparse


def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--vessl", type=int, default=1, help="whether to use vessl (0: False, 1:True)")

    parser.add_argument("--n_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--look_ahead", type=int, default=2, help="number of steel plates included in states")
    parser.add_argument("--record_events", type=int, default=0, help="Whether to record events")

    # 데이터 생성 관련 파라미터
    parser.add_argument("--n_rows", type=int, default=2, help="steel plates data for storage")

    parser.add_argument("--storage", type=int, default=1, help="steel plates data for storage")
    parser.add_argument("--reshuffle", type=int, default=1, help="steel plates data for reshuffle")
    parser.add_argument("--retrieval", type=int, default=1, help="steel plates data for retrieval")

    parser.add_argument("--n_bays_in_area1", type=int, default=15, help="number of bays in Area1")
    parser.add_argument("--n_bays_in_area2", type=int, default=6, help="number of bays in Area2")
    parser.add_argument("--n_bays_in_area3", type=int, default=3, help="number of bays in Area3")
    parser.add_argument("--n_bays_in_area4", type=int, default=6, help="number of bays in Area4")
    parser.add_argument("--n_bays_in_area5", type=int, default=9, help="number of bays in Area5")
    parser.add_argument("--n_bays_in_area6", type=int, default=1, help="number of bays in Area6")

    parser.add_argument("--n_from_piles_storage", type=int, default=1, help="number of from-piles in storage work")
    parser.add_argument("--n_to_piles_storage", type=int, default=5, help="number of to-piles in storage work")
    parser.add_argument("--n_from_piles_reshuffle", type=int, default=10, help="number of from-piles in reshuffle work")
    parser.add_argument("--n_to_piles_reshuffle", type=int, default=10, help="number of to-piles for in reshuffle work")
    parser.add_argument("--n_from_piles_retrieval_cn1", type=int, default=5, help="number of from-piles for conveyor 1 in retrieval work")
    parser.add_argument("--n_from_piles_retrieval_cn2", type=int, default=5, help="number of from-piles for conveyor 2 in retrieval work")
    parser.add_argument("--n_from_piles_retrieval_cn3", type=int, default=2, help="number of from-piles for trailers in retrieval work")

    parser.add_argument("--n_plates_storage", type=int, default=500, help="average number of steel plates per pile in storage work")
    parser.add_argument("--n_plates_reshuffle", type=int, default=150, help="average number of steel plates per pile in reshuffle work")
    parser.add_argument("--n_plates_retrieval", type=int, default=150, help="average number of steel plates per pile in retrieval work")

    parser.add_argument("--is_crane1_working", type=bool, default=True, help="Crane-1 is working")
    parser.add_argument("--is_crane2_working", type=bool, default=True, help="Crane-2 is working")
    parser.add_argument("--safety_margin", type=int, default=5, help="safety margin between cranes")

    parser.add_argument("--multi_num", type=int, default=3, help="Number of plates allowed for multi-loading")
    parser.add_argument("--multi_w", type=float, default=20.0, help="Total weight of plates allowed for multi-loading")
    parser.add_argument("--multi_dis", type=int, default=2, help="Distance allowed for multi-loading")

    parser.add_argument("--reward_sig", type=int, default=0, help="Reward function")
    parser.add_argument("--parameter_sharing", type=int, default=1, help="Use parameter sharing")
    parser.add_argument("--team_reward", type=int, default=1, help="Use team reward")

    parser.add_argument("--embed_dim", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="multi-head attention in HGT layers")
    parser.add_argument("--num_HGT_layers", type=int, default=2, help="number of HGT layers")
    parser.add_argument("--num_actor_layers", type=int, default=2, help="number of actor layers")
    parser.add_argument("--num_critic_layers", type=int, default=2, help="number of critic layers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay ratio")
    parser.add_argument("--lr_step", type=int, default=2000, help="step size to reduce learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping paramter")
    parser.add_argument("--K_epoch", type=int, default=5, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=10, help="the number of steps to obtain samples")
    parser.add_argument("--P_coeff", type=float, default=1, help="coefficient for policy loss")
    parser.add_argument("--V_coeff", type=float, default=0.5, help="coefficient for value loss")
    parser.add_argument("--E_coeff", type=float, default=0.01, help="coefficient for entropy loss")

    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x episodes")
    parser.add_argument("--new_instance_every", type=int, default=10, help="Generate new scenarios every x episodes")

    parser.add_argument("--val_dir", type=str, default=None, help="directory where the validation data are stored")

    return parser.parse_args()