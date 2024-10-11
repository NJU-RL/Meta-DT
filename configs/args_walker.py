import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', default="WalkerRandParams-v0")
    parser.add_argument('--data_quality', default="medium")
    parser.add_argument('--seed', type=int, default=412, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--max_episode_steps', type=int, default=200)
    parser.add_argument('--num_eval_episodes', type=int, default=5)

    # for data collection
    parser.add_argument('--num_tasks', type=int, default=50)

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=50000, metavar='N', help='maximum number of steps (default: 2000)')
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--start_steps', type=int, default=100, metavar='N', help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--replay_size', type=int, default=10000, metavar='N', help='size of replay buffer (default: 10000000)')
    
    ### for the context encoder and reward/transition decoder
    parser.add_argument('--context_hidden_dim', type=int, default=128)
    parser.add_argument('--context_dim', type=int, default=16)
    parser.add_argument('--context_batch_size', type=int, default=128)
    parser.add_argument('--context_train_epochs', type=int, default=201)
    parser.add_argument('--save_context_model_every', type=int, default=100)
    parser.add_argument('--context_lr', type=float, default=0.001)
    parser.add_argument('--num_train_tasks', type=int, default=45)

    parser.add_argument('--dt_batch_size', default=32, type=int)
    parser.add_argument('--dt_horizon', type=int, default=8)
    parser.add_argument('--dt_embed_dim', type=int, default=128)
    parser.add_argument('--dt_n_layer', type=int, default=3)
    parser.add_argument('--dt_n_head', type=int, default=1)
    parser.add_argument('--dt_activation_function', type=str, default='relu')
    parser.add_argument('--dt_dropout', type=float, default=0.1)
    parser.add_argument('--dt_lr', type=float, default=5e-5)
    parser.add_argument('--dt_weight_decay', type=float, default=1e-4)
    parser.add_argument('--dt_warmup_steps', type=int, default=1000)
    parser.add_argument('--dt_num_epochs', type=int, default=100)
    parser.add_argument('--dt_return_scale', type=float, default=500.)

    ### for meta-decision transformer
    parser.add_argument('--meta_dt_batch_size', default=128, type=int)
    parser.add_argument('--meta_dt_num_epochs', type=int, default=200)
    parser.add_argument('--meta_dt_warmup_steps', type=int, default=10000)
    parser.add_argument('--prompt_length', type=int, default=3)
    parser.add_argument('--state_dim', type=int, default=17)
    parser.add_argument('--act_dim', type=int, default=6)
    parser.add_argument('--max_step', type=int, default=100000)
    parser.add_argument('--eval_step', type=int, default=500)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--scale', type=int, default=500.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_train_eposides', type=int, default=100)
    parser.add_argument("--pdt", type=bool, default=False)
    parser.add_argument('--warm_train', type=int, default=20000)
    parser.add_argument("--meta_dt_few", type=bool, default=True)
    parser.add_argument('--total_epi', type=int, default=100)
    args = parser.parse_args(rest_args)
    return args