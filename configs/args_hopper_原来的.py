import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # parser.add_argument('--env_name', default="hopper-medium-v2")
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--device', type=str, default='cuda:0')

    ### for decision transformer
    parser.add_argument("--dt_batch_size", default=64, type=int)
    parser.add_argument('--dt_horizon', type=int, default=20)
    parser.add_argument('--dt_embed_dim', type=int, default=128)
    parser.add_argument('--dt_n_layer', type=int, default=3)
    parser.add_argument('--dt_n_head', type=int, default=1)
    parser.add_argument('--dt_activation_function', type=str, default='relu')
    parser.add_argument('--dt_dropout', type=float, default=0.1)
    parser.add_argument('--dt_lr', type=float, default=1e-4)
    parser.add_argument('--dt_weight_decay', type=float, default=1e-4)
    parser.add_argument('--dt_warmup_steps', type=int, default=10000)
    parser.add_argument('--dt_num_eval_episodes', type=int, default=5)
    parser.add_argument('--dt_steps_per_eval', type=int, default=1000)
    parser.add_argument('--dt_training_steps', type=int, default=100000)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--dt_return_scale', type=float, default=1000)
    args = parser.parse_args(rest_args)
    return args