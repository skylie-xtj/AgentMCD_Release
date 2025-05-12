import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Overall settings
    parser.add_argument('--llm_type', type=str, default='THUDM/glm-4-9b-chat', 
                        help='LLM generator')
    parser.add_argument('--lang', type=str, default='zh', 
                        help='Language')
    parser.add_argument('--cuda', type=int, default=2,
                        help='Specify which gpu to use.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--num_avatars', type=int, default=20,
                        help='Number of avatars for sandbox simulation.')
    parser.add_argument('--execution_mode', type=str, default='parallel',
                        choices=['serial', 'parallel'],
                        help='Specify execution mode: serial or parallel.')
    parser.add_argument("--use_wandb", action="store_true",
                        help="whether to use wandb")
    
    # Only for validating the effectiveness of agents
    parser.add_argument("--val_users", action="store_true",
                        help="whether to validate users")
    parser.add_argument('--val_ratio', type=int, default=1,
                        help='Ratio of unobserved items vs ground truth for validation.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='mcd',
                        help='Dataset to use.')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1132)
    parser.add_argument('--simulation_name', type=str, default='prompt_final',
                        help='The name of one trial of simulation.')
    parser.add_argument('--test_num', type=str, default='test',
                        help='test_num')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='max_workers')
    
    args, _ = parser.parse_known_args()

    return args


