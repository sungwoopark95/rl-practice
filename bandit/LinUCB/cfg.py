import argparse

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--nsim", type=int, default=1, help="Number of simulations")
    parser.add_argument("--topN", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--model", type=str, 
                        choices=['linucb', 'hybrid', 'elinucb', 'lints'], 
                        default='linucb')
    
    return parser.parse_args()
