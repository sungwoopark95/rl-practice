import argparse

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--nsim", type=int, default=1, help="Number of simulations")
    parser.add_argument("--nsteps", type=int, default=1000, help="Runs in each simulation")
    parser.add_argument("--bernoulli", action='store_true')
    parser.add_argument("--n_arms", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--model", type=str, 
                        choices=['mab', 'linucb', 'hybridlinucb'], 
                        default='mab')
    
    return parser.parse_args()