# from functools import partial
import argparse

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--epoch", type=int, default=1, help="Training epochs")
    
    return parser.parse_args()