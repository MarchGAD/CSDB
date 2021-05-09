import sys
sys.path.append('../')
from score.eer_and_mindcf import Resulter
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('-t', '--trials', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    path = args.directory
    score_files = os.listdir(path)
    for file in score_files:
        print(file)
        a = Resulter(os.path.join(path, file), args.trials)
        scores = a.compute_score()
        print(scores)