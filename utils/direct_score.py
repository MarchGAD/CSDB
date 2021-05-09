import sys
sys.path.append('../')
import argparse
import kaldiio as ko


def out_sum(x, y):
    return (x @ y.T).sum()


def pw_sum(x, y):
    return (x * y).sum()


def square_score(scp_path, trial_path, output_path='./square_score'):
    '''
        assume x \in R^{N \times 1}
        score = 2 * x1 * x2 + x1 * x1 + x2 * x2
        Note that the '*' stands for pairwise multiply here.
    '''
    data = ko.load_scp(scp_path)
    start = False
    with open(output_path, 'w') as op:
        with open(trial_path, 'r') as tp:
            for line in tp:
                u1, u2, _ = line.strip().split()
                x1 = data.get(u1).reshape(-1, 1)
                x2 = data.get(u2).reshape(-1, 1)
                sco = 2 * pw_sum(x1, x2) + pw_sum(x1, x1) + pw_sum(x2, x2)
                op.write('{}{} {} {}'.format('\n' if start else '', u1, u2, sco))
                if not start:
                    start = True


def outer_score(scp_path, trial_path, output_path='./outer_score'):
    '''
        assume x \in R^{N \times 1}
        score = 2 * (x1^T * x2).sum() + (x1^T * x1).sum() + (x2^T * x2).sum()
    '''
    data = ko.load_scp(scp_path)
    start = False
    with open(output_path, 'w') as op:
        with open(trial_path, 'r') as tp:
            for line in tp:
                u1, u2, _ = line.strip().split()
                x1 = data.get(u1).reshape(-1, 1)
                x2 = data.get(u2).reshape(-1, 1)
                sco = 2 * out_sum(x1, x2) + out_sum(x1, x1) + out_sum(x2, x2)
                op.write('{}{} {} {}'.format('\n' if start else '', u1, u2, sco))
                if not start:
                    start = True


def inner_score(scp_path, trial_path, output_path='./inner_score'):
    '''
        assume x \in R^{N \times 1}
        score = x1^T * x2
    '''

    data = ko.load_scp(scp_path)
    start = False
    with open(output_path, 'w') as op:
        with open(trial_path, 'r') as tp:
            for line in tp:
                u1, u2, _ = line.strip().split()
                sco = (data.get(u1) * data.get(u2)).sum()
                op.write('{}{} {} {}'.format('\n' if start else '', u1, u2, sco))
                if not start:
                    start = True


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-sp', '--scp_path', type=str)
    args.add_argument('-tp', '--trials_path', type=str)
    args.add_argument('-st', '--score_type', type=str)
    args.add_argument('-od', '--output_dir', type=str)
    return args.parse_args()


if __name__ == '__main__':
    import os
    args =main()
    if args.score_type == 'inner':
        inner_score(args.scp_path, args.trials_path, os.path.join(args.output_dir, 'inner_score'))
    elif args.score_type == 'outer':
        outer_score(args.scp_path, args.trials_path, os.path.join(args.output_dir, 'outer_score'))
    elif args.score_type == 'square':
        square_score(args.scp_path, args.trials_path, os.path.join(args.output_dir, 'square_score'))