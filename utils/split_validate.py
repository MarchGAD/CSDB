import sys
sys.path.append('../.')
from utils.tools import RandomSelector
import re
import argparse


def generate_validation(scp_path, seed=19971222, validate_ratio=0.001, utt_per_spk=10):
    utt2path = {}
    spk2utt = {}
    with open(scp_path, 'r') as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp) != 2:
                continue
            utt, path = tmp
            utt2path[utt] = path
            spk = re.match(r'(.*?)-.*', utt).group(1)
            if spk is None:
                raise Exception('Wrong at %s' % line)
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)
    spks = list(spk2utt.keys())
    rs = RandomSelector(seed=seed)
    validate_spk_num = int(len(spk2utt) * validate_ratio)

    selected_spks = {spks[i]: [spk2utt[spks[i]].pop() for _ in range(utt_per_spk)] for i in
                     rs.get_random_ints(0, len(spks), validate_spk_num)}

    def write2file(file_path, s2t):
        with open(file_path, 'w') as f:
            for spk in s2t:
                for utt in s2t[spk]:
                    f.write(utt + ' ' + utt2path[utt] + '\n')

    write2file('./voxceleb_v2.scp', spk2utt)
    write2file('./validate.scp', selected_spks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scp_path', type=str)
    parser.add_argument('-r', '--ratio', type=float, default=0.001)
    parser.add_argument('-s', '--seed', type=int, default=19971222)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    generate_validation(args.scp_path, validate_ratio=args.ratio, seed=args.seed)
