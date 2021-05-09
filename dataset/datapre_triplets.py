import os
import sys
sys.path.append('../.')
import argparse
import random as rd
from dataset.kaldi_reader import KaldiLoader
from itertools import combinations


def print2file(str, end='\n'):
    global log_path
    with open(log_path, 'a') as f:
        print(str, end=end, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--scp_path', type=str, default='F:\BOOKS\DTPLDA-like-neural-backend\scps\demo\\test.scp')
    parser.add_argument('-sn', '--spk_num', type=int, default=10)
    parser.add_argument('-un', '--utt_num', type=int, default=5,
                        help='remove spks who have less utts than utt_num')
    parser.add_argument('-s', '--seed', type=int, default=19990110)
    parser.add_argument('-rt', '--repeat_times', type=int, default=5)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-td', '--target_directory', default='F:\BOOKS\DTPLDA-like-neural-backend\\result')
    parser.add_argument('-r', '--ratio', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    rd.seed(args.seed)
    log_path = os.path.join(args.target_directory, 'log')
    config_path = os.path.join(args.target_directory, 'config')
    utt_path = os.path.join(args.target_directory, 'train_utts')
    tri_path = os.path.join(args.target_directory, 'train_triplets')
    val_path = os.path.join(args.target_directory, 'validate_trials')
    if os.path.exists(utt_path) or os.path.exists(tri_path) or os.path.exists(val_path):
        print2file('paths occupied.')
        raise Exception('paths occupied.')

    loader = KaldiLoader(scp_path=args.scp_path)
    (spk_num, utt_num) = (args.spk_num, args.utt_num)
    assert spk_num > 0 and utt_num > 0
    anchor_size = spk_num * utt_num
    batch_size = int(spk_num * utt_num * (utt_num - 1) / 2 + spk_num * (spk_num - 1) / 2 * utt_num * utt_num)
    tot_spk_num = len(loader.spk_list)
    val_spk_num = int(args.ratio * tot_spk_num)
    # tra_spk_num = tot_spk_num - val_spk_num
    raw_train_spk_list = loader.spk_list[:-val_spk_num]
    valid_spk_list = loader.spk_list[-val_spk_num:]

    with open(config_path, 'w') as cp:
        cp.write(args.scp_path + '\n' +
                str(spk_num) + '\t' + str(utt_num) + '\t' + str(anchor_size) +
                 '\t' + str(batch_size) + '\t' + str(tot_spk_num) + '\t' + str(args.seed) +
                 '\t' + str(val_spk_num) + '\t' + str(args.repeat_times) + '\t' + str(args.epochs)
                 )

    print('scp_path: %s,\n'
          'spk_num: %d, utt_num: %d, \n'
          'anchor_size(ap_size): %d, batch_size(an_size): %d, tot_spk_num: %d, val_spk_num: %d'
          % (args.scp_path, spk_num, utt_num, anchor_size, batch_size, tot_spk_num, val_spk_num))
    print2file('scp_path: %s,\n'
               'spk_num: %d, utt_num: %d, \n'
               'anchor_size(ap_size): %d, batch_size(an_size): %d, tot_spk_num: %d, val_spk_num: %d'
               % (args.scp_path, spk_num, utt_num, anchor_size, batch_size, tot_spk_num, val_spk_num))

    raw_train_spk_dict = {spk :
                        [loader.raw_labels[ind] for ind in loader.spk2indexes[spk]]
                          for spk in raw_train_spk_list}
    valid_spk_dict = {spk :
                        [loader.raw_labels[ind] for ind in loader.spk2indexes[spk]]
                    for spk in valid_spk_list}
    print('seed is %d' % args.seed)
    print2file('seed is %d' % args.seed)

    train_spk_dict = {spk: uttlist for spk, uttlist in raw_train_spk_dict.items()
                      if len(uttlist) >= args.utt_num}
    train_spk_list = list(train_spk_dict.keys())
    tra_spk_num = len(train_spk_dict)
    print('training num after removing spk with less utts is %d' % len(train_spk_dict))
    print2file('training  num after removing spk with less utts is %d' % len(train_spk_dict))

    print('spk num for train is %d, for valid is %d' % (tra_spk_num, val_spk_num))
    print2file('spk num for train is %d, for valid is %d' % (tra_spk_num, val_spk_num))

    # generate valid pairs
    print('generating valid pairs.')
    print2file('generating valid pairs.')
    head = True
    with open(val_path, 'w') as vp:
        for spk in valid_spk_dict.keys():
            for utt1, utt2 in combinations(valid_spk_dict[spk], 2):
                vp.write(('\n' if not head else '') + utt1 + '\t' + utt2 + '\t' + '1')
                if head:
                    head = False

        for spk1, spk2 in combinations(valid_spk_dict.keys(), 2):
            for utt1 in valid_spk_dict[spk1]:
                for utt2 in valid_spk_dict[spk2]:
                    vp.write(('\n' if not head else '') + utt1 + '\t' + utt2 + '\t' + '0')

    # generate train triplets
    uhead = True
    thead = True
    for _ in range(args.repeat_times * args.epochs):
        print('generating epoch %d.' % (_ // args.repeat_times + 1))
        print2file('generating epoch %d.' % (_ // args.repeat_times + 1))
        spk_inds = rd.sample(range(tra_spk_num), tra_spk_num)
        with open(utt_path, 'a') as up:
            with open(tri_path, 'a') as tp:
                for i in range(tra_spk_num // spk_num):
                    sub_spk_inds = spk_inds[i * spk_num: (i + 1) * spk_num]
                    sub_spk2utt = {train_spk_list[ind]: []
                                   for ind in sub_spk_inds}
                    for spk in sub_spk2utt.keys():
                        utt_inds = rd.sample(range(len(train_spk_dict[spk])), utt_num)
                        sub_spk2utt[spk] = [train_spk_dict[spk][ind]
                                            for ind in utt_inds]

                    for ind in sub_spk_inds:
                        spk = train_spk_list[ind]
                        utts = sub_spk2utt[spk]
                        for utt in utts:
                            up.write(('\n' if not uhead else '') + utt)
                            if uhead:
                                uhead = False

                    # generate ap pairs
                    for ind in sub_spk_inds:
                        spk = train_spk_list[ind]
                        for utt1, utt2 in combinations(sub_spk2utt[spk], 2):
                            tp.write(('\n' if not thead else '') + utt1 + '\t' + utt2 + '\t' + '1')
                            if thead:
                                thead = False

                    # generate an pairs
                    for ind1, ind2 in combinations(sub_spk_inds, 2):
                        spk1 = train_spk_list[ind1]
                        spk2 = train_spk_list[ind2]
                        for utt1 in sub_spk2utt[spk1]:
                            for utt2 in sub_spk2utt[spk2]:
                                tp.write(('\n' if not thead else '') + utt1 + '\t' + utt2 + '\t' + '0')