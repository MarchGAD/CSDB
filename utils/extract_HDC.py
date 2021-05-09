import os
import sys
sys.path.append('../')
import torch
import random
import torch.nn.functional as F
import kaldiio as ko

from utils.tools import flat
from itertools import combinations
from argparse import ArgumentParser
from dataset.kaldi_reader import KaldiLoader



def load(path):
    return torch.load(path)


def forward(xvec1, xvec2, model):
    # now is specified for LDAWCCN, should rewrite nnet.py in the future
    xvec1 = model.lda(xvec1)
    xvec1 = model.wccn(xvec1)
    xvec1 = F.normalize(xvec1)

    xvec2 = model.lda(xvec2)
    xvec2 = model.wccn(xvec2)
    xvec2 = F.normalize(xvec2)

    tmp = flat(xvec1, xvec2)
    val = torch.cat([tmp, xvec1 + xvec2], dim=1)

    return val


def get_utt_list(raw_labels, utt_inds):
    return [raw_labels[i] for i in utt_inds]


def test(scp_path, dif_pairs=-1, sam_pairs=-1, expected_length=10010):
    loder = KaldiLoader(scp_path=scp_path)
    sam_cnt = 0
    dif_cnt = 0

    # same pairs
    for spk in loder.spk_list:
        utt_num = len(loder.spk2indexes[spk])
        utt_squ = utt_num * utt_num
        sam_cnt += min(utt_squ, sam_pairs) if sam_pairs != -1 \
                    else utt_squ

    # different pairs
    for spk1, spk2 in combinations(loder.spk_list, 2):
        utt_num1 = len(loder.spk2indexes[spk1])
        utt_num2 = len(loder.spk2indexes[spk2])
        utt_squ = utt_num1 * utt_num2
        dif_cnt += min(utt_squ, dif_pairs) if dif_pairs != -1 \
            else utt_squ

    space = int(expected_length * (dif_cnt + sam_cnt) * 4 / 1024 / 1024)

    print('spk_num:{},\n'
          'same_pairs:{}\n'
          'different_pairs:{}\n'
          'required space(MB):{}'.format(len(loder.spk_list), sam_cnt, dif_cnt, space))


def extract(scp_path, model,
            name=None, path='./',
            dif_pairs=20, sam_pairs=-1):

    name = name if name is not None else \
    os.path.basename(scp_path).split('\.')[0]

    loder = KaldiLoader(scp_path=scp_path)
    meta_dict = {}
    key_list = []

    # same pairs
    for spk in loder.spk_list:
        utt_list = get_utt_list(loder.raw_labels,
                                loder.spk2indexes[spk])
        key_list.clear()
        for utt1 in utt_list:
            for utt2 in utt_list:
                key = "{}#{}".format(utt1, utt2)
                key_list.append(key)

        sam_key_pairs = random.sample(key_list,
                                  sam_pairs if sam_pairs != -1
                                  else min(len(key_list), sam_pairs))

        for key in sam_key_pairs:
            utt1, utt2 = key.split('#')

            utt_xvec1 = loder.datas.get(utt1)
            utt_xvec2 = loder.datas.get(utt2)

            val = forward(utt_xvec1, utt_xvec2, model)
            meta_dict[key] = val
    # different pairs
    for spk1, spk2 in combinations(loder.spk_list, 2):
        utt_list1 = get_utt_list(loder.raw_labels,
                                loder.spk2indexes[spk1])
        utt_list2 = get_utt_list(loder.raw_labels,
                                loder.spk2indexes[spk2])

        key_list.clear()
        for utt1 in utt_list1:
            for utt2 in utt_list2:
                key = "{}#{}".format(utt1, utt2)
                key_list.append(key)

        dif_key_pairs = random.sample(key_list,
                                      dif_pairs if dif_pairs != -1
                                      else min(len(key_list), dif_pairs))

        for key in dif_key_pairs:
            utt1, utt2 = key.split('#')

            utt_xvec1 = loder.datas.get(utt1)
            utt_xvec2 = loder.datas.get(utt2)

            val = forward(utt_xvec1, utt_xvec2, model)
            meta_dict[key] = val

    ko.save_ark(ark=os.path.join(path, '{}.ark'.format(name)),
                array_dict=meta_dict,
                scp=os.path.join(path, '{}.scp'.format(name)))


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--type',
                        action='store', choices=['test', 'generate'],
                        default='test')
    parser.add_argument('-sn', '--sam_num', type=int, default=-1)
    parser.add_argument('-dn', '--dif_num', type=int, default=-1)
    parser.add_argument('-sp', '--scp_path', type=str)

    parser.add_argument('-l', '--length', type=int)
    parser.add_argument('-n', '--name', type=str, default='xvector')
    parser.add_argument('-tp', '--tar_path', type=str, default='./')

    parser.add_argument('-mp', '--model_path', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':

    test('../scps/voxceleb_v2.scp',
         1,
         1,
         10010)

    exit(0)

    args = main()

    '%py35% extract_HDC.py -t test -sn 20 -dn 30 -l 10010 -sp ../scps/demo/test.scp'

    if args.type == 'test':
    # extract('../scps/demo/test.scp', None, 30, 20)
    #     test('../scps/demo/voxceleb_v2.scp', -1, -1, 10010)
        test(args.scp_path, args.dif_num, args.sam_num,
             args.length)
    elif args.type == 'generate':
        extract(args.scp_path,
                load(args.model_path),
                args.name,
                args.dif_num,
                args.sam_num)

