import numpy as np
import torch
import json
import os
import re
from itertools import combinations


def ffsemior(mat):

    P = mat @ mat.transpose(0, 1)
    PPT = P @ P.transpose(0, 1)
    alpha = torch.trace(PPT) / torch.trace(P)
    P = 1.0 / alpha * mat @ mat.transpose(0, 1)
    Q = P - torch.diag(torch.ones(mat.size(0))).cuda()
    return torch.trace(Q @ Q.transpose(0, 1))


def semior(mat):

    P = mat @ mat.transpose(0, 1)
    Q = P - torch.diag(torch.ones(mat.size(0))).cuda()
    return torch.trace(Q @ Q.transpose(0, 1))


class RandomSelector:

    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(self.seed)

    @staticmethod
    def get_random_ints(left, right, nums):
        if nums is None:
            return [i for i in range(left, right)]
        if right - left < nums:
            raise Exception('In RandomSelector.get_random_ints(): span smaller than nums.')
        ints = set()
        while len(ints) < nums:
            tmp = np.random.randint(left, right)
            if tmp in ints:
                continue
            else:
                ints.add(tmp)
        return list(ints)


def flat(mat1, mat2):
    assert mat1.size() == mat2.size()
    if mat1.squeeze().dim() == 1:
        mat1.unsqueeze_(0)
        mat2.unsqueeze_(0)
    assert len(mat1.size()) == 2
    batch = mat1.size(0)
    # abT + baT
    dif = mat1.unsqueeze(2) @ mat2.unsqueeze(1) + mat2.unsqueeze(2) @ mat1.unsqueeze(1)
    # aaT + bbT
    sam = mat1.unsqueeze(2) @ mat1.unsqueeze(1) + mat2.unsqueeze(2) @ mat2.unsqueeze(1)
    return torch.cat([dif.reshape(batch, -1), sam.reshape(batch, -1)], dim=1)


def inter(mat1, mat2):
    assert mat1.size() == mat2.size()
    if mat1.squeeze().dim() == 1:
        mat1.unsqueeze_(0)
        mat2.unsqueeze_(0)
    assert len(mat1.size()) == 2
    dif = mat1.unsqueeze(2) @ mat2.unsqueeze(1) + mat2.unsqueeze(2) @ mat1.unsqueeze(1)
    return dif.unsqueeze(1)


def overlay(mat1, mat2):
    assert mat1.size() == mat2.size()
    if mat1.squeeze().dim() == 1:
        mat1.unsqueeze_(0)
        mat2.unsqueeze_(0)
    assert len(mat1.size()) == 2
    batch = mat1.size(0)
    # abT + baT
    dif = mat1.unsqueeze(2) @ mat2.unsqueeze(1) + mat2.unsqueeze(2) @ mat1.unsqueeze(1)
    # aaT + bbT
    sam = mat1.unsqueeze(2) @ mat1.unsqueeze(1) + mat2.unsqueeze(2) @ mat2.unsqueeze(1)
    return torch.cat([dif.unsqueeze(1), sam.unsqueeze(1)], dim=1)

class Params:

    def __init__(self, json_path=None):
        if json_path is not None:
            self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def createdir(path, override=False, append=False):
    if os.path.isfile(path):
        raise Exception('Path %s has already existed as a file, but we need a dir.' % path)
    elif os.path.isdir(path):
        if append:
            pass
        elif not override:
            raise Exception('Directory %s has existed' % path)
        elif override == append:
            raise Exception('Can\'t do or not to do override and append at the same time while the %s exists.' % path)
        elif override and not append:
            os.remove(path)
            os.mkdir(path)
    else:
        os.mkdir(path)
    return path


def epoch_control(dir_path, prefix, save_epochs):
    ls = os.listdir(dir_path)
    tmpls = []
    for name in ls:
        model_path = os.path.join(dir_path, name)
        # ignore directories under the 'dirpath'
        if os.path.isfile(model_path):
            find = re.match(prefix + '_epoch_([0-9]+)', name)
            if find is None:
                continue
            else:
                tmpls.append((name, int(find.group(1))))
    tmpls = sorted(tmpls, key=lambda x:x[1])
    sub = len(tmpls) + 1 - save_epochs
    if sub <= 0:
        pass
    else:
        return [i[0] for i in tmpls[:1 - save_epochs]]


def generate_inds(spk_num, utt_num):
    ap_inds = [[] for _ in range(spk_num * utt_num)]
    an_inds = [[] for _ in range(spk_num * utt_num)]
    pos = -1

    for si in range(spk_num):
        for ui, uj in combinations(range(utt_num), 2):
            pos += 1
            ap_inds[ui + si * utt_num].append(pos)
            ap_inds[uj + si * utt_num].append(pos)

    pos = -1
    for si, sj in combinations(range(spk_num), 2):
        for ui in range(utt_num):
            for uj in range(utt_num):
                pos += 1
                an_inds[ui + si * utt_num].append(pos)
                an_inds[uj + sj * utt_num].append(pos)
    return ap_inds, an_inds


def generate_inds_mu(spk_num, utt_num):
    ap_inds = [[] for _ in range(spk_num * utt_num)]
    an_inds = [[] for _ in range(spk_num * utt_num)]
    pos = -1
    half_utt = int(utt_num / 2)
    # ap
    for s in range(spk_num):
        for u in range(half_utt):
            pos += 1
            ap_inds[u + s * utt_num].append(pos)
            ap_inds[u + s * utt_num + half_utt].append(pos)

    pos = -1
    # an
    for si, sj in combinations(range(spk_num), 2):
        for u in range(utt_num):
            pos += 1
            an_inds[u + si * utt_num].append(pos)
            an_inds[u + sj * utt_num].append(pos)
    return ap_inds, an_inds


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    torch.manual_seed(100)
    a = nn.Linear(4, 2)
    opt = torch.optim.SGD(a.parameters(), lr=10)
    inp = torch.rand(5, 4)

    def tr(mat):
        P = mat @ mat.transpose(0, 1)
        Q = P @ P.transpose(0, 1) - torch.diag(torch.ones(mat.size(0)))
        return torch.trace(Q @ Q.transpose(0, 1))

    we = a.state_dict()['weight']
    print(we, tr(we))
    res = a(inp)
    loss = torch.mean(res)

    for name, param in a.named_parameters():
        if name == 'weight':
            loss += ffsemior(param)

    loss.backward()
    opt.step()

    print(we, tr(we))

'''
tensor([[-0.3883,  0.3158, -0.2374, -0.0161],
        [ 0.1765,  0.2539, -0.2373, -0.4572]])
tensor([[-0.5947, -0.0247, -0.4948, -0.4177],
        [-0.0299, -0.0866, -0.4947, -0.8588]])
'''