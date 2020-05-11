from kaldiio import load_scp
from torch.utils.data import Dataset
import torch
import re
from itertools import combinations
from utils.tools import RandomSelector


class KaldiLoader(Dataset):

    def __init__(self, scp_path, use_lda=False, lda_loader=None):
        super(KaldiLoader, self).__init__()
        self.scp_path = scp_path
        self.datas = load_scp(scp_path)
        self.raw_labels = sorted([_ for _ in self.datas.keys()])
        self.use_lda = use_lda
        self.spk_labels = list()
        self.spk2indexes = dict()
        self.lda_loader = lda_loader
        _ = -1

        for rl in self.raw_labels:
            _ += 1
            spk = re.match(r'(.*?)-.*', rl).group(1)
            self.spk_labels.append(spk)
            if spk not in self.spk2indexes:
                self.spk2indexes[spk] = []
            self.spk2indexes[spk].append(_)

        self.spk_list = sorted(list(self.spk2indexes.keys()))
        self.spk_num = len(self.spk2indexes)
        self.spk2num = {_: i for _, i in zip(self.spk2indexes.keys(), range(self.spk_num))}
        self.num2spk = {self.spk2num[_]: _ for _ in self.spk2num.keys()}
        self.num_labels = torch.tensor([self.spk2num[label] for label in self.spk_labels])

        if self.use_lda:
            if self.lda_loader is not None:
                if not self.lda_loader.trained:
                    raise Exception('Please trained the LDALoader before.')
            else:
                raise Exception('`use_lda` is True but have no `LDALoader` to implement lda.')

    def __getitem__(self, item):
        raw_label = self.raw_labels[item]
        num_label = self.num_labels[item]
        xvec = torch.from_numpy(self.datas.get(raw_label))
        if self.use_lda:
            xvec = self.lda_loader.transform(xvec.unsqueeze(1))
            xvec = torch.from_numpy(xvec.squeeze())
        return xvec, num_label

    def get_xvec(self, item):
        raw_label = self.raw_labels[item]
        xvec = torch.from_numpy(self.datas.get(raw_label))
        if self.use_lda:
            xvec = self.lda_loader.transform(xvec.unsqueeze(1))
            xvec = torch.from_numpy(xvec.squeeze())
        return xvec

    def get_num_label(self, item):
        num_label = self.num_labels[item]
        return num_label

    def __len__(self):
        return len(self.spk_labels)

class SiameseSet(KaldiLoader):
    def __init__(self, scp_path, use_lda=False, lda_loader=None, utt_per_spk=None, seed=19971222,
                 spk_num=None, pre_load=False, strategy='totrand', ratio=0.1):
        super(SiameseSet, self).__init__(scp_path, use_lda, lda_loader)
        self.rs = RandomSelector(seed=seed)
        self.pre_load = pre_load
        self.pre_load_data = dict()
        self.strategy = strategy
        self.pairs = []
        self.spk_num = spk_num
        self.utt_per_spk = utt_per_spk
        self.spk_selected_utts = {}
        self.ratio = ratio

        self.choose_strategy()
        if self.pre_load:
            self.pre_load_data2dict()

    def pre_load_data2dict(self):
        keys2load = set()
        if self.pre_load and self.strategy == 'totrand':
            for pairs in self.pairs:
                keys2load.add(pairs[2])
                keys2load.add(pairs[3])

        elif self.strategy == 'spkrand':
            keys2load = set(self.raw_labels)

        for key in keys2load:
            xvec = torch.from_numpy(self.datas.get(key))
            if self.use_lda:
                xvec = self.lda_loader.transform(xvec.unsqueeze(0))
                xvec = torch.from_numpy(xvec.squeeze())
            self.pre_load_data[key] = xvec

    def choose_strategy(self):
        if self.strategy == 'totrand':
            tmp_spk_list = [self.spk_list[_] for _ in self.rs.get_random_ints(0, len(self.spk_list), self.spk_num)]
            self.spk_selected_utts = {
                spk: [
                    self.raw_labels[self.spk2indexes[spk][_]] for _ in
                    self.rs.get_random_ints(0,
                                            len(self.spk2indexes[spk]),
                                            self.utt_per_spk)
                ]
                for spk in tmp_spk_list
            }
            tmp = set()
            for spk1, spk2 in combinations(tmp_spk_list, 2):
                self.pairs.extend(self.generate_pairs(spk1, spk2))
                for spk in (spk1, spk2):
                    if spk not in tmp:
                        self.pairs.extend(self.generate_pairs(spk, spk))
        elif self.strategy == 'spkrand':
            # for spk1, spk2 in combinations(self.spk_list, 2):
            #     self.pairs.append((spk1, spk2))
            # isomor = int(len(self.pairs) * self.ratio)
            # for spk in self.spk_list:
            #     for i in range(isomor):
            #         self.pairs.append((spk, spk))
            pass
        elif self.strategy == 'multi_spkrand':
            assert self.spk_num is not None and self.utt_per_spk is not None
            pass
        elif self.strategy == 'batch_hard':
            pass
        else:
            raise Exception('Unrecognized strategy %s' % self.strategy)

    def generate_pairs(self, spk1, spk2):
        for a in self.spk_selected_utts[spk1]:
            for b in self.spk_selected_utts[spk2]:
                yield (spk1, spk2, a, b)

    def randomly_get_spk_pairs(self):
        import numpy as np
        a = np.random.randint(0, 100) > (100 * self.ratio)
        if a:
            return [self.spk_list[i] for i in self.rs.get_random_ints(0, len(self.spk_list), 2)]
        else:
            spk = self.spk_list[np.random.randint(0, len(self.spk_list))]
            return spk, spk

    def __len__(self):
        if self.strategy == 'totrand':
            return len(self.pairs)
        else:
            return 1145141919

    def __getitem__(self, item):

        def get_utts_per_spk(spk, utt_num):
            length = len(self.spk2indexes[spk])
            return [self.raw_labels[self.spk2indexes[spk][_]]
                    for _ in self.rs.get_random_ints(0, length, min(length, utt_num))
                    ]

        if self.strategy == 'totrand':
            pair = self.pairs[item]
            label = torch.tensor(1 if pair[0] == pair[1] else -1).float()
            if self.pre_load:
                xvec_a = self.pre_load_data[pair[2]]
                xvec_b = self.pre_load_data[pair[3]]
            else:
                xvec_a = torch.from_numpy(self.datas.get(pair[2]))
                xvec_b = torch.from_numpy(self.datas.get(pair[3]))
                if self.use_lda:
                    # Attention: the dim in function 'unsqueeze' must be set to 0 here
                    xvec_a = self.lda_loader.transform(xvec_a.unsqueeze(0))
                    xvec_b = self.lda_loader.transform(xvec_b.unsqueeze(0))
                    xvec_a = torch.from_numpy(xvec_a.squeeze())
                    xvec_b = torch.from_numpy(xvec_b.squeeze())
            return pair[0], pair[1], pair[2], pair[3], xvec_a, xvec_b, label
        elif self.strategy == 'spkrand':
            spk1, spk2 = self.randomly_get_spk_pairs()
            if self.utt_per_spk is None:
                spk1list = [self.raw_labels[_] for _ in self.spk2indexes[spk1]]
                spk2list = [self.raw_labels[_] for _ in self.spk2indexes[spk2]]
            else:
                spk1list = get_utts_per_spk(spk1, self.utt_per_spk)
                spk2list = get_utts_per_spk(spk2, self.utt_per_spk)
            tmp_pairs = [(a, b) for a in spk1list for b in spk2list]
            vec_a_index = [t[0] for t in tmp_pairs]
            vec_b_index = [t[1] for t in tmp_pairs]
            if self.pre_load:
                xvecs_a = torch.cat([self.pre_load_data[key].unsqueeze(0) for key in vec_a_index], dim=0)
                xvecs_b = torch.cat([self.pre_load_data[key].unsqueeze(0) for key in vec_b_index], dim=0)
            else:
                if self.use_lda and self.lda_loader is not None:

                    def lda_transform(key):
                        xvec = torch.from_numpy(self.datas.get(key))
                        xvec = self.lda_loader.transform(xvec.unsqueeze(0))
                        return torch.from_numpy(xvec)

                    xvecs_a = torch.cat([lda_transform(key) for key in vec_a_index], dim=0)
                    xvecs_b = torch.cat([lda_transform(key) for key in vec_b_index], dim=0)
                else:
                    xvecs_a = torch.tensor([self.datas.get(key) for key in vec_a_index])
                    xvecs_b = torch.tensor([self.datas.get(key) for key in vec_b_index])
            label = torch.ones(len(tmp_pairs)).float() * (1 if spk1 == spk2 else -1)
            return spk1, spk2, vec_a_index, vec_b_index, xvecs_a, xvecs_b, label
        elif self.strategy == 'multi_spkrand':
            spks_inds = self.rs.get_random_ints(0, len(self.spk_list), self.spk_num)
            spks = [self.spk_list[i] for i in spks_inds]
            utts = []
            for spk in spks:
                utts.extend(get_utts_per_spk(spk, self.utt_per_spk))
            utts = utts if len(utts) % 2 == 0 else utts[:-1]
            mid = int(len(utts) / 2)
            rand_indexes = torch.randperm(len(utts))
            spk1s = []
            spk2s = []
            xvecs_a = None
            xvecs_b = None
            labels = []
            for i in range(mid):
                utt1 = utts[int(rand_indexes[i])]
                utt2 = utts[int(rand_indexes[i + mid])]
                spk1 = re.match(r'(.*?)-.*', utt1).group(1)
                spk2 = re.match(r'(.*?)-.*', utt2).group(1)
                label = 1 if spk1 == spk2 else -1
                labels.append(label)
                spk1s.append(spk1)
                spk2s.append(spk2)
                if self.use_lda and self.lda_loader is not None:
                    def lda_transform(key):
                        xvec = torch.from_numpy(self.datas.get(key))
                        xvec = self.lda_loader.transform(xvec.unsqueeze(0))
                        return torch.from_numpy(xvec)
                    xvec1 = lda_transform(utt1)
                    xvec2 = lda_transform(utt2)
                else:
                    xvec1 = torch.from_numpy(self.datas.get(utt1)).unsqueeze(0)
                    xvec2 = torch.from_numpy(self.datas.get(utt2)).unsqueeze(0)
                if xvecs_a is None:
                    xvecs_a = xvec1
                else:
                    xvecs_a = torch.cat([xvecs_a, xvec1], dim=0)
                if xvecs_b is None:
                    xvecs_b = xvec2
                else:
                    xvecs_b = torch.cat([xvecs_b, xvec2], dim=0)
            labels = torch.tensor(labels).float()
            return spk1s, spk2s, utts[:mid], utts[mid:], xvecs_a, xvecs_b, labels
        elif self.strategy == 'batch_hard':
            spks_inds = self.rs.get_random_ints(0, len(self.spk_list), self.spk_num)
            spks = [self.spk_list[i] for i in spks_inds]
            utts = []
            spk2utt = {}
            utt2xvec = {}
            for spk in spks:
                tmp = get_utts_per_spk(spk, self.utt_per_spk)
                spk2utt[spk] = tmp
                utts.extend(tmp)
            for utt in utts:
                utt2xvec[utt] = torch.from_numpy(self.datas.get(utt)).unsqueeze(0)
            # the following block can be optimized
            a2p_inds = {}
            for utt in utts:
                a2p_inds[utt] = []
            ApS = None
            PS = None
            pos = -1
            for spk in spks:
                for u1, u2 in combinations(spk2utt[spk], 2):
                    xa = utt2xvec[u1]
                    xb = utt2xvec[u2]
                    if ApS is None:
                        ApS = xa
                    else:
                        ApS = torch.cat([ApS, xa], dim=0)
                    if PS is None:
                        PS = xb
                    else:
                        PS = torch.cat([PS, xb], dim=0)
                    pos += 1
                    a2p_inds[u1].append(pos)
                    a2p_inds[u2].append(pos)
            a2n_inds = {}
            for utt in utts:
                a2n_inds[utt] = []
            AnS = None
            NS = None
            pos = -1
            for spk1, spk2 in combinations(spks, 2):
                for u1 in spk2utt[spk1]:
                    for u2 in spk2utt[spk2]:
                        xa = utt2xvec[u1]
                        xb = utt2xvec[u2]
                        if AnS is None:
                            AnS = xa
                        else:
                            AnS = torch.cat([AnS, xa], dim=0)
                        if NS is None:
                            NS = xb
                        else:
                            NS = torch.cat([NS, xb], dim=0)
                        pos += 1
                        a2n_inds[u1].append(pos)
                        a2n_inds[u2].append(pos)
           
            return utts, a2p_inds, ApS, PS, a2n_inds, AnS, NS


class KaldiTester(Dataset):

    def __init__(self, scp_path, trials, use_gpu=False):
        super(KaldiTester, self).__init__()
        self.datas = load_scp(scp_path)
        self.use_gpu = use_gpu
        self.trials = trials

    def __getitem__(self, item):
        utt1, utt2, is_target = self.trials[item]
        xvec1 = torch.from_numpy(self.datas.get(utt1))
        xvec2 = torch.from_numpy(self.datas.get(utt2))
        if self.use_gpu:
            xvec1 = xvec1.cuda()
            xvec2 = xvec2.cuda()
        return xvec1, xvec2, utt1, utt2

    def __len__(self):
        return len(self.trials)

