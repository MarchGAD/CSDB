import torch
import torch.utils.data as Data

from kaldiio import load_scp


def parse_config(config_path):
    lines = open(config_path).readlines()
    scp_path = lines[0].strip()
    (spk_num, utt_num, anchor_size, batch_size, tot_spk_num, seed, val_spk_num, repeat_times, epochs) = \
        tuple(int(i) for i in lines[1].strip().split())
    return scp_path, spk_num, utt_num, anchor_size, batch_size, tot_spk_num, seed, val_spk_num, repeat_times, epochs


class SiSetValid(Data.Dataset):

    def __init__(self, scp_path, trial_path, use_gpu=True):
        self.datas = load_scp(scp_path)
        self.trials = []
        self.utt_buffer = {}
        tmp_set = set()
        with open(trial_path, 'r') as f:
            for line in f:
                utt1, utt2, label = line.strip().split()
                tmp_set.add(utt1)
                tmp_set.add(utt2)
                self.trials.append((utt1, utt2, int(label)))
        for utt in tmp_set:
            xvec = torch.from_numpy(self.datas.get(utt)).float()
            self.utt_buffer[utt] = xvec.cuda() if use_gpu else xvec

    def __getitem__(self, item):
        utt1, utt2, label = self.trials[item]
        return utt1, utt2, self.utt_buffer[utt1], self.utt_buffer[utt2], label

    def __len__(self):
        return len(self.trials)


class SiSet(Data.Dataset):

    def __init__(self, utt_path, tri_path, scp_path, anchor_size, buffer_size, use_gpu=True):
        self.datas = load_scp(scp_path)
        self.anchor_size = anchor_size
        self.utts = []
        self.use_gpu = use_gpu
        self.tot_line_num = -1
        self.tot_utts = -1
        self.buffer_size = buffer_size

        with open(utt_path, 'r') as up:
            for self.tot_utts, line in enumerate(up):
                pass
            self.tot_utts += 1
        self.buffer = {}
        self.utt_handler = open(utt_path, 'r')

        with open(tri_path, 'r') as tp:
            for self.tot_line_num, line in enumerate(tp):
                pass
            self.tot_line_num += 1

        self.tris_handler = open(tri_path, 'r')
        self.tris = []

    def refresh_tris_buffer(self):
        self.tris.clear()
        for _ in range(self.buffer_size):
            line = self.tris_handler.readline()
            if line == '':
                break
            utt1, utt2, label = line.strip().split()
            self.tris.append((utt1, utt2, int(label)))

    def refresh_buffer(self):
        self.buffer.clear()
        for _ in range(self.anchor_size):
            utt = self.utt_handler.readline().strip()
            self.buffer[utt] = torch.from_numpy(self.datas.get(utt)).float()
        if self.use_gpu:
            for utt in self.buffer:
                self.buffer[utt] = self.buffer[utt].cuda()

    def __getitem__(self, item):

        if item % self.buffer_size == 0:
            self.refresh_tris_buffer()

        utt1, utt2, label = self.tris[item % self.buffer_size]

        if utt1 not in self.buffer or utt2 not in self.buffer:
            self.refresh_buffer()

        try:
            return self.buffer[utt1], self.buffer[utt2], label
        except:
            raise Exception('ha')

    def __len__(self):
        return self.tot_line_num


class SNSet(Data.Dataset):

    def __init__(self, tri_path, scp_path, pre_load=True):
        self.datas = load_scp(scp_path)
        self.pre_datas = {}
        self.pre_load = pre_load
        self.tris_handler = open(tri_path, 'r')
        self.tris = []
        with open(tri_path, 'r') as f:
            for line in f:
                utt1, utt2, label = line.split()
                if utt1 not in self.datas or utt2 not in self.datas:
                    continue
                self.tris.append((utt1, utt2, int(label)))
        if self.pre_load:
            for utt in self.datas:
                self.pre_datas[utt] = torch.from_numpy(self.datas.get(utt)).float().cuda()

    def __getitem__(self, item):
        utt1, utt2, label = self.tris[item]

        if self.pre_load:
            return self.pre_datas[utt1], self.pre_datas[utt2], label
        else:
            return utt1, utt2, torch.from_numpy(self.datas.get(utt1)).float().cuda(), \
                   torch.from_numpy(self.datas.get(utt2)).float().cuda(), \
                   label

    def __len__(self):
        return len(self.tris)

if __name__ == '__main__':

    scp_path, spk_num, utt_num, anchor_size, batch_size, tot_spk_num, seed, val_spk_num, repeat_times, epochs = \
        parse_config('F:\BOOKS\DTPLDA-like-neural-backend\\result\config')

    t = SiSet(
        utt_path='F:\BOOKS\DTPLDA-like-neural-backend\\result\\train_utts',
        tri_path='F:\BOOKS\DTPLDA-like-neural-backend\\result\\train_triplets',
        scp_path='F:\BOOKS\DTPLDA-like-neural-backend\scps\demo\\test.scp',
        buffer_size=10000,
        anchor_size=anchor_size
    )

    train_loader = Data.DataLoader(
        dataset=t,
        shuffle=False,
        batch_size=batch_size,
    )
    import torch.nn as nn

    model = nn.Bilinear(512, 512, 1)
    model.cuda()

    for step, (utt1, utt2, label) in enumerate(train_loader):
        # print(utt1.size(), utt2.size(), label.size())
        print(step)

    print('*********************')
    v = SiSetValid(
        scp_path='F:\BOOKS\DTPLDA-like-neural-backend\scps\demo\\test.scp',
        trial_path='F:\BOOKS\DTPLDA-like-neural-backend\\result\\validate_trials',
    )

    valid_loader = Data.DataLoader(
        dataset=v,
        shuffle=False,
        batch_size=batch_size
    )
    # for step, (a, b, utt1, utt2, label) in enumerate(valid_loader):
    #     print(step)
