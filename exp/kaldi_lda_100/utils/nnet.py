import torch
import torch.nn as nn
from utils.tools import flat, overlay, inter
import torch.nn.functional as F

class KaldiReductionCNN(nn.Module):

    def __init__(self, kaldi_lda, frozen=False, contains_bias=True):
        super(KaldiReductionCNN, self).__init__()
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1 if contains_bias else shape[1]
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=shape[0],
                             bias=True if contains_bias else False)
        if contains_bias:
            self.lda.weight.data = torch.from_numpy(kaldi_lda[:, :-1]).float()
            self.lda.bias.data = torch.from_numpy(kaldi_lda[:, -1]).float()
        else:
            self.lda.weight.data = torch.from_numpy(kaldi_lda).float()
        if frozen:
            for p in self.parameters():
                p.requires_grad = False
        self.cnnnet = CnnNet(4)

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        a = F.normalize(a)
        b = self.lda(input[1])
        b = F.normalize(b)
        return self.cnnnet(a, b)


class LdaNet(nn.Module):
    def __init__(self, kaldi_lda, input_process='out_add_vec_cat', frozen=False, contains_bias=True, mid_feature=1000, hidden_layers=1):
        super(LdaNet, self).__init__()
        self.frozen = frozen
        self.contains_bias = contains_bias
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1 if contains_bias else shape[1]
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=shape[0],
                             bias=True if contains_bias else False)
        if self.contains_bias:
            self.lda.weight.data = torch.from_numpy(kaldi_lda[:, :-1]).float()
            self.lda.bias.data = torch.from_numpy(kaldi_lda[:, -1]).float()
        else:
            self.lda.weight.data = torch.from_numpy(kaldi_lda).float()
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

        self.basic = BasicNet(shape[0], 1, input_process, init=True, mid_feature=mid_feature, hidden_layers=hidden_layers)

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        a = F.normalize(a)
        b = self.lda(input[1])
        b = F.normalize(b)
        return self.basic(a, b)



class CnnNet(nn.Module):

    def __init__(self, in_channels=2):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=5, stride=2,
                               padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1),
        )
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, *input):
        assert len(input) == 2
        def outer(a, b):
            return (input[a].unsqueeze(2) @ input[b].unsqueeze(1)).unsqueeze(1)
        x = torch.cat([outer(0, 1), outer(1, 0), outer(0, 0), outer(1, 1)], 1)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.convs(x)

        mean = torch.mean(x, [2, 3])
        return self.fcs(mean)


class BasicNet(nn.Module):
    def __init__(self, in_features, out_features, input_process='out_add_vec_cat', init=True, mid_feature=1024, hidden_layers=1):
        super(BasicNet, self).__init__()
        self.input_process = input_process
        if self.input_process is None or self.input_process == 'out_add_vec':
            in_features = 2 * in_features ** 2
        elif self.input_process == 'out_vec':
            in_features = in_features ** 2
        elif self.input_process == 'concat':
            in_features = 2 * in_features
        elif self.input_process == 'out_add_vec_dif':
            in_features = in_features ** 2
        elif self.input_process == 'out_add_vec_sam':
            in_features = in_features ** 2
        elif self.input_process == 'out_add_vec_cat':
            in_features = 2 * in_features ** 2 + in_features
        elif self.input_process == 'add':
            in_features = in_features
        elif self.input_process == 'addout':
            in_features = in_features ** 2 + in_features


        self.fcs = self._make_fcs(int(in_features), mid_feature, hidden_layers)
        if init:
            self._weight_init()

    def _make_fcs(self, inp_f, mid_f, hid_layers):
        ans = nn.Sequential()
        for i in range(hid_layers):
            ans.add_module('fcs' + str(i), nn.Linear(inp_f if i == 0 else mid_f, mid_f))
            ans.add_module('relu' + str(i), nn.ReLU())
            ans.add_module('dropout' + str(i), nn.Dropout())
        ans.add_module('output', nn.Linear(mid_f if hid_layers > 0 else inp_f, 1))
        return ans

    def _weight_init(self):
        for m in self.fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, *input):
        assert len(input) == 2
        if self.input_process is None:
            tmp = flat(input[0], input[1])
        else:
            if self.input_process == 'out_add_vec':
                tmp = flat(input[0], input[1])
            elif self.input_process == 'out_vec':
                tmp = input[0].unsqueeze(2) @ input[1].unsqueeze(1)
                tmp = tmp.reshape(input[0].size(0), -1)
            elif self.input_process == 'concat':
                tmp = torch.cat([input[0], input[1]], dim=1)
            elif self.input_process == 'out_add_vec_dif':
                tmp = input[0].unsqueeze(2) @ input[1].unsqueeze(1) + input[1].unsqueeze(2) @ input[0].unsqueeze(1)
                tmp = tmp.reshape(input[0].size(0), -1)
            elif self.input_process == 'out_add_vec_sam':
                tmp = input[0].unsqueeze(2) @ input[0].unsqueeze(1) + input[1].unsqueeze(2) @ input[1].unsqueeze(1)
                tmp = tmp.reshape(input[0].size(0), -1)
            elif self.input_process == 'out_add_vec_cat':
                tmp = flat(input[0], input[1])
                tmp = torch.cat([tmp, input[0] + input[1]], dim=1)
            elif self.input_process == 'add':
                tmp = input[0] + input[1]
            elif self.input_process == 'addout':
                tmp = input[0].unsqueeze(2) @ input[1].unsqueeze(1) + input[1].unsqueeze(2) @ input[0].unsqueeze(1) \
                    + input[0].unsqueeze(2) @ input[0].unsqueeze(1) + input[1].unsqueeze(2) @ input[1].unsqueeze(1)
                tmp = tmp.reshape(input[0].size(0), -1)
                tmp = torch.cat([tmp, input[0] + input[1]], dim=1)
#        tmp = F.normalize(tmp)
        return self.fcs(tmp)




