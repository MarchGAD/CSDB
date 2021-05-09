import torch
import torch.nn as nn
from utils.tools import flat, overlay, inter
import torch.nn.functional as F


class HDC(nn.Module):
    def __init__(self, mean, kaldi_lda, kaldi_plda, frozen=False, contains_bias=True, SC=False, mid_feature=None):
        super(HDC, self).__init__()
        self.frozen = frozen
        self.SC = SC
        self.contains_bias = contains_bias
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1 if contains_bias else shape[1]
        to_dim = shape[0]
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=to_dim,
                             bias=True if contains_bias else False)
        if self.contains_bias:
            self.lda.weight.data = torch.from_numpy(kaldi_lda[:, :-1]).float()
            self.lda.bias.data = torch.from_numpy(kaldi_lda[:, -1]).float() - \
                                 torch.from_numpy(kaldi_lda[:, :-1]).float() @ torch.from_numpy(mean).float()
        else:
            self.lda.weight.data = torch.from_numpy(kaldi_lda).float()
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)

        tsd = self.state_dict()
        tsd['wccn.weight'].data.copy_(torch.from_numpy(kaldi_plda['diagonalizing_transform']).float())
        tsd['wccn.bias'].data.copy_(torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float())

        self.fcs = self._make_fcs(2 * self.wccn.out_features ** 2 + self.wccn.out_features, mid_feature, 1)
    def _make_fcs(self, inp_f, mid_f, hid_layers):
        ans = nn.Sequential()
        for i in range(hid_layers):
            ans.add_module('fcs' + str(i), nn.Linear(inp_f if i == 0 else mid_f, mid_f))
            ans.add_module('relu' + str(i), nn.ReLU())
            ans.add_module('dropout' + str(i), nn.Dropout())
        ans.add_module('output', nn.Linear(mid_f if hid_layers > 0 else inp_f, 1))
        return ans

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        a = F.normalize(a)
        b = self.lda(input[1])
        b = F.normalize(b)
        a = self.wccn(a)
        b = self.wccn(b)

        tmp = flat(a, b)
        S = torch.cat([tmp, a + b], dim=1)

        # S = torch.cat([2 * a * b, a * a + b * b], 1)

        if self.SC:
            ans = self.fcs(S)
        else:
            ans = S @ self.trans.T
        ans.squeeze_()
        return ans


class LDC(nn.Module):
    def __init__(self, mean, kaldi_lda, kaldi_plda, frozen=False, contains_bias=True, SC=False, mid_feature=None):
        super(LDC, self).__init__()
        self.frozen = frozen
        self.SC = SC
        self.contains_bias = contains_bias
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1 if contains_bias else shape[1]
        to_dim = shape[0]
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=to_dim,
                             bias=True if contains_bias else False)
        if self.contains_bias:
            self.lda.weight.data = torch.from_numpy(kaldi_lda[:, :-1]).float()
            self.lda.bias.data = torch.from_numpy(kaldi_lda[:, -1]).float() - \
                                 torch.from_numpy(kaldi_lda[:, :-1]).float() @ torch.from_numpy(mean).float()
        else:
            self.lda.weight.data = torch.from_numpy(kaldi_lda).float()
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)

        tsd = self.state_dict()
        tsd['wccn.weight'].data.copy_(torch.from_numpy(kaldi_plda['diagonalizing_transform']).float())
        tsd['wccn.bias'].data.copy_(torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float())
        if self.SC:
            self.fcs = self._make_fcs(2 * self.wccn.out_features, mid_feature, 1)
        else:
            self.trans = nn.Parameter(torch.cat([
            torch.from_numpy(kaldi_plda['diagP']).float(), torch.from_numpy(kaldi_plda['diagQ']).float()
        ]), requires_grad=True)

    def _make_fcs(self, inp_f, mid_f, hid_layers):
        ans = nn.Sequential()
        for i in range(hid_layers):
            ans.add_module('fcs' + str(i), nn.Linear(inp_f if i == 0 else mid_f, mid_f))
            ans.add_module('relu' + str(i), nn.ReLU())
            ans.add_module('dropout' + str(i), nn.Dropout())
        ans.add_module('output', nn.Linear(mid_f if hid_layers > 0 else inp_f, 1))
        return ans

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        a = F.normalize(a)
        b = self.lda(input[1])
        b = F.normalize(b)
        a = self.wccn(a)
        b = self.wccn(b)

        S = torch.cat([2 * a * b, a * a + b * b], 1)

        if self.SC:
            ans = self.fcs(S)
        else:
            ans = S @ self.trans.T
        ans.squeeze_()
        return ans


class Square(nn.Module):
    def __init__(self, kaldi_lda, frozen=False, contains_bias=True, mid_feature=1000, hidden_layers=1):
        super(Square, self).__init__()
        self.frozen = frozen
        self.contains_bias = contains_bias
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1 if contains_bias else shape[1]
        mid_dim = shape[0]
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=mid_dim,
                             bias=True if contains_bias else False)
        if self.contains_bias:
            self.lda.weight.data = torch.from_numpy(kaldi_lda[:, :-1]).float()
            self.lda.bias.data = torch.from_numpy(kaldi_lda[:, -1]).float()
        else:
            self.lda.weight.data = torch.from_numpy(kaldi_lda).float()
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

        self.fcs = self._make_fcs(3 * mid_dim, mid_feature, hidden_layers)
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
        a = self.lda(input[0])
        a = F.normalize(a)
        b = self.lda(input[1])
        b = F.normalize(b)

        tmp = torch.cat([a * a, b * b, a * b], dim=1)

        ans = self.fcs(tmp)

        return ans


class NPLDA(nn.Module):
    def __init__(self, mean, kaldi_lda, kaldi_plda, frozen=True, contains_bias=True):
        super(NPLDA, self).__init__()
        self.frozen = frozen
        self.contains_bias = contains_bias
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1 if contains_bias else shape[1]
        to_dim = shape[0]
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=to_dim,
                             bias=True if contains_bias else False)
        if self.contains_bias:
            self.lda.weight.data = torch.from_numpy(kaldi_lda[:, :-1]).float()
            self.lda.bias.data = torch.from_numpy(kaldi_lda[:, -1]).float() - \
                                 torch.from_numpy(kaldi_lda[:, :-1]).float() @ torch.from_numpy(mean).float()
        else:
            self.lda.weight.data = torch.from_numpy(kaldi_lda).float()
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.P_sqrt = nn.Parameter(torch.rand(to_dim, requires_grad=True))
        self.Q = nn.Parameter(torch.rand(to_dim, requires_grad=True))

        tsd = self.state_dict()
        tsd['wccn.weight'].data.copy_(torch.from_numpy(kaldi_plda['diagonalizing_transform']).float())
        tsd['wccn.bias'].data.copy_(torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float())
        tsd['P_sqrt'].data.copy_(torch.sqrt(torch.from_numpy((kaldi_plda['diagP']))).float())
        tsd['Q'].data.copy_(torch.from_numpy(kaldi_plda['diagQ']).float())


    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        a = F.normalize(a)
        b = self.lda(input[1])
        b = F.normalize(b)
        a = self.wccn(a)
        b = self.wccn(b)

        P = self.P_sqrt * self.P_sqrt
        Q = self.Q
        S = (a * Q * a).sum(dim=1) + (b * Q * b).sum(dim=1) + 2 * (a * P * b).sum(dim=1)
        return S


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
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
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
        elif self.input_process == 'sam_compress':
            in_features = int((in_features ** 2 + in_features) * 3 / 2)

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
            elif self.input_process == 'sam_compress':
                tri_inds = torch.triu_indices(input[0].size(1), input[0].size(1))
                batch = input[0].size(0)
                # abT + baT
                dif = input[0].unsqueeze(2) @ input[1].unsqueeze(1) + input[1].unsqueeze(2) @ input[0].unsqueeze(1)
                # aaT + bbT
                sam = input[0].unsqueeze(2) @ input[0].unsqueeze(1) + input[1].unsqueeze(2) @ input[1].unsqueeze(1)
                tmp = torch.cat([dif.reshape(batch, -1),
                                 sam[:, tri_inds[0], tri_inds[1]],
                                 input[0] + input[1]], dim=1)

        return self.fcs(tmp)


class LDAWCCN(LdaNet):

    def __init__(self, kaldi_lda, kaldi_plda, mid_feature=1000, hidden_layers=1):
        super(LDAWCCN, self).__init__(kaldi_lda, input_process='out_add_vec_cat',
                                      frozen=False, contains_bias=True,
                                      mid_feature=mid_feature, hidden_layers=hidden_layers)
        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        # a = F.normalize(a)
        a = self.wccn(a)
        a = F.normalize(a)

        b = self.lda(input[1])
        # b = F.normalize(b)
        b = self.wccn(b)
        b = F.normalize(b)
        return self.basic(a, b)


class LDASUBMEAN(LdaNet):

    def __init__(self, kaldi_lda, kaldi_plda, kaldi_mean, mid_feature=1000, hidden_layers=1):
        super(LDASUBMEAN, self).__init__(kaldi_lda, input_process='out_add_vec_cat',
                                      frozen=False, contains_bias=True,
                                      mid_feature=mid_feature, hidden_layers=hidden_layers)
        wccn_fea = kaldi_lda.shape[0]
        self.mean = nn.Parameter(torch.from_numpy(kaldi_mean).float(), requires_grad=False)
        self.register_parameter('mean', self.mean)
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()

    def forward(self, *input):
        assert len(input) == 2
        (a, b) = input

        a -= self.mean
        a = self.lda(input[0])
        a = F.normalize(a)
        a = self.wccn(a)

        b -= self.mean
        b = self.lda(input[1])
        b = F.normalize(b)
        b = self.wccn(b)
        return self.basic(a, b)


class NPLDApTriplet(nn.Module):

    def __init__(self, kaldi_lda, kaldi_plda):
        super(NPLDApTriplet, self).__init__()
        wccn_fea = kaldi_lda.shape[0]
        shape = kaldi_lda.shape
        in_dim = shape[1] - 1
        self.lda = nn.Linear(in_features=in_dim,
                             out_features=shape[0],
                             bias=True)

        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()
        self.trans = nn.Parameter(torch.cat([
            torch.from_numpy(kaldi_plda['diagP']), torch.from_numpy(kaldi_plda['diagQ'])
        ]), requires_grad=True)

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        # a = self.wccn(a)
        a = F.normalize(a)
        b = self.lda(input[1])
        # b = self.wccn(b)
        b = F.normalize(b)

        S = torch.cat([2 * a * b, a * a + b * b], 1)

        ans = S @ self.trans.T
        return ans.unsqueeze(1)


class LDAWCCNcom(LdaNet):

    def __init__(self, kaldi_lda, kaldi_plda, mid_feature=1000, hidden_layers=1):
        super(LDAWCCNcom, self).__init__(kaldi_lda, input_process='out_add_vec_cat',
                                      frozen=False, contains_bias=True,
                                      mid_feature=mid_feature, hidden_layers=hidden_layers)
        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()

        self.lda_wccn = nn.Linear(in_features=self.lda.in_features, out_features=wccn_fea)
        self.lda_wccn.weight.data.copy_(self.wccn.weight.data @ self.lda.weight.data)
        self.lda_wccn.bias.data.copy_(self.lda.bias.data @ torch.transpose(self.wccn.weight.data, 0, 1)
                                      + self.wccn.bias.data)

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda_wccn(input[0])
        a = F.normalize(a)
        b = self.lda_wccn(input[1])
        b = F.normalize(b)
        return self.basic(a, b)


class LDAWCCN_compress(LdaNet):

    def __init__(self, kaldi_lda, kaldi_plda, mid_feature=1000, hidden_layers=1):
        super(LDAWCCN_compress, self).__init__(kaldi_lda, input_process='sam_compress',
                                      frozen=False, contains_bias=True,
                                      mid_feature=mid_feature, hidden_layers=hidden_layers)
        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()

    def forward(self, *input):
        assert len(input) == 2
        a = self.lda(input[0])
        # a = F.normalize(a)
        a = self.wccn(a)
        a = F.normalize(a)
        b = self.lda(input[1])
        # b = F.normalize(b)
        b = self.wccn(b)
        b = F.normalize(b)
        return self.basic(a, b)


class LDAWCCN_bn(LdaNet):

    def __init__(self, kaldi_lda, kaldi_plda, mid_feature=1000, hidden_layers=1, n_code='001'):
        super(LDAWCCN_bn, self).__init__(kaldi_lda, input_process='out_add_vec_cat',
                                      frozen=False, contains_bias=True,
                                      mid_feature=mid_feature, hidden_layers=hidden_layers)
        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()
        self.bn0 = nn.BatchNorm1d(num_features=wccn_fea)
        self.bn1 = nn.BatchNorm1d(num_features=kaldi_lda.shape[1])
        self.bn2 = nn.BatchNorm1d(num_features=kaldi_lda.shape[1])
        self.ncode = n_code

    def transform(self, x):
        if self.ncode[0] == '1':
            x = self.bn0(x)
        x = self.lda(x)
        if self.ncode[1] == '1':
            x = self.bn1(x)
        x = self.wccn(x)
        if self.ncode[2] == '1':
            x = self.bn2(x)
        return x

    def forward(self, *input):
        assert len(input) == 2
        a = self.transform(input[0])
        b = self.transform(input[1])
        return self.basic(a, b)

class LDAWCCN_2n(LdaNet):

    def __init__(self, kaldi_lda, kaldi_plda, mid_feature=1000, hidden_layers=1, n_code='001'):
        super(LDAWCCN_2n, self).__init__(kaldi_lda, input_process='out_add_vec_cat',
                                      frozen=False, contains_bias=True,
                                      mid_feature=mid_feature, hidden_layers=hidden_layers)
        wccn_fea = kaldi_lda.shape[0]
        self.wccn = nn.Linear(in_features=wccn_fea, out_features=wccn_fea)
        self.wccn.weight.data = torch.from_numpy(kaldi_plda['diagonalizing_transform']).float()
        self.wccn.bias.data = torch.from_numpy(-kaldi_plda['diagonalizing_transform'].
                                               dot(kaldi_plda['plda_mean'])).float()
        # self.bn0 = nn.BatchNorm1d(num_features=wccn_fea)
        # self.bn1 = nn.BatchNorm1d(num_features=kaldi_lda.shape[1])
        # self.bn2 = nn.BatchNorm1d(num_features=kaldi_lda.shape[1])
        self.ncode = n_code

    def transform(self, x):
        if self.ncode[0] == '1':
            x = F.normalize(x)
        x = self.lda(x)
        if self.ncode[1] == '1':
            x = F.normalize(x)
        x = self.wccn(x)
        if self.ncode[2] == '1':
            x = F.normalize(x)
        return x

    def forward(self, *input):
        assert len(input) == 2
        a = self.transform(input[0])
        b = self.transform(input[1])
        return self.basic(a, b)


if __name__ == '__main__':
    import numpy as np
    import torch
    # np.random.seed(12)
    # torch.manual_seed(1230)
    # a = Square(np.random.rand(10, 20), contains_bias=False)
    # in1 = torch.rand(3, 20)
    # in2 = torch.rand(3, 20)
    # S = a(in1, in2)
    # print(S)
    # inpu = torch.rand(4, 10)
    # a = nn.Linear(in_features=10, out_features=20, bias=True)
    # b = nn.Linear(in_features=20, out_features=20, bias=False)
    # c = nn.Linear(in_features=10, out_features=20, bias=True)
    #
    # c.weight.data.copy_((b.weight.data @ a.weight.data).float())
    # c.bias.data.copy_((a.bias.data @ torch.transpose(b.weight.data, 0, 1)).float())
    #
    # t = b(a(inpu))
    # d = c(inpu)
    #
    # print(a.bias.data.size())
    #
    # print(t)
    # print(d)
    # print(torch.sum((t - d) * (t - d)))
