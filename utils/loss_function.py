import re
import torch

import torch.nn as nn


def sia_bhtriplet(output, target, alpha, betas):
    target.squeeze_()
    return alpha - torch.min(output * target) + torch.max(output * (1 - target))


def sia_triplet(output, target, alpha, betas):
    target.squeeze_()
    return alpha - (output * target).sum() / (target).sum() + \
           (output * (1 - target)).sum() / (1 - target).sum()
    # loss = sum(losses) / len(losses)
    # return loss


def tri_softcdet(utts, aps, aps_inds, ans, ans_inds, alpha, div=True, betas=None):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    cnt = 0
    if not has_pre_scd:
        raise Exception('Run pre_scd() before use softcdet loss.')
    for anchor in utts:
        taps = aps[aps_inds[anchor], :]
        tans = ans[ans_inds[anchor], :]
        losses = [(sigmoid(alpha * (threshold[i][1] - taps)).sum() / len(taps) +
                   threshold[i][0] * sigmoid(alpha * (tans - threshold[i][1])).sum() / len(tans))
                  for i in threshold]
        anchor_loss = sum(losses) / len(losses)
        # anchor_loss = alpha - taps + tans
        # print('aps', taps.item(), 'tans', tans.item())
        if anchor_loss > 0:
            cnt += 1
            result += anchor_loss
    return result / (cnt if cnt > 0 else 1)


def bhtri_softcdet(utts, aps, aps_inds, ans, ans_inds, alpha, div=True, betas=None):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    if not has_pre_scd:
        raise Exception('Run pre_scd() before use softcdet loss.')
    for anchor in utts:
        taps = torch.min(aps[aps_inds[anchor], :])
        tans = torch.max(ans[ans_inds[anchor], :])
        losses = [(sigmoid(alpha * (threshold[i][1] - taps)) +
                   threshold[i][0] * sigmoid(alpha * (tans - threshold[i][1])))
                  for i in threshold]
        anchor_loss = sum(losses) / len(losses)
        # anchor_loss = alpha - taps + tans
        # print('aps', taps.item(), 'tans', tans.item())
        if anchor_loss > 0:
            result += anchor_loss.squeeze()
    return result / len(utts)


def batch_hard_v2(aps, aps_inds, ans, ans_inds, alpha):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    cnt = 0
    for p, n in zip(aps_inds, ans_inds):
        taps = torch.min(aps[p])
        tans = torch.max(ans[n])
        anchor_loss = alpha - taps + tans
        if anchor_loss > 0:
            cnt += 1
            result += anchor_loss
    return result / cnt


def sf_batch_hard_triplet_loss(utts, aps, aps_inds, ans, ans_inds, alpha, div=False):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    for anchor in utts:
        anchor_loss = alpha - torch.min(aps[aps_inds[anchor], :]) + torch.max(ans[ans_inds[anchor], :])
        result += torch.log(1 + torch.exp(anchor_loss))
    if div:
        return result / len(utts)
    else:
        return result


def lifted_embedding_loss(utts, aps, aps_inds, ans, ans_inds, alpha, div=False):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    cnt = 0

    for anchor in utts:
        tmp = torch.log(torch.sum(torch.exp(alpha - aps[aps_inds[anchor], :]))) + \
              torch.log(torch.sum(torch.exp(ans[ans_inds[anchor], :])))
        if tmp > 0:
            result += tmp
            cnt += 1
    if cnt > 0 and div:
        return result / cnt
    else:
        return result


def batch_hard_triplet_loss(utts, aps, aps_inds, ans, ans_inds, alpha, div=False):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    cnt = 0
    for anchor in utts:
        taps = torch.min(aps[aps_inds[anchor], :])
        tans = torch.max(ans[ans_inds[anchor], :])
        anchor_loss = alpha - taps + tans
        # print('aps', taps.item(), 'tans', tans.item())
        if anchor_loss > 0:
            cnt += 1
            result += anchor_loss
    # print('---------------------------------------------------')
    if div and cnt > 0:
        return result / cnt
    else:
        return result


sigmoid = None
has_pre_scd = False
threshold = None


def adjust_beta():
    global threshold
    for i in threshold:
        threshold[i][0] = min(threshold[i][0] * 2, threshold[i][2])


def pre_scd(model, beta):
    global sigmoid, has_pre_scd, threshold
    sigmoid = nn.Sigmoid()
    has_pre_scd = True
    threshold = {}
    cnt = 0
    for i in beta:
        cnt += 1
        threshold[cnt] = [beta, nn.Parameter(torch.zeros(1, requires_grad=True)), i]
        model.register_parameter('Th{}'.format(cnt), threshold[cnt][1])


def softcdet(output, target, alpha, betas):
    target.squeeze_()
    if not has_pre_scd:
        raise Exception('Run pre_scd() before use softcdet loss.')
    # losses = [((sigmoid(alpha * (threshold[i] - output)) * target).sum() / (target.sum()) + i * (
    #         sigmoid(alpha * (output - threshold[i])) * (1 - target)).sum() / ((1 - target).sum()))
    #           for i in betas]



    losses = [((sigmoid(alpha * (threshold[i][1] - output)) * target).sum() / (target.sum()) +
               threshold[i][0] * (sigmoid(alpha * (output - threshold[i][1])) * (1 - target)).sum()
               / ((1 - target).sum()))
              for i in threshold]
    loss = sum(losses) / len(losses)
    return loss


def like_hinge_loss(mat, label, alpha, strip=True):
    '''
    :param mat: Results of network.
    :param label: Labels of input pairs.
    :param alpha: A hyper-parameter.
    :param strip: If True, ignore easy pairs whose absolute value greater than alpha.
    :return:
    '''
    assert mat.size() == label.size()
    tmp = alpha - label * mat
    if strip:
        k = tmp[tmp > 0]
        if torch.sum(k) == 0:
            tmp[tmp < 0] = 0
            ans = torch.mean(tmp[tmp <= 0])
        else:
            ans = torch.mean(k)
    else:
        tmp[tmp < 0] = 0
        ans = torch.mean(tmp)
    return ans


# def one_batch_hard_triplet_loss(utts, aps_size, scores, aps_inds, ans_inds, alpha, div=False):
#     result = torch.tensor(0.0, dtype=torch.float32).cuda()
#     cnt = 0
#     for anchor in utts:
#         taps = torch.min(scores[aps_inds[anchor], :])
#         tans = torch.max(scores[[i + aps_size for i in ans_inds[anchor]], :])
#         anchor_loss = alpha - taps  + tans
#         #print('aps', taps.item(), 'tans', tans.item())
#         if anchor_loss > 0:
#             cnt += 1
#             result += anchor_loss
#    # print('---------------------------------------------------')
#     if div and cnt > 0:
#         return result / cnt
#     else:
#         return result

if __name__ == '__main__':
    # import torch.nn as nn
    # import torch
    #
    # torch.manual_seed(123)
    # model = nn.Sequential(
    #     nn.Linear(20, 10),
    #     nn.Linear(10, 1)
    # )
    # pre_scd(model, [1])
    # model.cuda()
    # opt = torch.optim.Adam(model.parameters(), lr=0.1)
    # inp = torch.rand(4, 20).cuda()
    #
    # for i in range(10):
    #     ans = model(inp)
    #     taps = torch.max(ans)
    #     tans = torch.min(ans)
    #     alpha = 5
    #     a = sigmoid(alpha * (threshold[1] - taps))
    #     b = sigmoid(alpha * (tans - threshold[1]))
    #     loss = sigmoid(alpha * (threshold[1] - taps)) + sigmoid(alpha * (tans - threshold[1]))
    #
    #     # print((alpha * a * (1 - a)).item(),  (-alpha * b * (1 - b)).item())
    #     opt.zero_grad()
    #     loss.backward()
    #     opt.step()
    #
    #     print(taps.item(), tans.item())
    #     print(ans.reshape(-1))
    #     print('==============')
    # print(threshold)
    import torch.nn as nn
    import torch
    model = nn.Linear(20, 10)
    betas = [4, 10]
    pre_scd(model=model, beta=betas)
    model.cuda()
    print(threshold.keys())
    print('========')
    for i in range(5):
        adjust_beta(model, beta=betas)
        print(threshold.keys())
        print('========')