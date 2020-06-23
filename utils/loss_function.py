import re
import torch


def sf_batch_hard_triplet_loss(utts, aps, aps_inds, ans, ans_inds, alpha, div=False):
    result = torch.tensor(0.0, dtype=torch.float32).cuda()
    for anchor in utts:
        anchor_loss = alpha - torch.min(aps[aps_inds[anchor], :]) + torch.max(ans[ans_inds[anchor], :])
        result += torch.log(1 + torch.exp(anchor_loss))
    if div :
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
        anchor_loss = alpha - taps  + tans
        #print('aps', taps.item(), 'tans', tans.item())
        if anchor_loss > 0:
            cnt += 1
            result += anchor_loss
   # print('---------------------------------------------------')
    if div and cnt > 0:
        return result / cnt
    else:
        return result

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

