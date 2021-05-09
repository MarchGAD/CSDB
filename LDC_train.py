import sys
sys.path.append('.')

import os
import torch
import torch.utils.data as Data
import shutil
import argparse
from tensorboardX import SummaryWriter
from dataset.dataset_pre import SiSetValid, SNSet
from utils.nnet import BasicNet, LdaNet, KaldiReductionCNN, NPLDA, LDC
from utils.tools import Params, createdir, epoch_control
from utils.loss_function import batch_hard_triplet_loss, lifted_embedding_loss, sf_batch_hard_triplet_loss, \
    softcdet, pre_scd, like_hinge_loss, bhtri_softcdet, tri_softcdet, adjust_beta, sia_bhtriplet, sia_triplet
from score.eer_and_mindcf import Resulter
from kaldiio import load_mat
from utils.kaldiPlda2numpydict import kaldiPlda2numpydict

exp_path = './exp/ldc'
utils_path = 'utils'
score_path = 'score'
dataset_path = 'dataset'
result_path = 'result/20200509.log'
final_name = None
torch.manual_seed(990110)
lr_adjuster = {'best_valid_eer': {'epoch': -1, 'value': 1145141919.0},
               'epochs_valid_eer_no_improve': 0}
restrict_name = ['lda.weight', 'wccn.weight']


def model_name(parameters, append=True):
    if final_name is not None:
        return final_name
    return parameters.model_name + \
           '_opt_' + parameters.optimizer + \
           '_lr_' + str(parameters.lr) + \
           '_loss_' + str(parameters.loss_type) if append else parameters.model_name


def loss_name(parameters):
    return 'opt_' + parameters.optimizer + \
           '_lr_' + str(parameters.lr) + \
           '_loss_' + str(parameters.loss_type)


def lr_schedule():
    if lr_adjuster['epochs_valid_eer_no_improve'] >= 3:
        lr_adjuster['epochs_valid_eer_no_improve'] = 0
        lrs = []
        for param in opt.param_groups:
            param['lr'] /= 2.0
            lrs.append(float(param['lr']))
        print2file('Haven\'t imporved for %d epochs, down the lr to %s' % (3, str(lrs)))
        if 'softcdet' in params.loss_type:
            adjust_beta()
        if lrs[0] < params.min_lr:
            print2file('Reach the expected min_lr %s, stop training.' % str(params.min_lr))
            print2file(lr_adjuster)
            return True
    return False


def print2file(str, end='\n'):
    global log_path
    with open(log_path, 'a') as f:
        print(str, end=end, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('-c', '--cards', type=int, nargs='+', default=[0])
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    params = Params(args.config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if not params.use_gpu else str(args.cards).strip('[]')
    createdir(exp_path, override=False, append=True)

    exp_model_path = os.path.join(exp_path, model_name(params, append=False))
    final_name = model_name(params, append=os.path.exists(exp_model_path))
    exp_model_path = os.path.join(exp_path, model_name(params))
    createdir(exp_model_path, override=False)

    nnet_path = os.path.join(exp_model_path, 'nnet')
    createdir(nnet_path, override=False, append=True)
    path_of_scores = os.path.join(nnet_path, 'scores')
    createdir(path_of_scores, override=False, append=True)
    shutil.copyfile(__file__, os.path.join(nnet_path, 'train.py'))
    shutil.copyfile(args.config_path, os.path.join(nnet_path, 'config.json'))
    shutil.copytree(utils_path, os.path.join(exp_model_path, 'utils'))
    shutil.copytree(score_path, os.path.join(exp_model_path, 'score'))
    shutil.copytree(dataset_path, os.path.join(exp_model_path, 'dataset'))
    recorder = SummaryWriter(exp_model_path)
    log_path = os.path.join(exp_model_path, 'log')

    if params.net_type == 'basic':
        model = BasicNet(in_features=params.feature_dim, out_features=1, input_process=params.input_process)
    elif params.net_type == 'kaldi_lda':
        model = LdaNet(load_mat(params.kaldi_transform_path), frozen=params.frozen, contains_bias=params.contains_bias, 
                       mid_feature=2048 if 'mid_feature' not in params.dict else params.mid_feature,
                       hidden_layers=1 if 'hidden_layers' not in params.dict else params.hidden_layers,
                       input_process='out_add_vec_cat' if 'input_process' not in params.dict else params.input_process)
    elif params.net_type == 'kaldi_cnn':
        model = KaldiReductionCNN(load_mat(params.kaldi_transform_path), frozen=params.frozen, contains_bias=params.contains_bias)
    elif params.net_type == 'nplda':
        model = NPLDA(load_mat(params.kaldi_mean), load_mat(params.kaldi_transform_path), frozen=params.frozen, contains_bias=params.contains_bias,
                      kaldi_plda=kaldiPlda2numpydict(params.kaldi_plda_path))
    elif params.net_type == 'ldc':
        model = LDC(
            load_mat(params.kaldi_mean),
            load_mat(params.kaldi_transform_path),
            kaldiPlda2numpydict(params.kaldi_plda_path),
            frozen=params.frozen,
            SC=params.SC,
            mid_feature=params.mid_feature if params.SC else None
        )

    print2file(model)
    if params.use_gpu:
        model = model.cuda()
    batch_size = params.batch_size
    print2file('Dealing with data.')
    train_sl = SNSet(
        tri_path=params.tri_path,
        scp_path=params.train_path,
        pre_load=False
    )
    valid_sl = SNSet(
        tri_path=params.valid_path,
        scp_path=params.train_path,
        pre_load=False
    )


    print2file('Using strategy %s' % params.strategy)
    print2file('The validate_num is %d, the train_num is %d' % (len(valid_sl), len(train_sl)))
    batch1list = {'spkrand', 'multi_spkrand', 'batch_hard'}
    train_loader = Data.DataLoader(
        dataset=train_sl,
        batch_size=batch_size,
    )
    validate_loader = Data.DataLoader(
        dataset=valid_sl,
        batch_size=batch_size,
        shuffle=False
    )

    if params.loss_type == 'batch_hard_triplet':
        loss_function = batch_hard_triplet_loss
    elif params.loss_type == 'lifted_embedding':
        loss_function = lifted_embedding_loss
    elif params.loss_type == 'sf_bh_triplet':
        loss_function = sf_batch_hard_triplet_loss
    elif params.loss_type == 'softcdet':
        pre_scd(model=model, beta=params.betas)
        model.cuda()
        loss_function = softcdet
    elif params.loss_type == 'like_hinge':
        loss_function = like_hinge_loss
    elif params.loss_type == 'siamese_triplet':
        loss_function = sia_triplet
    elif params.loss_type == 'siamese_bhtriplet':
        loss_function = sia_bhtriplet



    if params.optimizer == 'sgd':
        opt = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=params.lr, weight_decay=params.weight_decay)
    elif params.optimizer == 'adam':
        opt = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=params.lr, weight_decay=params.weight_decay)
    elif params.optimizer == 'adadelta':
        opt = torch.optim.Adadelta(filter(lambda p:p.requires_grad, model.parameters()), lr=params.lr, weight_decay=params.weight_decay)

    gstep = 0
    alpha = 1
    if 'alpha' in params.dict:
        alpha = params.alpha

    for tot_cnt in range(params.epoch):
        model.train()
        for step, (utt1, utt2, a, b, y) in enumerate(train_loader):
            opt.zero_grad()
            a = a.float().squeeze()
            b = b.float().squeeze()
            y = y.squeeze().float().unsqueeze(1)

            sd = model(a, b)
            if params.loss_type == 'softcdet':
                loss = loss_function(sd, y, alpha, params.betas)
            elif params.loss_type == 'like_hinge':
                loss = loss_function(sd, y, alpha)

            recorder.add_scalar(loss_name(params), loss.item(), global_step=gstep)
            if step % params.show_step == 0:
                print2file('epoch %d, step %d, loss: %f' % (tot_cnt + 1, step, loss.item()), end='')
                print2file('.')
            if loss != 0:
                loss.backward()
                opt.step()
        model.eval()
        model_path = os.path.join(nnet_path, model_name(params) + '_epoch_' + str(tot_cnt))

        if 'maintain_epochs' in params.dict:
            del_ls = epoch_control(nnet_path, model_name(params), params.maintain_epochs)
            if del_ls is None:
                pass
            else:
                for name in del_ls:
                    print2file('Delete model %s.' % name)
                    os.remove(os.path.join(nnet_path, name))

        print2file('Saving model to %s,' % model_path, end='')
        torch.save(model, model_path)
        print2file('finished.')
        print2file('Calculating the valid_eer and valid_dcf.')
        with torch.no_grad():
            validate_scores_path = os.path.join(path_of_scores, 'validate_scores_' + str(tot_cnt))
            test_model = torch.load(model_path)
            for cnt, (utt1, utt2, a, b, y) in enumerate(validate_loader):
                a = a.float().squeeze()
                b = b.float().squeeze()
                y = torch.unsqueeze(y.float(), 1)
                if params.use_gpu:
                    a = a.cuda()
                    b = b.cuda()
                    y = y.cuda()
                answer = test_model(a, b)
                if params.use_gpu:
                    answer = answer.cpu()
                with open(validate_scores_path, 'a') as f:
                    for u1, u2, score in zip(utt1, utt2, answer):
                        f.write(u1 + ' ' + u2 + ' ' + str(float(score)) + '\n')

            res = Resulter(validate_scores_path, params.valid_path).compute_score()
            eer = res['EER:']
            minDCF1 = res['minDCF(p-target=0.01):']
            minDCF2 = res['minDCF(p-target=0.001):']

            with open(os.path.join(nnet_path, 'valid_eer_mindcf'), 'a') as f:
                valid_result = str(tot_cnt) + ' ' + str(eer) + ' ' + str(minDCF1) + ' ' + str(minDCF2)
                print2file('valid_eer: %s, '
                           '\nvalid_minDCF(p-target=0.01): %s,\nvalid_minDCF(p-target=0.001): %s'
                           % (eer, minDCF1, minDCF2))
                print2file('----------------------------------------')
                f.write(valid_result + '\n')
            if lr_schedule():
                break
            score_path = os.path.join(exp_model_path, 'score')
            os.system('python %s/get_scores.py -e %d -rp %s -c %d' %
                      (score_path, tot_cnt, os.path.abspath(os.path.join(exp_model_path, 'results')),
                       int(os.environ['CUDA_VISIBLE_DEVICES'])))