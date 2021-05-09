import sys
import os
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirpath + '/../')

import torch.utils.data as Data
import multiprocessing as mp
from utils.nnet import *
from kaldiio import load_scp
from dataset.kaldi_reader import KaldiTester


class Scorer:

    def __init__(self, model_path, trials_path, scp_path,
                 batch_size=1, use_gpu=False,
                 mgr=None, processes=None, check=False):
        if processes is not None and processes < 0:
            raise Exception('`processes` should be a positive integer.')
        self.check = check
        self.model_path = model_path
        self.scp_path = scp_path
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.processes = int(processes) if processes is not None else None
        self.trials = []
        self.data = load_scp(scp_path)
        self.shared_scores = mgr.dict()
        self.miss = 0

        with open(trials_path, 'r') as f:
            for line in f:
                tmp = line.strip().split()
                if len(tmp) != 3:
                    continue
                else:
                    utt1, utt2, is_target = tmp
                    self.trials.append((utt1, utt2, is_target))
        self.tot = len(self.trials)

    def get_all_scores(self):
        if self.processes is None or self.processes == 0:
            self.get_scores(0, len(self.trials))
        else:
            block = int(len(self.trials) / self.processes)
            jobs = [mp.Process(target=self.get_scores,
                               args=(i * block, (i + 1) * block if i < self.processes - 1 else None))
                    for i in range(self.processes)]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
        # print('Finished calculate score.')

    def save_scores(self, save_path, override=True):
        if os.path.exists(save_path) and override is False:
            raise Exception('File/Directory %s exists, please choose another path or set `override` to True' % save_path)
        with open(save_path, 'w') as f:
            shared_scores = dict(self.shared_scores)
            for pair in shared_scores:
                utt1, utt2 = pair
                f.write(utt1 + ' ' + utt2 + ' ' + str(shared_scores[pair]) + '\n')

    def get_scores(self, start, end):
        model = torch.load(self.model_path)
        model.eval()
        if self.use_gpu:
            model = model.cuda()
        kt = KaldiTester(self.scp_path, self.trials[start:end],
                         use_gpu=self.use_gpu)
        if self.check:
            kt.clear()
            self.trials = kt.trials
        self.miss = kt.miss
        dataloader = Data.DataLoader(
            dataset=kt,
            batch_size=self.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for cnt, (mat1, mat2, utt1, utt2) in enumerate(dataloader):
                mat1 = mat1.float()
                mat2 = mat2.float()
                scores = model(mat1, mat2).squeeze()
                if self.use_gpu:
                    scores = scores.cpu()
                for utt1, utt2, score in zip(utt1, utt2, scores):
                    self.shared_scores[(utt1, utt2)] = float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rp', '--result_path', type=str, default=None)
    parser.add_argument('-e', '--epochs', type=int, nargs='+')
    parser.add_argument('-ug', '--use_gpu', type=bool, default=True)
    parser.add_argument('-c', '--cards', type=int, nargs='+', default=[0])
    parser.add_argument('-t', '--trial', type=str, default=dirpath + '/trials')
    parser.add_argument('-chc', '--check', type=bool, default=False)
    parser.add_argument('-nb', '--num_of_process', type=int, default=0)
    parser.add_argument('-sp', '--scp_path', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    from utils.tools import Params
    import time
    import re
    import argparse
    args = main()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if not args.use_gpu else str(args.cards).strip('[]')
    params = Params(dirpath + '/../nnet/config.json')
    models_path = dirpath + '/../nnet'
    model_path = None
    for epoch in args.epochs:
        for i in os.listdir(models_path):
            a = re.search(r'epoch_([0-9]*)$', i)
            if a is None:
                continue
            elif int(a.group(1)) == epoch:
                model_path = os.path.join(models_path, i)

        scorer = Scorer(model_path=model_path,
                        scp_path=params.test_scp_path if args.scp_path is None else args.scp_path,
                        trials_path=args.trial, batch_size=50,
                        use_gpu=args.use_gpu, 
                        mgr=mp.Manager(), processes=args.num_of_process)
        # pre = time.time()
        scorer.get_all_scores()
        fr = open('./result' if args.result_path is None else args.result_path, 'a')

        basename = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))
        print('***********************************', file=fr)
        print('tot is %d, miss %d, %f' % (scorer.tot, scorer.miss, scorer.miss / scorer.tot), file=fr)
        print('%s/%s' % (basename, params.model_name), file=fr)
        print('trial:%s' % (args.trial), file=fr)
        print(time.ctime(), file=fr)
        print('Epoch %d' % epoch, file=fr)
        # print('Cost time %s' % (time.time() - pre), file=fr)
        # print('Saving Scores, ', end='', file=fr)
        scorer.save_scores(dirpath + '/alpha0.5.txt')
        from score.eer_and_mindcf import Resulter
        a = Resulter(dirpath + '/alpha0.5.txt',
                     trial_file=None if args.check else args.trial,
                     trials=scorer.trials if args.check else None)
        print(a.compute_score(), file=fr)
        print('-----------------------------------', file=fr)
        fr.close()

