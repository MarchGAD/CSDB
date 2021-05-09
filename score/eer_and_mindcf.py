class Resulter:

    def __init__(self, scores, trial_file, trials=None):
        self.scores = {}
        self.trials = {}

        assert (trial_file is None) ^ (trials is None)

        with open(scores, 'r') as f:
            for line in f:
                tmp = line.strip().split()
                if len(tmp) != 3:
                    continue
                utt1, utt2, score = tmp
                self.scores[(utt1, utt2)] = float(score)

        if trials is None:
            with open(trial_file, 'r') as f:
                for line in f:
                    tmp = line.strip().split()
                    if len(tmp) != 3:
                        continue
                    utt1, utt2, target = tmp
                    self.trials[(utt1, utt2)] = target
        else:
            for i in trials:
                utt1, utt2, target = i
                self.trials[(utt1, utt2)] = target

        assert len(self.scores) == len(self.trials)
        # # assume their relationship is containment
        # if len(self.scores) < len(self.trials):
        #     for key in self.trials:
        #         if key not in self.scores:
        #             self.trials.pop(key)
        # elif len(self.scores) > len(self.trials):
        #     for key in self.scores:
        #         if key not in self.trials:
        #             self.scores.pop(key)

    def compute_eer(self):
        target_scores = []
        non_target_scores = []
        for tup in self.trials:
            if self.trials[tup] == 'target' or self.trials[tup] == '1':
                target_scores.append(self.scores[tup])
            else:
                non_target_scores.append(self.scores[tup])
        target_scores = sorted(target_scores)
        non_target_scores = sorted(non_target_scores)
        target_size = len(target_scores)
        non_target_size = len(non_target_scores)
        target_pos = target_size
        for target_pos in range(target_size):
            non_target_n = int(non_target_size * target_pos * 1.0 / target_size)
            non_target_pos = max(0, non_target_size - 1 - non_target_n)
            if non_target_scores[non_target_pos] < target_scores[target_pos]:
                break
        threshold = target_scores[target_pos]
        eer = target_pos * 1.0 / target_size * 100
        return eer, threshold

    def compute_min_dcf(self, p_target=0.01, c_miss=1, c_fa=1):
        scores = []
        labels = []
        for tup in self.scores:
            scores.append(self.scores[tup])
            labels.append(1 if self.trials[tup] == 'target' or self.trials[tup] == '1' else 0)
        sorted_indexes, thresholds = zip(*sorted([(index, threshold) for (index, threshold) in enumerate(scores)],
                                                 key=lambda x:x[1]))
        labels = [labels[i] for i in sorted_indexes]
        fnrs = []
        fprs = []
        for i in range(len(labels)):
            if i == 0:
                fnrs.append(labels[i])
                fprs.append(1 - labels[i])
            else:
                fnrs.append(fnrs[i - 1] + labels[i])
                fprs.append(fprs[i - 1] + 1 - labels[i])
        fnrs_norm = sum(labels)
        fprs_norm = len(labels) - fnrs_norm
        fnrs = [x / float(fnrs_norm) for x in fnrs]
        fprs = [1 - x / float(fprs_norm) for x in fprs]

        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(0, len(fnrs)):
            c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf, min_c_det_threshold

    def compute_score(self):
        eer = self.compute_eer()
        minDCF1 = self.compute_min_dcf(p_target=0.01)
        minDCF2 = self.compute_min_dcf(p_target=0.001)
        minDCF = (self.compute_min_dcf(p_target=0.005)[0] + minDCF1[0]) / 2
        return {'EER:': eer[0],
                'minDCF(p-target=0.01):': minDCF1[0],
                'minDCF(p-target=0.001):': minDCF2[0],
                'minDCF(0.01 + 0.005)/2:': minDCF}

if __name__ == '__main__':
    a = Resulter('/raid/sdd/wuyc/kaldi/egs/voxceleb/0007_voxceleb_v2_1a/exp/scores_voxceleb1_test', './trials')
    #a = Resulter('./scores', './trials')
    print(a.compute_score())
