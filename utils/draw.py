import re
import os
import matplotlib.pyplot as plt


def hist(files, target_dir, split=False):
    multi_scores = []
    multi_labels = []
    for file in files:
        name = os.path.basename(file)
        scores = []
        with open(file, 'r') as f:
            cnt = 0
            for line in f:
                score = line.strip().split()[2]
                scores.append(float(score))
                cnt += 1
            print(cnt)
            if split:
                plt.hist(x=scores,
                         bins=20,
                         edgecolor='black',
                         # normed=True,
                         )
                plt.xlabel('score')
                plt.ylabel('frequency')
                plt.title('hist of {}'.format(name))

                plt.show()
                plt.savefig(os.path.join(target_dir, name + '_hist.png'),
                            dpi=90,
                            )
            else:
                multi_labels.append(name)
                multi_scores.append(scores)

    plt.hist(x=multi_scores,
             bins=20,
             edgecolor='black',
             label=multi_labels,
             # normed=True,
             # stacked=True,
             )
    plt.xlabel('score')
    plt.ylabel('frequency')
    plt.title('hists')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(target_dir, 'hist.png'),
                dpi=90,
                )
    # if not split:


def eerdcf(files, target_dir, ratio=[3, 1], split=False):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    EER, Mindcf1, Mindcf01, namelist = read(files)

    for data, sup in zip([EER, Mindcf1, Mindcf01], ['eer', 'minDCF01', 'minDCF001']):
        for i, ra, tmp in zip(range(len(namelist)), ratio, ['-', '--']):
            data[i] = [data[i][j] for j in range(len(data[i])//ra)]
            plt.plot([x[0] * ra for x in data[i]],
                     [x[1] for x in data[i]],
                     label=namelist[i],
                     linestyle=tmp,
                     linewidth=2)
            plt.xlabel('epoch')
            plt.ylabel(sup)
            plt.legend()

            if split:
                plt.savefig(os.path.join(target_dir, sup),
                            dpi=90,
                            )
                plt.show()
        plt.savefig(os.path.join(target_dir, sup),
                    dpi=90,
                    )
        plt.show()


def read(files):
    EER = []
    Mindcf1 = []
    Mindcf01 = []
    namelist = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        cnt = -1
        for line in lines:
            if cnt >= 0:
                cnt += 1
            if '{' in line:
                if cnt == -1:
                    cnt = 0
                else:
                    break
        print('lines: {}, block size: {}'.format(len(lines), cnt))
        tmp_eer = []
        tmp_mdcf1 = []
        tmp_mdcf01 = []
        from0 = False
        for i in range(len(lines) // cnt):
            tmp = ''
            for j in range(cnt):
                tmp += lines[i * cnt + j].lower()
            epoch = int(re.search(r'epoch.*?([0-9]+)', tmp).group(1))
            if epoch == 0:
                from0 = True
            scorepart = re.search(r'{.*?}', tmp).group(0)
            eer = float(re.search(r'\'eer:\'.*?([0-9]+\.[0-9]+)', scorepart).group(1))
            mdcf1 = float(re.search(r'\'mindcf\(p-target=0.01\):\'.*?([0-9]+\.[0-9]+)', scorepart).group(1))
            mdcf01 = float(re.search(r'\'mindcf\(p-target=0.001\):\'.*?([0-9]+\.[0-9]+)', scorepart).group(1))

            tmp_eer.append((epoch + (1 if from0 else 0), eer))
            tmp_mdcf1.append((epoch + (1 if from0 else 0), mdcf1))
            tmp_mdcf01.append((epoch + (1 if from0 else 0), mdcf01))
        namelist.append(os.path.basename(file))
        EER.append(sorted(tmp_eer))
        Mindcf1.append(sorted(tmp_mdcf1))
        Mindcf01.append(sorted(tmp_mdcf01))
    return EER, Mindcf1, Mindcf01, namelist


def getmin(lis):
    return sorted(lis, key=lambda x:x[1])[0]

if __name__ == '__main__':

    # a = {'12':232, '2':29}
    print(eerdcf([
        '../exp/CSB_3',
        '../exp/NPLDA',
                # '../exp/lda_plda_wccn',

                ],
               '../exp/result_figs/DBEvsNPLDA_3',
                 ratio=[1,3],
                 split=False))
    # hist([
    #     # '../exp/ldawccn_config4/alpha0.5.txt',
    #     # '../exp/ldawccn_config4/validate_scores_100',
    #     # '../exp/ldawccn_config4/alpha10_epoch75.txt',
    #     # '../exp/ldawccn_config4/alpha2_epoch57.txt',
    #     # '../exp/ldawccn_config4/alpha100_epoch46.txt',
    #     # '../exp/ldawccn_config4/kaldi.txt',
    #     '../exp/nplda2.58_epoch24.txt',
    #       ],
    #      '../exp/ldawccn_config4',
    #      split=False
    #      )
    # import os
    # a = os.listdir('../exp/RES')
    # def gett(epo, x, y, z):
    #     return x[0][epo], y[0][epo], z[0][epo]
    # for f in a:
    #     b, c, d, e = read([os.path.join('../exp/RES', f)])
    #     print(f)
    #     print(gett(getmin(b[0])[0] - 1, b, c, d))
    #     print(gett(getmin(c[0])[0] - 1, b, c, d))
    #     print(gett(getmin(d[0])[0] - 1, b, c, d))
    #     print('*************************')