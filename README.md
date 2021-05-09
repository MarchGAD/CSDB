# CSDB

A Deep Backend for Speaker Verification Based on Xvector Coupling Space and Metric Learning.

# Requirements

* python3

* torch

* kaldiio

* kaldi

# Others

LDA and PLDA generated from kaldi, zero centering is unnecessary for LDA.
# Results

Models of results we showed in paper didn't trained really well at that time, so the following results may be better.

| Name | Extractor | Network | GPLDA EER(\%) | **CSDB** EER(\%)|
| ------ | ------ | ------ | ------ | ------ |
| FC A | TDNN | kaldi_lda | 3.1177 | **2.4549** |
| CNN  | TDNN | kaldi_cnn | 3.1177 | **2.7678** |
| FC B | ResNet | kaldi_lda | 2.3171 | **1.9459** |
| FC C | EFTDNN | kaldi_lda | 1.0392 | **0.9067** |
| NPLDA | TDNN | NPLDA/LDC | 3.1177 |  2.5875 |

# Triplet demo

This section is for our best performance triplet network. A light demo with needed data has been prepared, please run the demo to check the environment is fully setted.

**Note** that all the following commands are started under './CSDB/.'

#### step 1: use absolute path in '.json' and '.scp'
```
python utils/scp_relative2abs.py scps/demo
python utils/json_relative2abs.py json_configs/demo.json
```
#### step 2: run demo
```
python train.py json_configs/demo.py &
```
#### step 3: see the log
Log is in 'exp/demo/log'
```
tail -f exp/demo/log
```
#### step 4: test
```
cd exp/demo
cat result
```
or we can score by ourselves.
```
cd exp/demo/score
python get_scores.py -e 1
// wait for scoring
cat result
```

#### see validate result

```
cat exp/demo/nnet/validate_eer_mindcf
```


# Siamese

Training for siamese network is slightly different from triplet in data selection. Thus, we provide another script 'LDC_train.py' for siamese training such as NPLDA or LDC/HDC in our CSDB.

To run siamese and better compare with NPLDA, we follow its way on data preparing which generates a tsv file for training. After that, we replace the 'train.py' in step 2 with 'LDC_train.py'.