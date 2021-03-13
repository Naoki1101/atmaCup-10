#!/bin/bash

cd ../src

# python train.py -m 'lightgbm_001' -c 'test'
# python train.py -m 'lightgbm_002' -c 'GroupKFold'
# python train.py -m 'lightgbm_003' -c 'optuna'
# python train.py -m 'lightgbm_004' -c 'optuna params'
# python train.py -m 'lightgbm_005' -c 'drop outlier'
# python train.py -m 'lightgbm_006' -c 'custom_002'
# python train.py -m 'lightgbm_007' -c 'custom_003'
# python train.py -m 'lightgbm_008' -c 'custom_004'
# python train.py -m 'lightgbm_009' -c 'custom_005'
# python train.py -m 'lightgbm_010' -c 'n_split=7'
# python train.py -m 'lightgbm_011' -c 'n_split=5'
# python train.py -m 'lightgbm_012' -c 'custom_007'
# python train.py -m 'lightgbm_013' -c 'materialのw2v特徴量を追加'   # material w2vない方がLB高い...？
# python train.py -m 'lightgbm_014' -c 'techniqueのw2v特徴量を追加'
# python train.py -m 'lightgbm_015' -c '残差予測'
# python train.py -m 'lightgbm_016' -c 'BERT特徴量を追加'   # なかなかスコア上がった
# python train.py -m 'lightgbm_017' -c 'titleの言語特徴量を追加'   # CVちょっと上がってLBちょっと下がった
python train.py -m 'lightgbm_018' -c 'tfidf特徴量を追加'
