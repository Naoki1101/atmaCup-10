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
python train.py -m 'lightgbm_009' -c 'custom_005'