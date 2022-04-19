#!/bin/bash
DATA='/scratch/sbp354/DSGA1012/Final_Project/data/founta_test.csv'
DATA_COL='tweet'
DATA_DIR='/scratch/ppg228/dsga-1012_nlu/nlu_project/toxic_self_debias/data/'
DATA_OUT='founta_test_blodgett.csv'

python /scratch/ppg228/dsga-1012_nlu/nlu_project/toxic_self_debias/data/blodgett_inference/blodgett.py \
    --data $DATA \
    --data_col $DATA_COL \
    --data_dir $DATA_DIR \
    --data_out $DATA_OUT