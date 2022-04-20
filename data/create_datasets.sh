#!/bin/bash

export DATASET_LIST=(covert_comments)
export DATA_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export CIVIL_TOXICITY_THRESHOLD=0.5
export COVERT_TOXICITY_THRESHOLD=0.5

python ../DSGA1012/Final_Project/git/toxic_self_debias/data/create_datasets.py \
    --dataset_list "${DATASET_LIST[@]}"\
    --data_dir $DATA_DIR\
    --toxicity_threshold $CIVIL_TOXICITY_THRESHOLD\
    --covert_toxicity_threshold $COVERT_TOXICITY_THRESHOLD