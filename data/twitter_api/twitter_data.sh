#!/bin/bash
USER_DATA='twitter-users-demo.csv'
DATA_DIR='/scratch/ppg228/dsga-1012_nlu/nlu_project/toxic_self_debias/data/twitter_api'
TWEET_OUT='user_tweets.csv'
ERROR_OUT='user_errors.csv'
MAX_RESULTS=100
MAX_USERS=5000

python /scratch/ppg228/dsga-1012_nlu/nlu_project/toxic_self_debias/data/twitter_api/twitter_data.py \
    --user_data $USER_DATA \
    --data_dir $DATA_DIR \
    --tweet_out $TWEET_OUT \
    --error_out $ERROR_OUT \
    --max_results $MAX_RESULTS \
    --max_users $MAX_USERS