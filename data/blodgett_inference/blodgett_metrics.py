import pandas as pd
import os
import argparse
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from scipy import stats

results_df = pd.read_csv('/scratch/sbp354/DSGA1012/Final_Project/models/founta/bert-base-uncased/finetune_founta_challenge_founta_results.csv')
inf_df = pd.read_csv('/scratch/ppg228/dsga-1012_nlu/nlu_project/toxic_self_debias/data/blodgett_inference/founta_test_blodgett.csv').iloc[1: , :].reset_index()

def main():
    print(inf_df.shape)
    print(results_df.shape)

    merged_df = pd.concat([results_df,inf_df], axis = 1)
    merged_df.to_csv('/scratch/sbp354/DSGA1012/Final_Project/models/founta/bert-base-uncased/founta_test_blodgett_results.csv')
    print(merged_df.head(10))
    print(merged_df.tail(10))
    black_corr = stats.pearsonr(merged_df.black, merged_df.predictions)
    white_corr = stats.pearsonr(merged_df.white, merged_df.predictions)

    print({
            'black_correlation':black_corr,
            'white_correlation':white_corr,
                })

    print('script complete')

if __name__ == '__main__':
    main()