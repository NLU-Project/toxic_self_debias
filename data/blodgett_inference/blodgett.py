import pandas as pd
import os
import argparse
import pandas as pd
import numpy as np
import sys


#requires python 2.7
## predict module with in the 'twitteraae/code' directory
os.chdir('/scratch/ppg228/dsga-1012_nlu/nlu_project/twitteraae/code') #path to twitteraae git directory code folder
sys.path.insert(0, '/scratch/ppg228/dsga-1012_nlu/nlu_project/twitteraae/code') #path to twitteraae git directory code folder
print(os.getcwd())

import predict
predict.load_model()

def calc_race_proportions(string):
    ustring = unicode(string, "utf-8")
    prop = predict.predict(ustring)
    return(prop)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required = True,
                        help = "data with text to classify")
    parser.add_argument("--data_col", type=str, required = True,
                        help = "column specificing text to classify")                        
    parser.add_argument("--data_dir", type = str, required = True,
                        help = "Based directory for reading in and writing out datasets")
    parser.add_argument("--data_out", type=str, required = True,
                        help = "where to write user's collected tweets")                      
    args = parser.parse_args()



    print('COMPLETE')    
    print('PREDICT MODULE LOADED')
    tweets_df = pd.read_csv(args.data)
    print('DATA LOADED INTO CSV')
    print("shape: ", tweets_df.shape)
    tweets_df[['black','hispanic','asian','white']] = tweets_df.apply(lambda row: calc_race_proportions(row[args.data_col]), axis=1, result_type='expand')
    print('INFERENCE COMPLETE')
    print(tweets_df.head())
    tweets_df.to_csv(os.path.join(args.data_out, args.data_out))
    print('INFERED DATA WRITTEN TO CSV')

if __name__ == '__main__':
    main()