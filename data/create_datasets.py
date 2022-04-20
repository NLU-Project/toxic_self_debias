'''
This script re-creates the train/test/val splits from Toxic_Debias https://github.com/XuhuiZhou/Toxic_Debias/data/founta on the Founta dataset. 
Breakdown of the splits are:
    Train : 62103
    Dev : 10970
    Test : 12893
'''
import pandas as pd
import os, sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def create_comments_finetune(civil_df,toxicity_threshold,toxic_labels, label_type = 'binary'):
        '''
        Function for converting either civil comments or covert civil comments data into format that matches 
        founta and works with toxic debias repo
        
        Inputs:
        -------
            civil_df : pandas DataFrame - full dataframe of civil comments data
            label_type : str - options (binary, multi-class). Type of target label we are using
            toxic_labels : list - list of columns to use in toxic label 
            toxicity_threshold : threshold above which if a text gets above for any of the toxic_labels list we count as "toxic"

        Outputs:
        --------
            pandas DataFrame w/ two columns : one with text and the other associated label with that text
        '''
        civil_df['ND_label'] = civil_df[toxic_labels].max(axis = 1)
        civil_df['ND_label'] = np.where(civil_df['ND_label']>toxicity_threshold, 1, 0)
        civil_df = civil_df[['comment_text', 'ND_label']]
        return civil_df
def agg_sbic(sbic_df):
  print("full length", len(sbic_df))
  YNcols = ['intentYN', 'sexYN', 'offensiveYN']
  sbic_df = sbic_df[['intentYN', 'sexYN', 'offensiveYN', 'post']]
  sbic_df_agg = sbic_df.groupby('post').mean().reset_index().rename({'intentYN':'intentYN_agg',
                                                                     'sexYN': 'sexYN_agg',
                                                                     'offensiveYN': 'offensiveYN_agg'})
  for col in YNcols:
    sbic_df_agg[col] = np.where(sbic_df_agg[col]>=0.5, 1, 0)
  print("de-dupped length", len(sbic_df_agg))
  print(sbic_df_agg.offensiveYN.value_counts())
  return sbic_df_agg

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_list", nargs="+", type=str, required = True,
                        help = "List of datasets to generate with this script")
    parser.add_argument("--data_dir", type = str, required = True,
                        help = "Based directory for reading in and writing out datasets")
    parser.add_argument("--toxicity_threshold", type = float, required = True,
                        help = "Toxicity threshold to use for civil comments data assigning binary toxicity label")
    parser.add_argument("--covert_toxicity_threshold", type = float, required = False,
                        help = "Covert toxicity threshold to use for assigning binary covert toxicity label")
    args = parser.parse_args()

    print("Generating the following datasets:", args.dataset_list)
    if 'founta' in(args.dataset_list):
        toxic_debias_dir = '/scratch/sbp354/DSGA1012/Final_Project/git/Toxic_Debias/data/founta'

        founta_full = pd.read_csv(os.path.join(args.data_dir, 'founta_full.csv'))

        train_ids = pd.read_csv(os.path.join(toxic_debias_dir, 'train', 'original.csv'))
        dev_ids = pd.read_csv(os.path.join(toxic_debias_dir, 'dev.csv'))
        test_ids = pd.read_csv(os.path.join(toxic_debias_dir, 'test.csv'))

        train_idx = train_ids.id.apply(lambda x: str.replace(x, 't', '')).astype(int).tolist()
        dev_idx = dev_ids.id.apply(lambda x: str.replace(x, 't', '')).astype(int).tolist()
        test_idx = test_ids.id.apply(lambda x: str.replace(x, 't', '')).astype(int).tolist()

        #Double check no overlap between different splits
        assert len([i for i in train_idx if (i in dev_idx) | (i in test_idx)])==0
        assert len([i for i in dev_idx if (i in train_idx) | (i in test_idx)])==0
        assert len([i for i in test_idx if (i in train_idx) | (i in dev_idx)])==0

        #print(max(founta_full.index.tolist()))

        founta_train = founta_full.iloc[train_idx]
        founta_dev = founta_full.iloc[dev_idx]
        founta_test = founta_full.iloc[test_idx]
    
        #output full dataset
        founta_train.to_csv(os.path.join(args.data_dir, 'founta_train.csv'), index = False)
        founta_dev.to_csv(os.path.join(args.data_dir, 'founta_dev.csv'), index = False)
        founta_test.to_csv(os.path.join(args.data_dir, 'founta_test.csv'), index = False)

        #output dataset formatted for the Toxic_Debias model repo
        if os.path.exists(os.path.join(args.data_dir, 'founta'))==False:
            os.mkdir(os.path.join(args.data_dir, 'founta'))
        founta_train.to_csv(os.path.join(args.data_dir, 'founta','founta_train_finetune.csv'), header = False, index = False)
        founta_dev.to_csv(os.path.join(args.data_dir, 'founta','founta_dev_finetune.csv'), header = False, index = False)
        founta_test.to_csv(os.path.join(args.data_dir,'founta', 'founta_test_finetune.csv'), header = False, index = False)


        print(f"Writing out {len(founta_train)} founta training instances")
        print(f"Writing out {len(founta_dev)} founta dev instances")
        print(f"Writing out {len(founta_test)} founta test instances")

    if 'civil_comments' in(args.dataset_list):
        
        civil_train = pd.read_csv(os.path.join(args.data_dir, 'civil_train.csv'))
        civil_test = pd.read_csv(os.path.join(args.data_dir, 'civil_test.csv'))
        civil_val = pd.read_csv(os.path.join(args.data_dir, 'civil_val.csv'))
        
        toxic_label_list = ['toxicity', 'severe_toxicity','obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
        civil_train_finetune = create_comments_finetune(civil_train, args.toxicity_threshold, toxic_label_list)
        civil_test_finetune = create_comments_finetune(civil_test, args.toxicity_threshold, toxic_label_list)
        civil_dev_finetune = create_comments_finetune(civil_val, args.toxicity_threshold, toxic_label_list)

        if os.path.exists(os.path.join(args.data_dir, 'civil_comments'))==False:
            os.mkdir(os.path.join(args.data_dir, 'civil_comments'))

        civil_train_finetune.to_csv(os.path.join(args.data_dir, 'civil_comments', f'civil_train_{args.toxicity_threshold}_finetune.csv'), header = False, index = False)
        civil_test_finetune.to_csv(os.path.join(args.data_dir, 'civil_comments', f'civil_test_{args.toxicity_threshold}_finetune.csv'), header = False, index = False)
        civil_dev_finetune.to_csv(os.path.join(args.data_dir, 'civil_comments', f'civil_dev_{args.toxicity_threshold}_finetune.csv'), header = False, index = False)

        print(f"Writing out {len(civil_train_finetune)} civil comments training instances")
        print(f"Writing out {len(civil_dev_finetune)} civil comments dev instances")
        print(f"Writing out {len(civil_test_finetune)} civil comments test instances")
    
    if 'covert_comments' in(args.dataset_list):
        
        covert_train_full = pd.read_csv(os.path.join(args.data_dir, 'covert_train.csv'))
        covert_test = pd.read_csv(os.path.join(args.data_dir, 'covert_test.csv'))

        print(len(covert_test))
        np.random.seed(42)
        test_idx = np.random.choice(range(len(covert_train_full)), len(covert_test))
        print(len(covert_train_full))
        
        covert_train = covert_train_full[~covert_train_full.index.isin(test_idx)]
        covert_val = covert_train_full.iloc[test_idx]
            
        
        covert_label_list = ['implicitly_offensive']
        covert_train_finetune = create_comments_finetune(covert_train, args.covert_toxicity_threshold, covert_label_list)
        covert_val_finetune = create_comments_finetune(covert_val, args.covert_toxicity_threshold, covert_label_list)
        covert_test_finetune = create_comments_finetune(covert_test, args.covert_toxicity_threshold, covert_label_list)

        print("covert train toxicity breakdown:")
        print(covert_train_finetune.ND_label.value_counts())
        print("covert val toxicity breakdown:")
        print(covert_val_finetune.ND_label.value_counts())
        print("covert test toxicity breakdown:")
        print(covert_test_finetune.ND_label.value_counts())

        if os.path.exists(os.path.join(args.data_dir, 'covert_comments'))==False:
            os.mkdir(os.path.join(args.data_dir, 'covert_comments'))

        covert_train_finetune.to_csv(os.path.join(args.data_dir, 'covert_comments', f'covert_train_{args.covert_toxicity_threshold}_finetune.csv'), header = False, index = False)
        covert_train_finetune.to_csv(os.path.join(args.data_dir, 'covert_comments', f'covert_val_{args.covert_toxicity_threshold}_finetune.csv'), header = False, index = False)
        covert_test_finetune.to_csv(os.path.join(args.data_dir, 'covert_comments', f'covert_test_{args.covert_toxicity_threshold}_finetune.csv'), header = False, index = False)

        print(f"Writing out {len(covert_train_finetune)} covert comments training instances")
        print(f"Writing out {len(covert_val_finetune)} covert comments val instances")
        print(f"Writing out {len(covert_test_finetune)} covert comments test instances")

    if 'SBIC' in(args.dataset_list):
        
        sbic_train = pd.read_csv(os.path.join(args.data_dir, 'sbic_train.csv'))
        sbic_test = pd.read_csv(os.path.join(args.data_dir, 'sbic_test.csv'))
        sbic_val = pd.read_csv(os.path.join(args.data_dir, 'sbic_val.csv'))
        
        sbic_train = agg_sbic(sbic_train)
        sbic_train = sbic_train[['post', 'offensiveYN']]
        sbic_test = agg_sbic(sbic_test)
        sbic_test = sbic_test[['post', 'offensiveYN']]
        sbic_val = agg_sbic(sbic_val)
        sbic_val = sbic_val[['post', 'offensiveYN']]
        
        print("SBIC train toxicity breakdown:")
        print(sbic_train.offensiveYN.value_counts())
        print("SBIC test toxicity breakdown:")
        print(sbic_test.offensiveYN.value_counts())
        print("SBIC val toxicity breakdown:")
        print(sbic_val.offensiveYN.value_counts())

        if os.path.exists(os.path.join(args.data_dir, 'SBIC'))==False:
            os.mkdir(os.path.join(args.data_dir, 'SBIC'))

        sbic_train.to_csv(os.path.join(args.data_dir, 'SBIC', f'sbic_train_finetune.csv'), header = False, index = False)
        sbic_test.to_csv(os.path.join(args.data_dir, 'SBIC', f'sbic_test_finetune.csv'), header = False, index = False)
        sbic_val.to_csv(os.path.join(args.data_dir, 'SBIC', f'sbic_val_finetune.csv'), header = False, index = False)

        print(f"Writing out {len(sbic_train)} SBIC training instances")
        print(f"Writing out {len(sbic_test)} SBIC test instances")
        print(f"Writing out {len(sbic_val)} SBIC val instances")
    
    if 'BiBiFi' in(args.dataset_list):

        bibifi_full = pd.read_csv(os.path.join(args.data_dir, 'BiBiFi.csv'))

        bibifi_label_list = ['train', 'test', 'valid']
        
        if os.path.exists(os.path.join(args.data_dir, 'bibifi'))==False:
            os.mkdir(os.path.join(args.data_dir, 'bibifi'))
        for l in  bibifi_label_list:
            subset = bibifi_full[bibifi_full.train_test_valid==l]
            subset = subset[['text', 'label']]
            output_file = os.path.join(args.data_dir, 'bibifi', f'bibifi_{l}_finetune.csv')
            print(f"Writing out to {output_file}")
            subset.to_csv(output_file)
            for a_s in ['adversarial', 'standard']:
                print(bibifi_full.head())
            
                subset = bibifi_full[(bibifi_full.train_test_valid==l) & (bibifi_full.adversarial_standard==a_s)]
                print(f"# examples for {l} + {a_s}: {len(subset)}")

                subset = subset[['text', 'label']]
                output_file = os.path.join(args.data_dir, 'bibifi', f'bibifi_{l}_{a_s}.csv')
                print(f"Writing out to {output_file}")
                subset.to_csv(output_file)



if __name__ == '__main__':
    main()