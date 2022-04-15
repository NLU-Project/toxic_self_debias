'''
This script re-creates the train/test/val splits from Toxic_Debias https://github.com/XuhuiZhou/Toxic_Debias/data/founta on the Founta dataset. 
Breakdown of the splits are:
    Train : 62103
    Dev : 10970
    Test : 12893
'''
import pandas as pd
import os, sys

toxic_debias_dir = '/scratch/sbp354/DSGA1012/Final_Project/git/Toxic_Debias/data/founta'
data_dir = '/scratch/sbp354/DSGA1012/Final_Project/data'

founta_full = pd.read_csv(os.path.join(data_dir, 'founta.csv'))

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

print(max(founta_full.index.tolist()))
print(f"max train_idx: {max(train_idx)}")
print(f"max val_idx: {max(dev_idx)}")
print(f"max test_idx: {max(test_idx)}")

founta_train = founta_full.iloc[[i for i in train_idx if i < max(founta_full.index)]]
founta_dev = founta_full.iloc[[i for i in dev_idx if i < max(founta_full.index)]]
founta_test = founta_full.iloc[[i for i in test_idx if i < max(founta_full.index)]]


founta_train.to_csv(os.path.join(data_dir, 'founta_train.csv'), index = False)
founta_dev.to_csv(os.path.join(data_dir, 'founta_dev.csv'), index = False)
founta_test.to_csv(os.path.join(data_dir, 'founta_test.csv'), index = False)

