import os

import matplotlib.patches as mpatches
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from tqdm.notebook import tqdm
from collections import defaultdict
from itertools import combinations
import pickle
import warnings
import gc

warnings.filterwarnings('ignore')

# Constants ----------------------------------------------------------------------------------
DIR_ROOT = '../../data/predict-student-performance-from-game-play/'
PIECES = 10

path_train = os.path.join(DIR_ROOT, "train.csv")
path_train_labels = os.path.join(DIR_ROOT, "train_labels.csv")
path_test = os.path.join(DIR_ROOT, "test.csv")
path_submission = os.path.join(DIR_ROOT, "sample_submission.csv")


# ---------------------------------------------------------------------------------------------
def get_read_chunk_sizes(str_path, debug=True):
    # Read user id column only
    tmp = pd.read_csv(str_path, usecols=[0])

    # Get session lengths for each session_id
    tmp = tmp.groupby('session_id').session_id.agg('count')
    if debug:
        tmp = tmp.head()

    print(f'Number of unique sessions: {len(tmp)}')

    # COMPUTE READS AND SKIPS
    chunk = int(np.ceil(len(tmp) / PIECES))

    read_sizes = []
    skips = [0]
    for k in range(PIECES):
        kth_chunk_start = k * chunk
        kth_chunk_end = (k + 1) * chunk
        if kth_chunk_end > len(tmp):
            kth_chunk_end = len(tmp)

        # Get length of current chunk of data
        len_curr_chunk = tmp.iloc[kth_chunk_start:kth_chunk_end].sum()

        read_sizes.append(len_curr_chunk)
        skips.append(skips[-1] + len_curr_chunk)

    print(f'To avoid memory error, we will read train in {PIECES} pieces of sizes:')
    print(read_sizes)

    return read_sizes, skips


def get_unique_texts(str_path):
    tmp = pd.read_csv(str_path,
                      usecols=['text', 'fqid', 'text_fqid'])

    # Get unique text values
    texts = tmp['text'].unique()

    del tmp
    _ = gc.collect()

    return texts


def flow_01():
    read_sizes, skips = get_read_chunk_sizes(path_train)

    train = pd.read_csv(path_train,
                        nrows=read_sizes[0])
    print('Train size of first piece: ', train.shape)
    train.head()

    # Read train_labels file
    targets = pd.read_csv(path_train_labels)

    # Split session id into session and question number
    targets['session'] = targets['session_id'].apply(lambda x: int(x.split('_')[0]))
    targets['q'] = targets['session_id'].apply(lambda x: int(x.split('_')[-1][1:]))

    # print target shape
    print(targets.shape)
    targets.head()

    list_texts = get_unique_texts(path_train)
    print(f"len(list_texts): {len(list_texts)}")

    _ = gc.collect()


def main():
    flow_01()


if __name__ == '__main__':
    main()
