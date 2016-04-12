'''this file contains:

- data loader
- data submit

'''
import numpy as np
import pandas as pd
from level1 import train_data_from_level1, test_data_from_level1

def load_data(train_path, test_path, train_split):
    train_x, train_y, standard_trans, clf1, clf2, clf3, clf4 = train_data_from_level1(train_path, train_split)
    test_x = test_data_from_level1(test_path, standard_trans, clf1, clf2, clf3, clf4)
    return train_x, train_y, test_x

def load_data_by_df(df_train, df_test, train_split):
    train_x, train_y, clf1, clf2, clf3, clf4 = train_data_from_level1(df_train, train_split)
    test_x = test_data_from_level1(df_test, clf1, clf2, clf3, clf4)
    return train_x, train_y, test_x

def submit(preds, ids):
    pd.DataFrame({"ID": ids, "PredictedProb": preds}).to_csv('ensemble_sub_1.csv', index=False)
