'''this file contains:

- data loader
- data submit

'''
import numpy as np
import pandas as pd
from level1 import train_data_from_level1, test_data_from_level1

def load_data(train_path, test_path, train_split):
    train_x, train_y, standard_trans, clf1, clf2, clf3, clf4, clf5 = train_data_from_level1(train_path, train_split)
    test_x = test_data_from_level1(test_path, standard_trans, clf1, clf2, clf3, clf4, clf5)
    return train_x, train_y, test_x

def load_data_by_df(df_train, df_test, train_split):
    train_x, train_y, standard_trans, clf1, clf2, clf3, clf4, clf5 = train_data_from_level1(df_train, train_split)
    test_x = test_data_from_level1(df_test, standard_trans, clf1, clf2, clf3, clf4, clf5)
    return train_x, train_y, test_x

def submit(submit_arr):
    sample_submit_df = pd.read_csv(r'../data/sample_submission.csv')
    submit_arr = np.insert(submit_arr, 0, range(1, sample_submit_df.shape[0]+1), axis=1)
    submit_df = pd.DataFrame(submit_arr, columns=sample_submit_df.columns)
    submit_df['id'] = submit_df['id'].apply(lambda x: int(x))
    submit_df.to_csv('submission_real_ensemble.csv', index=False)