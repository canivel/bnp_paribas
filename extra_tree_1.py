# Based on : https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/code
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss, make_scorer
from sklearn import cross_validation
from sklearn import preprocessing

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax()


def az_to_int(az, nanVal=None):
    if az == az:  # catch NaN
        hv = 0
        for i in range(len(az)):
            hv += (ord(az[i].lower()) - ord('a') + 1) * 26 ** (len(az) - 1 - i)
        return hv
    else:
        if nanVal is not None:
            return nanVal
        else:
            return az


print('Load data...')
# train = pd.read_csv("data/train.csv")
# target = train['target'].values
# train = train.drop(
#     ['ID', 'target', 'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79',
#      'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123',
#      'v124', 'v128'], axis=1)
# test = pd.read_csv("data/test.csv")
# id_test = test['ID'].values
# test = test.drop(
#     ['ID', 'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81',
#      'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124',
#      'v128'], axis=1)
#
# print('Clearing...')
# num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
#             'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
#             'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
#             'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
#             'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
#             'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84',
#             'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98',
#             'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
#             'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']
#
# train['v22'] = train['v22'].fillna('ZZZZZ')
# test['v22'] = test['v22'].fillna('ZZZZZ')
# train['v22'] = train['v22'].apply(az_to_int)
# test['v22'] = test['v22'].apply(az_to_int)
#
# train['v50'] = train['v50'] ** 0.125
# test['v50'] = test['v50'] ** 0.125
#
# vs = pd.concat([train, test])
# for c in num_vars:
#     if c not in train.columns:
#         continue
#
#     train.loc[train[c].round(5) == 0, c] = 0
#     test.loc[test[c].round(5) == 0, c] = 0
#
#     denominator = find_denominator(vs, c)
#     train[c] *= 1 / denominator
#     test[c] *= 1 / denominator
#
# for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
#     if train_series.dtype == 'O':
#         # for objects: factorize
#         train[train_name], tmp_indexer = pd.factorize(train[train_name], na_sentinel=-9999)
#         test[test_name] = tmp_indexer.get_indexer(test[test_name])
#         # but now we have -1 values (NaN)
#     else:
#         # for int or float: fill NaN
#         tmp_len = len(train[train_series.isnull()])
#         if tmp_len > 0:
#             # print "mean", train_series.mean()
#             train.loc[train_series.isnull(), train_name] = -9999
#             # and Test
#         tmp_len = len(test[test_series.isnull()])
#         if tmp_len > 0:
#             test.loc[test_series.isnull(), test_name] = -9999
#
#
# # for f in train.columns:
# #         new_c = f+'_new_vs_v55'
# #         new_c3 = f+'_new_vs_v38'
# #         new_c2 = f+'_new_vs_v129'
# #         train[new_c] = train[f]*(train['v38']+train['v55'])
# #         train[new_c3] = train[f]*(train['v38']+train['v129'])
# #         train[new_c2] = train[f]*(train['v129']+train['v55'])
# #
# #         test[new_c] = test[f]*(test['v38']+test['v55'])
# #         test[new_c3] = test[f]*(test['v38']+test['v129'])
# #         test[new_c2] = test[f]*(test['v129']+test['v55'])
#
# scaler = preprocessing.StandardScaler()
# train = scaler.fit_transform(train)
# test = scaler.fit_transform(test)


tot = pd.read_csv('ensemble/data/tot.csv')
train = tot[tot['target']!=-1].copy()
test = tot[tot['target']==-1].copy()
target = train['target'].copy()
id_test = test['ID'].copy().values

train = train.drop(['target', 'ID'], axis=1)
test = test.drop(['target', 'ID'], axis=1)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, random_state=1301, test_size=0.3)

print('Training...')
extc = ExtraTreesClassifier(n_estimators=1000,
                            max_features=50,
                            criterion='entropy',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            verbose=2)
# score = log_loss(y_test, extc.predict_proba(X_test)[:, 1])

# scores = cross_validation.cross_val_score(extc, X_train, y_train, scoring='log_loss', cv=5, verbose=2)
# print(scores.mean())
# exit()

extc.fit(train, target)
print('Predict...')
y_pred = extc.predict_proba(test)

# print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:, 1]}).to_csv('data/extra_trees_045.csv', index=False)
