# Based on : https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/code
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import log_loss, make_scorer
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel

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
train = pd.read_csv("data/train.csv")
target = train['target'].values
train = train.drop(['ID', 'target'], axis=1)
test = pd.read_csv("data/test.csv")
id_test = test['ID'].values
df_id_test = test['ID']
test = test.drop(['ID'], axis=1)

tokeep = ['v3', 'v10', 'v12', 'v14', 'v16', 'v21', 'v22', 'v24',
              'v30', 'v31', 'v34', 'v38', 'v40', 'v47',
              'v50', 'v52', 'v55', 'v56', 'v62', 'v66',
              'v71', 'v72', 'v74', 'v75', 'v79', 'v91',  # 'v107',
              'v112', 'v113', 'v114', 'v125', 'v129']

features = train.columns
todrop = list(set(features).difference(tokeep))
train.drop(todrop, inplace=True, axis=1)
test.drop(todrop, inplace=True, axis=1)

print('Clearing...')
num_vars = ['v10', 'v12', 'v14', 'v16', 'v21', 'v34', 'v38','v40', 'v50', 'v55', 'v62', 'v72', 'v129']
# num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
#             'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
#             'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
#             'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
#             'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
#             'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84',
#             'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98',
#             'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
#             'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

train['v22'] = train['v22'].fillna('ZZZZZ')
test['v22'] = test['v22'].fillna('ZZZZZ')
train['v22'] = train['v22'].apply(az_to_int)
test['v22'] = test['v22'].apply(az_to_int)

vs = pd.concat([train, test])
for c in num_vars:
    if c not in train.columns:
        continue

    train.loc[train[c].round(5) == 0, c] = 0
    test.loc[test[c].round(5) == 0, c] = 0

    denominator = find_denominator(vs, c)
    train[c] *= 1 / denominator
    test[c] *= 1 / denominator

encode_columns = []
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
    if train_series.dtype == 'O':
        # for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])

        encode_columns.append(train_name)
        # but now we have -1 values (NaN)
    else:
        # for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len > 0:
            # print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999
            # and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len > 0:
            test.loc[test_series.isnull(), test_name] = -999

# print('Dummies')
# for f in train.columns:
#     new_c = f + '_new_vs_v55'
#     new_c3 = f + '_new_vs_v38'
#     new_c2 = f + '_new_vs_v129'
#     train[new_c] = train[f] * (train['v38'] + train['v55'])
#     train[new_c3] = train[f] * (train['v38'] + train['v129'])
#     train[new_c2] = train[f] * (train['v129'] + train['v55'])
#
#     test[new_c] = test[f] * (test['v38'] + test['v55'])
#     test[new_c3] = test[f] * (test['v38'] + test['v129'])
#     test[new_c2] = test[f] * (test['v129'] + test['v55'])


# print('Features Selection')
# from sklearn.feature_selection import SelectPercentile, f_classif, chi2
# from sklearn.preprocessing import Binarizer, scale
# p = 80
#
# X_bin = Binarizer().fit_transform(scale(train))
# selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, target)
# selectF_classif = SelectPercentile(f_classif, percentile=p).fit(train, target)
#
# chi2_selected = selectChi2.get_support()
# chi2_selected_features = [f for i, f in enumerate(train.columns) if chi2_selected[i]]
# print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
#                                              chi2_selected_features))
# f_classif_selected = selectF_classif.get_support()
# f_classif_selected_features = [f for i, f in enumerate(train.columns) if f_classif_selected[i]]
# print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
#                                                   f_classif_selected_features))
# selected = chi2_selected & f_classif_selected
# print('Chi2 & F_classif selected {} features'.format(selected.sum()))
# features = [f for f, s in zip(train.columns, selected) if s]
# print (features)
# train = train[features]
# test = test[features]
# print ('Shape ', train.shape, test.shape)

scaler = preprocessing.StandardScaler()
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

g={'ne':100,'md':40,'mf':40,'rs':2017}
print('Training...')
etc = ExtraTreesClassifier(n_estimators=g['ne'],
                            #max_features=50,
                            criterion='entropy',
                            min_samples_split=4,
                            max_depth=g['md'],
                            random_state=g['rs'],
                            min_samples_leaf=2,
                            warm_start=True,
                            n_jobs=-1)

etr = ExtraTreesRegressor(n_estimators=g['ne'],
                          max_depth=g['md'],
                          #max_features=40,
                          random_state=g['rs'],
                          min_samples_split= 4,
                          min_samples_leaf= 2,
                          verbose = 0,
                          n_jobs =-1)

# clfs = [('etc', etc), ('etr', etr)]
# clf = VotingClassifier(clfs, voting='soft', weights=[1, 1])
# score = log_loss(y_test, extc.predict_proba(X_test)[:, 1])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, random_state=2017, test_size=0.3)
# scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='log_loss', cv=5, verbose=2)
# print(scores.mean())

clf = {'etc':etc, 'etr':etr}

def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_

LL = make_scorer(flog_loss, greater_is_better=False)


y_pred=[]
best_score = 0.0
id_results = df_id_test[:]
for c in clf:
    model = GridSearchCV(estimator=clf[c], param_grid={}, n_jobs=-1, cv=2, verbose=1, scoring=LL)
    model.fit(train, target)
    if c[-1:] != "c":  # not classifier
        y_pred = model.predict(test)
        print("Ensemble Model: ", c, " Best CV score: ", model.best_score_)
    else:  # classifier
        # best_score = (log_loss(target, model.predict_proba(train))) * -1
        # scores = cross_validation.cross_val_score(etc, X_train, y_train, scoring='log_loss', cv=5, verbose=1)
        # best_score = scores.mean()
        y_pred = model.predict_proba(test)[:, 1]
        #print("Ensemble Model: ", c, " Best CV score: ", best_score)

    for i in range(len(y_pred)):
        if y_pred[i]<0.0:
            y_pred[i] = 0.0
        if y_pred[i]>1.0:
            y_pred[i] = 1.0

    df_in = pd.DataFrame({"ID": df_id_test, c: y_pred})
    id_results = pd.concat([id_results, df_in[c]], axis=1)


id_results['avg'] = id_results.drop('ID', axis=1).apply(np.average, axis=1)
id_results['min'] = id_results.drop('ID', axis=1).apply(min, axis=1)
id_results['max'] = id_results.drop('ID', axis=1).apply(max, axis=1)
id_results['diff'] = id_results['max'] - id_results['min']

for i in range(10):
    print(i, len(id_results[id_results['diff']>(i/10)]))

id_results.to_csv("results_analysis.csv", index=False)
ds = id_results[['ID','avg']]
ds.columns = ['ID','PredictedProb']
ds.to_csv('data/extratree3_3.csv',index=False)

# extc.fit(train, target)
# print('Predict...')
# y_pred = extc.predict_proba(test)
#
# # print y_pred
#
# pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:, 1]}).to_csv('data/extra_trees_045.csv', index=False)
