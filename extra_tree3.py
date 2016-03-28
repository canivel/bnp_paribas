# Based on : https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/code
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.grid_search import GridSearchCV

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax() 

print('Load data...')
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print('Feature Engineering')
target = train['target'].values
train = train.drop(['ID','target'], axis=1)
id_test = test['ID'].values
test = test.drop(['ID'], axis=1)

high_correlations = [
    'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81',
    'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123',
    'v124', 'v128'
]

train = train.drop(high_correlations, axis=1)
test = test.drop(high_correlations, axis=1)

print('Clearing...')
num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
            'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
            'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
            'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
            'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84', 
            'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98', 
            'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
            'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

print ('Find denominators')

vs = pd.concat([train, test])
for c in num_vars:
    if c not in train.columns:
        continue

    train.loc[train[c].round(5) == 0, c] = 0
    test.loc[test[c].round(5) == 0, c] = 0

    denominator = find_denominator(vs, c)
    train[c] *= 1/denominator
    test[c] *= 1/denominator

print (train.shape, test.shape)

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -997
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -997


print (train.shape, test.shape)

X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.3)




# parameters = {
#               'max_features': ['auto', 50, 60],
#               'min_samples_split': [2, 4, 8],
#               'max_depth': [10, 35, 40],
#               'min_samples_leaf': [2, 4]
#               }

# clf = ExtraTreesClassifier(
#     n_estimators=100,
#     criterion='entropy',
#     n_jobs=-1,
#     verbose=2)

# clf = GridSearchCV(model, parameters, n_jobs=4,
#                    cv=StratifiedKFold(target, n_folds=10, shuffle=True),
#                    verbose=2, refit=True, scoring='log_loss')
# max_depth: 40
# max_features: 50
# min_samples_leaf: 2
# min_samples_split: 2


# clf.fit(train, target)

# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
# print('Log Loss score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))





##############################
print('Training...')
#100 estimators 100 features
#Local-0.05141 : Kaggle-0.46329

#1000 estimators 100 features
#local loss 0.05131 = 0.45566

#1000 estimators 40 features
#local loss .06681 = 0.45417

#1500 estimators 50 features
#local 0.07935 = 0.45566

clf = ExtraTreesClassifier(
    n_estimators=1500,
    max_features=50,
    criterion='entropy',
    min_samples_split=2,
    max_depth=35,
    min_samples_leaf=2,
    n_jobs=-1,
    verbose=2)

clf.fit(train,target)

print('Predict...')
score = log_loss(y_test, clf.predict_proba(X_test)[:, 1])
# score = log_loss(y_test, y_pred[:,1])
#
print('logloss Score: %.5f' % score)

if(score < 0.065):
    y_pred = clf.predict_proba(test)
    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('data/extra_trees_v3_2.csv',index=False)