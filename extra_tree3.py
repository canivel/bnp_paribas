# Based on : https://www.kaggle.com/chabir/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45-v5/code
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

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

shapeTrain = train.shape[0]
shapeTest = test.shape[0]
train = train.append(test)
encoded_columns = []
for f in train.columns:

    if train[f].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))
        encoded_columns.append(f)
        # if test[f].dtype == 'object':
        #     lbl = preprocessing.LabelEncoder()
        #     lbl.fit(list(test[f].values))
        #     test[f] = lbl.transform(list(test[f].values))

test = train[shapeTrain:shapeTrain + shapeTest]
train = train[0:shapeTrain]

print('Making dummy features')
for f in encoded_columns:
    print(train[f].max())
    #try without the limit with more processing
    if(train[f].max() == test[f].max() and train[f].max() < 1000):
        train_dummies = pd.get_dummies(train[f]).astype(np.int16)
        test_dummies = pd.get_dummies(test[f]).astype(np.int16)

        if(train_dummies.shape[1] == test_dummies.shape[1]):

            columns_train = train_dummies.columns.tolist()  # get the columns
            columns_test = test_dummies.columns.tolist()  # get the columns

            cols_to_use_train = columns_train[:len(columns_train) - 1]  # drop the last one
            cols_to_use_test = columns_test[:len(columns_test) - 1]  # drop the last one

            train = pd.concat([train, train_dummies[cols_to_use_train]], axis=1)
            test = pd.concat([test, test_dummies[cols_to_use_test]], axis=1)

            train.drop([f], inplace=True, axis=1)
            test.drop([f], inplace=True, axis=1)


print('filling NaN')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype != 'O':

        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

        #for objects: factorize
        # train[train_name], tmp_indexer = pd.factorize(train[train_name])
        # test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)

print (train.shape, test.shape)

X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.3)

print('Training...')
#0.46119
clf = ExtraTreesClassifier(
    n_estimators=1000,
    max_features=50,
    criterion='entropy',
    min_samples_split=4,
    max_depth=35,
    min_samples_leaf=2,
    n_jobs=4,
    verbose=2)

clf.fit(train,target)

print('Predict...')
score = log_loss(y_test, clf.predict_proba(X_test)[:, 1])
# score = log_loss(y_test, y_pred[:,1])
#
print('logloss Score: %.5f' % score)

if(score < 0.454):
    y_pred = clf.predict_proba(test)
    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('data/extra_trees_v3_1.csv',index=False)