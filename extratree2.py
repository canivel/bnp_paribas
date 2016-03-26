__author__ = 'canivel'
import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split, cross_val_score


def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax()

def predict(clf, train, target, test, score, id_test):
    clf.fit(train, target)
    print('-----------------------')

    print('Predict...')
    y_pred = clf.predict_proba(test)
    # print y_pred

    df = pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:, 1]})
    df.to_csv('data/submission_stackin_rf_extra_tree_{}.csv'.format(score), index=False)

if __name__ == '__main__':
    print('Load data...')
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    target = train['target'].values
    # train = train[['v3', 'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47',
    #                'v50', 'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110',
    #                'v112', 'v113', 'v114', 'v125', 'v129']]

    id_test = test['ID'].values
    # test = test[['v3', 'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47', 'v50',
    #              'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112',
    #              'v113', 'v114', 'v125', 'v129']]

    high_correlations = [
        'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81', 'v82', 'v89', 'v92',
        'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128'
    ]

    num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
                'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
                'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
                'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
                'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
                'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84',
                'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98',
                'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
                'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

    train = train.drop(['ID', 'target'], axis=1)
    train = train.drop(high_correlations, axis=1)

    test = test.drop(['ID'], axis=1)
    test = test.drop(high_correlations, axis=1)

    print('Clearing...')
    vs = pd.concat([train, test])
    for c in num_vars:
        if c not in train.columns:
            continue

        train.loc[train[c].round(5) == 0, c] = 0
        test.loc[test[c].round(5) == 0, c] = 0

        denominator = find_denominator(vs, c)
        train[c] *= 1/denominator
        test[c] *= 1/denominator

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
                train.loc[train_series.isnull(), train_name] = -999
            #and Test
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = -999

    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

    print('Training...')

    X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.3)

    #0.46176
    # -----------------------
    #   logloss train: 0.46203
    # -----------------------

    # ext = ExtraTreesClassifier(n_estimators=400,
    #                            max_features=30,
    #                            criterion='entropy',
    #                            min_samples_split=2,
    #                            max_depth=30,
    #                            min_samples_leaf=2,
    #                            n_jobs=4,
    #                            verbose=1,
    #                            warm_start=True
    #                            )

    et1 = ExtraTreesClassifier(n_estimators=1000,
                               max_features=50,
                               criterion='entropy',
                               min_samples_split=4,
                               max_depth=35,
                               min_samples_leaf=2,
                               verbose=2,
                               n_jobs=-1)



    rf1 = RandomForestClassifier(bootstrap=True,
                                 criterion='entropy',
                                 min_samples_split=4,
                                 min_samples_leaf=2,
                                 max_features=50,
                                 max_depth=35,
                                 n_estimators=1000,
                                 n_jobs=4,
                                 oob_score=False,
                                 random_state=1301,
                                 verbose=2)

    # param_grid = {
    #     'n_estimators': [10],
    #     'max_features': ['auto', 2, 30],
    #     'min_samples_leaf': [2, 8],
    #     'max_leaf_nodes': [2, 8],
    #     'min_samples_split': [2, 5],
    #     'max_depth': [5, 20, 40],
    #     'criterion': ['entropy', 'gini'],
    # }


    clfs = [('et1', et1), ('rf1', rf1)]
    # set up ensemble of rf_1 and rf_2
    clf = VotingClassifier(estimators=clfs, voting='soft', weights=[1, 1])

    # clf = GridSearchCV(estimator=ext, param_grid=param_grid, cv= 5, scoring='log_loss', verbose=1)

    # ('Raw LogLoss score:', -0.50747886759686722)
    # criterion: 'gini'
    # max_depth: 40
    # max_features: 30
    # max_leaf_nodes: 8
    # min_samples_leaf: 8
    # min_samples_split: 5
    # n_estimators: 10
    # best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    # print('Raw LogLoss score:', score)
    # for param_name in sorted(best_parameters.keys()):
    #     print("%s: %r" % (param_name, best_parameters[param_name]))

    clf.fit(X_train, y_train)
    clf_probs = clf.predict_proba(X_test)
    score = log_loss(y_test, clf_probs)

    print('logloss Score: %.5f' % score)

    if (score < 0.46):
        predict(clf, train, target, test, score, id_test)