__author__ = 'canivel'
import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn import ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split, cross_val_score

print('Load data...')
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = train['target'].values
train = train[['v3', 'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47',
               'v50', 'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110',
               'v112', 'v113', 'v114', 'v125', 'v129']]

id_test = test['ID'].values
test = test[['v3', 'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47', 'v50',
             'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112',
             'v113', 'v114', 'v125', 'v129']]

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
    if train_series.dtype == 'O':
        # for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
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

print('Training...')

X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.2)

clf = DecisionTreeClassifier(criterion='entropy',
                             min_samples_split=5,
                             max_depth=40,
                             max_features=30,
                             random_state=1301,
                             )

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)


clf = DecisionTreeClassifier(criterion='entropy',
                             min_samples_split=5,
                             max_depth=40,
                             max_features=30,
                             random_state=2602,
                             )

# Train random forest classifier, calibrate on validation data and evaluate
# on test data
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
sig_clf.fit(train, target)
sig_clf_probs = sig_clf.predict_proba(X_test)
sig_score = log_loss(y_test, sig_clf_probs)


print('\n-----------------------')
print('  logloss train: %.5f' % score)
print('  logloss valid: %.5f' % sig_score)
print('-----------------------')



# param_grid = {
#     'n_estimators': [10],
#     'max_features': ['auto', 2, 30],
#     'min_samples_leaf': [2, 8],
#     'max_leaf_nodes': [2, 8],
#     'min_samples_split': [2, 5],
#     'max_depth': [5, 20, 40],
#     'criterion': ['entropy', 'gini'],
# }


# clfs = [('rf1', rf1), ('rf2', rf2)]
# # set up ensemble of rf_1 and rf_2
# clf = VotingClassifier(estimators=clfs, voting='soft', weights=[1, 1])

# clf = GridSearchCV(estimator=ext, param_grid=param_grid, cv= 5, scoring='log_loss', verbose=1)
clf.fit(train, target)
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

# scores
log_train = log_loss(y_train, clf.predict_proba(X_train)[:, 1])
log_valid = log_loss(y_test, clf.predict_proba(X_test)[:, 1])

print('\n-----------------------')
print('  logloss train: %.5f' % log_train)
print('  logloss valid: %.5f' % log_valid)
print('-----------------------')

print('Predict...')
y_pred = clf.predict_proba(test)
# print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:, 1]}).to_csv('submission_decision_tree.csv', index=False)
