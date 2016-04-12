import pandas as pd
import time
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss, make_scorer
from sklearn import cross_validation
from sklearn import preprocessing
import xgboost as xgb

print('Load data...')

tot = pd.read_csv('ensemble/data/new_train.csv')

train = tot[tot['target']!=-1].copy()
test = tot[tot['target']==-1].copy()
target = train['target'].copy()
id_test = test['ID'].copy().values

train = train.drop(['target', 'ID'], axis=1)
test = test.drop(['target', 'ID'], axis=1)

print('Training...')
etc1 = ExtraTreesClassifier(n_estimators=100,
                            max_features=50,
                            criterion='entropy',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=2017,
                            verbose=2)

etc2 = ExtraTreesClassifier(n_estimators=100,
                            max_features=50,
                            criterion='gini',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=9527,
                            verbose=2)

etc3 = ExtraTreesClassifier(n_estimators=100,
                            max_features=50,
                            criterion='entropy',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=9527,
                            warm_start=True,
                            verbose=2)

rf1 = RandomForestClassifier(n_estimators=100,
                             bootstrap=True,
                                 criterion='entropy',
                                 min_samples_split=4,
                                 min_samples_leaf=2,
                                 max_features=50,
                                 max_depth=35,
                                 n_jobs=4,
                                 oob_score=False,
                                 random_state=1301,
                                 verbose=2)

xgb1 = xgb.XGBClassifier(max_depth=11,
                            n_estimators=100,
                            learning_rate=0.05,
                            subsample=0.96,
                            colsample_bytree=0.40,
                            colsample_bylevel=0.40,
                            objective='binary:logistic',
                            nthread=4,
                            seed=2017)

xgb2 = xgb.XGBClassifier(max_depth=11,
                            n_estimators=100,
                            learning_rate=0.03,
                            subsample=0.96,
                            colsample_bytree=0.45,
                            colsample_bylevel=0.45,
                            objective='binary:logistic',
                            nthread=4,
                            seed=1313)
#score = log_loss(y_test, extc.predict_proba(X_test)[:, 1])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, random_state=1301, test_size=0.3)

clfs = [('etc', etc1), ('rf', rf1), ('xgb', xgb1), ('etc2', etc2)]
# # set up ensemble of rf_1 and rf_2
clf = VotingClassifier(estimators=clfs, voting='soft', weights=[1, 1, 1, 1])
st = time.time()
scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='log_loss', cv=5, verbose=2)
print(scores.mean()*-1)
print("time elaspe", time.time() - st)
exit()

clf.fit(train, target)
print('Predict...')
y_pred = clf.predict_proba(test)

# print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:, 1]}).to_csv('data/extra_trees_1_7.csv', index=False)
