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

from sklearn import metrics, linear_model
import random


class addNearestNeighbourLinearFeatures:

    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd=random_state
        self.n=n_neighbours
        self.max_elts=max_elts
        self.verbose=verbose
        self.neighbours=[]
        self.clfs=[]

    def fit(self,train,y):
        if self.rnd!=None:
            random.seed(rnd)
        if self.max_elts==None:
            self.max_elts=len(train.columns)
        list_vars=list(train.columns)
        random.shuffle(list_vars)

        lastscores=np.zeros(self.n)+1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars=list_vars[self.n:]

        for elt in list_vars:
            indice=0
            scores=[]
            for elt2 in self.neighbours:
                if len(elt2)<self.max_elts:
                    clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
                    clf.fit(train[elt2+[elt]], y)
                    scores.append(metrics.log_loss(y,clf.predict(train[elt2 + [elt]])))
                    indice=indice+1
                else:
                    scores.append(lastscores[indice])
                    indice=indice+1
            gains=lastscores-scores
            if gains.max()>0:
                temp=gains.argmax()
                lastscores[temp]=scores[temp]
                self.neighbours[temp].append(elt)

        indice=0
        for elt in self.neighbours:
            clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice=indice+1

    def transform(self, train):
        indice=0
        for elt in self.neighbours:
            train['_'.join(pd.Series(elt).sort_values().values)]=self.clfs[indice].predict(train[elt])
            indice=indice+1
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)

print('Load data...')

tot = pd.read_csv('ensemble/data/new_train2.csv')

train = tot[tot['target']!=-1].copy()
test = tot[tot['target']==-1].copy()
target = train['target'].copy()
id_test = test['ID'].copy().values

train = train.drop(['target', 'ID'], axis=1)
test = test.drop(['target', 'ID'], axis=1)

#
# etc1 = ExtraTreesClassifier(n_estimators=100,
#                             max_features=50,
#                             criterion='entropy',
#                             min_samples_split=4,
#                             max_depth=35,
#                             min_samples_leaf=2,
#                             n_jobs=-1,
#                             random_state=2017,
#                             verbose=2)
#
# etc2 = ExtraTreesClassifier(n_estimators=100,
#                             max_features=50,
#                             criterion='gini',
#                             min_samples_split=4,
#                             max_depth=35,
#                             min_samples_leaf=2,
#                             n_jobs=-1,
#                             random_state=9527,
#                             verbose=2)
#
# etc3 = ExtraTreesClassifier(n_estimators=100,
#                             max_features=50,
#                             criterion='entropy',
#                             min_samples_split=4,
#                             max_depth=35,
#                             min_samples_leaf=2,
#                             n_jobs=-1,
#                             random_state=9527,
#                             warm_start=True,
#                             verbose=2)
#
# rf1 = RandomForestClassifier(n_estimators=100,
#                              bootstrap=True,
#                              criterion='entropy',
#                              min_samples_split=4,
#                              min_samples_leaf=2,
#                              max_features=50,
#                              max_depth=35,
#                              n_jobs=4,
#                              oob_score=False,
#                              random_state=1301,
#                              verbose=2)
#
# xgb1 = xgb.XGBClassifier(max_depth=11,
#                          n_estimators=100,
#                          learning_rate=0.05,
#                          subsample=0.96,
#                          colsample_bytree=0.40,
#                          colsample_bylevel=0.40,
#                          objective='binary:logistic',
#                          nthread=4,
#                          seed=2017)
#
# xgb2 = xgb.XGBClassifier(max_depth=11,
#                          n_estimators=100,
#                          learning_rate=0.03,
#                          subsample=0.96,
#                          colsample_bytree=0.45,
#                          colsample_bylevel=0.45,
#                          objective='binary:logistic',
#                          nthread=4,
#                          seed=1313)
# # score = log_loss(y_test, extc.predict_proba(X_test)[:, 1])
#
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, random_state=1301, test_size=0.3)
#
# clfs = [('etc', etc1), ('rf', rf1), ('xgb', xgb1), ('etc2', etc2)]
# # # set up ensemble of rf_1 and rf_2
# clf = VotingClassifier(estimators=clfs, voting='soft', weights=[1, 1, 1, 1])
# st = time.time()
# scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='log_loss', cv=5, verbose=2)
# print(scores.mean()*-1)
# print("time elaspe", time.time() - st)
# exit()



# seeds = [2017, 3249, 5705, 5310, 6271, 1430, 4102, 9071, 6855, 9313, 7516, 8512, 8044, 3383, 777,
#          5540, 9313, 9266, 5759, 873, 9216, 6743, 9330, 8479, 2195, 939, 4817, 6938, 6356, 4312]
seeds = [2017, 3249, 5705, 5310, 6271, 1430]
n_ft=45 #Number of features to add
max_elts=3 #Maximum size of a group of linear features

print('NNLF...')

rnd = 2017
random.seed(rnd)

a = addNearestNeighbourLinearFeatures(n_neighbours=n_ft, max_elts=max_elts, verbose=True, random_state=2017)

a.fit(train, target)

train = a.transform(train)
test = a.transform(test)

for s in seeds:

    print('Training... {}'.format(s))

    etc1 = ExtraTreesClassifier(n_estimators=1000,
                                max_features=50,
                                criterion='entropy',
                                min_samples_split=4,
                                max_depth=35,
                                min_samples_leaf=2,
                                n_jobs=-1,
                                random_state=s,
                                verbose=1)

    etc2 = ExtraTreesClassifier(n_estimators=1000,
                                max_features=50,
                                criterion='gini',
                                min_samples_split=4,
                                max_depth=35,
                                min_samples_leaf=2,
                                n_jobs=-1,
                                random_state=s,
                                verbose=1)

    rf1 = RandomForestClassifier(n_estimators=1000,
                                 bootstrap=True,
                                     criterion='entropy',
                                     min_samples_split=4,
                                     min_samples_leaf=2,
                                     max_features=50,
                                     max_depth=35,
                                     n_jobs=4,
                                     oob_score=False,
                                     random_state=s,
                                     verbose=1)

    xgb1 = xgb.XGBClassifier(max_depth=11,
                                n_estimators=1000,
                                learning_rate=0.05,
                                subsample=0.96,
                                colsample_bytree=0.40,
                                colsample_bylevel=0.40,
                                objective='binary:logistic',
                                nthread=4,
                                seed=s)
    #score = log_loss(y_test, extc.predict_proba(X_test)[:, 1])

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, random_state=1301, test_size=0.3)

    clfs = [('etc', etc1), ('rf', rf1), ('xgb', xgb1), ('etc2', etc2)]
    # # set up ensemble of rf_1 and rf_2
    clf = VotingClassifier(estimators=clfs, voting='soft', weights=[1, 1, 1, 1])
    # st = time.time()
    # scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='log_loss', cv=5, verbose=2)
    # print(scores.mean()*-1)
    # print("time elaspe", time.time() - st)
    # exit()

    clf.fit(train, target)
    print('Predict...')
    y_pred = clf.predict_proba(test)

    # print y_pred

    pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:, 1]}).to_csv('data/strongs/extra_trees_strongs_{}.csv'.format(s), index=False)
