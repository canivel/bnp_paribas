import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

np.random.seed(1301)
n_folds = 5
verbose = True
shuffle = False


tot = pd.read_csv('ensemble/data/new_train.csv')

train = tot[tot['target']!=-1].copy()
test = tot[tot['target']==-1].copy()
target = train['target'].copy()
test_id = test.ID
train_id = train.ID.values

print(train.shape, test.shape)

train = train.drop(['target', 'ID', 'n0'], axis=1)
test = test.drop(['target', 'ID', 'n0'], axis=1)

# Remove Columns with zero Variance
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# Selecting KBest and Transform
selectK = SelectKBest(f_classif, k='all')
selectK.fit(train, target)
trainSelect = selectK.transform(train)
testSelect = selectK.transform(test)

BestKfeatures = train.columns[selectK.get_support()]
print (BestKfeatures)

# convert them to NPArray
train, target, test = np.array(trainSelect), np.array(target.astype(int)).ravel(), np.array(testSelect)

if shuffle:
    idx = np.random.permutation(target.size)
    train = train[idx]
    target = target[idx]

# Making a Classifier Blend
etc1 = ExtraTreesClassifier(n_estimators=1000,
                            max_features=50,
                            criterion='entropy',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=2017,
                            class_weight='balanced',
                            verbose=2)

etc2 = ExtraTreesClassifier(n_estimators=1000,
                            max_features=50,
                            criterion='gini',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=9527,
                            class_weight='balanced',
                            verbose=2)

rf1 = RandomForestClassifier(n_estimators=1000,
                             bootstrap=True,
                             criterion='entropy',
                             min_samples_split=4,
                             min_samples_leaf=2,
                             max_features=50,
                             max_depth=35,
                             n_jobs=4,
                             oob_score=False,
                             random_state=1301,
                             class_weight='balanced',
                             verbose=2)

xgb1 = xgb.XGBClassifier(max_depth=11,
                         n_estimators=1000,
                         learning_rate=0.05,
                         subsample=0.96,
                         colsample_bytree=0.40,
                         colsample_bylevel=0.40,
                         objective='binary:logistic',
                         nthread=4,
                         seed=2017)

clfs = [etc1, etc2, rf1, xgb1]

# Preparing Dataset for Blending
print ("Creating train and test sets for blending.")
dataset_blend_train = np.zeros((train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((test.shape[0], len(clfs)))
stratified_K_Fold = cross_validation.StratifiedKFold(target, n_folds, shuffle=True)

for index1, classifier in enumerate(clfs):
    print (index1, classifier)
    dataset_blend_test_index1 = np.zeros((test.shape[0], n_folds))
    for index2, (trainFold, testFold) in enumerate(stratified_K_Fold):
        print ("Fold", index2)
        X_train, y_train = train[trainFold], target[trainFold]
        X_test, y_test = train[testFold], target[testFold]
        if index1 < len(clfs)-1:
            classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="logloss",eval_set=[(X_test, y_test)])
            foldProbability = classifier.predict_proba(X_test)[:,1]
            dataset_blend_train[testFold, index1] = foldProbability
            dataset_blend_test_index1[:, index2] = classifier.predict_proba(test)[:,1]
            dataset_blend_test[:,index1] = dataset_blend_test_index1.mean(axis=1)

# Started Blending
print ("Blending.")

classifier = LogisticRegression()
classifier.fit(dataset_blend_train, target)
testProbabilty = classifier.predict_proba(dataset_blend_test)[:,1]

print ("Linear stretch of predictions to [0,1]")
testProbabilty = (testProbabilty - testProbabilty.min()) / (testProbabilty.max() - testProbabilty.min())

print ("Saving Results.")
submission = pd.DataFrame({"ID":test_id, "TARGET":testProbabilty}).to_csv("data/stacking_model1.csv", index=False)