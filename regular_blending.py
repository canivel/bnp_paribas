from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB

#Create custom logloss function
def logloss(attempt, actual, epsilon=1.0e-15):
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = train['target'].values
id_test = test['ID'].values

high_correlations = [
        'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89'
        ,'v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'
    ]

train = train.drop(['ID', 'target'], axis=1)
train = train.drop(high_correlations, axis=1)

test = test.drop(['ID'], axis=1)
test = test.drop(high_correlations, axis=1)

lbl = LabelEncoder()
lbl.fit(np.unique(list(train.v22.values) + list(test.v22.values)))
train.v22  = lbl.transform(list(train.v22.values))

features = train.columns.tolist()

def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)   #Assign 0 or 1.
    df.drop(columnName, inplace=True, axis=1)
    return df, features


for col in features:
    if ((train[col].dtype == 'object') and (col != "v22")):  # Col v22 has an extremely high number of single
        print(col)  # inputs, so doing this procedure would be too time consuming
        train, binfeatures = Binarize(col, train)
        test, _ = Binarize(col, test, binfeatures)
        nb = BernoulliNB()
        nb.fit(train[col + '_' + binfeatures].values, target)
        train[col] = \
            nb.predict_proba(train[col + '_' + binfeatures].values)[:, 1]
        test[col] = \
            nb.predict_proba(test[col + '_' + binfeatures].values)[:, 1]
        train.drop(col + '_' + binfeatures, inplace=True, axis=1)
        test.drop(col + '_' + binfeatures, inplace=True, axis=1)


train = train.fillna(-977)
test = test.fillna(-977)

print("Shape", train.shape, test.shape)

np.random.seed(0) # seed to shuffle the train set

#Set folds for cross-validation procedure
n_folds = 6
verbose = True
shuffle = False

X = train[features].values
y = target
X_submission= test[features].values


if shuffle:
    idx = np.random.permutation(y.size)
    X = X[idx]
    y = y[idx]

#StratifiedKFold is a cross validation method that subdivides the dataset in a variety of 'folds' using them repeatedly for training
#and subsequent validation. The optimization of the solution is created by decreasing iteratively the loss function. 

skf = list(StratifiedKFold(y, n_folds))

#My beautiful models. 
clfs = [RandomForestClassifier(n_estimators=850, max_features= 45, min_samples_split= 3,
                            max_depth= 40, min_samples_leaf= 2, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=850,max_features= 45, min_samples_split= 3,
                            max_depth= 40, min_samples_leaf= 2, n_jobs = -1, criterion= 'entropy'),
        
        ExtraTreesClassifier(n_estimators=1000, min_samples_leaf= 1, max_depth = 43,min_samples_split= 3,  n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=1000, min_samples_leaf= 1, max_depth = 43,min_samples_split= 3, n_jobs=-1, criterion='entropy'),
        
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.9, max_features=0.75, 
                                   max_depth=9, n_estimators=600, min_samples_split=0.1, verbose=1)]

print "Creating train and test sets for blending."
    

#Create data set frame responsible for encapsulating the blended predictions
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)   #Fit model to data
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]  #Predict probability in each model scenario
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)                 #Create clended matrix



#Perform logistic regression on the blended dataset:

print "Blending."
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)         #Apply logistic regression to the blended predictions and the targets. This
                                        #model will then be used to create the actual final prediction.
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

print "Linear stretch of predictions to [0,1]"       #Linearize them so that they fall in the 0-1 limit
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
print "Saving Results."

# print y_pred
submission = pd.DataFrame()
submission["ID"] = id_test
submission["PredictedProb"] = y_submission

submission.to_csv('bnp_ensembled.csv', index=False)