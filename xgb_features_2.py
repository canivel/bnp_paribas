import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.preprocessing import Binarizer, scale, StandardScaler, OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures, Imputer
import pickle
from sklearn.feature_extraction import DictVectorizer


def oneHotEncoding(data, cols, replace=False):
    vec = DictVectorizer()
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return data

def impute_most_frequent(data):
    clf=Imputer(missing_values='NaN', strategy='most_frequent', axis=0).fit(data)
    data=clf.transform(data)
    return pd.DataFrame(data)

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess

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
    df.to_csv('data/submission_extra_tree_{}.csv'.format(score), index=False)


if __name__ == '__main__':

    print('Load data...')
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    target = train['target'].values

    id_test = test['ID'].values

    # high_correlations = [
    #     'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81', 'v82', 'v89', 'v92',
    #     'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128'
    # ]

    high_correlations = ['v75', 'v107', 'v110']

    # num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
    #             'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
    #             'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
    #             'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
    #             'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
    #             'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84',
    #             'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98',
    #             'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
    #             'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

    train = train.drop(['ID', 'target'], axis=1)
    train = train.drop(high_correlations, axis=1)

    test = test.drop(['ID'], axis=1)
    test = test.drop(high_correlations, axis=1)

    print('Enginneering...')

    print('v22 encode')
    train['v22'].fillna('C0',inplace=True)
    test['v22'].fillna('C0',inplace=True)
    v22_freqs = dict(train['v22'].value_counts() )
    train.loc[:,'v22'] = [('C%d' % v22_freqs[s] ) for s in train['v22'].values]
    test.loc[:,'v22'] = [('C%d' % v22_freqs.get(s,1) ) for s in test['v22'].values]

    shapeTrain = train.shape[0]
    shapeTest = test.shape[0]
    train = train.append(test)

    # lbl = preprocessing.LabelEncoder()
    # lbl.fit(list(train['v22'].values))
    # train['v22_new'] = lbl.transform(list(train['v22'].values))
    # train = train.drop(['v22'], axis=1)

    print('Generating dummies...')
    for f in train.columns:
        if (train[f].dtype == 'object'):
            train_dummies = pd.get_dummies(train[f], dummy_na=True).astype(np.int16)

            columns_train = train_dummies.columns.tolist() # get the columns

            cols_to_use_train = columns_train[:len(columns_train)-1] # drop the last one

            train = pd.concat([train, train_dummies[cols_to_use_train]], axis=1)

            train.drop([f], inplace=True, axis=1)

    print('Reshaping Train and test ...')
    test = train[shapeTrain:shapeTrain+shapeTest]
    train = train[0:shapeTrain]

    print (train.shape, test.shape)


    print('fill the nans ...')
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
        # if train_series.dtype == 'O':
        #     #for objects: factorize
        #     train[train_name], tmp_indexer = pd.factorize(train[train_name])
        #     test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #
        #     #but now we have -1 values (NaN)
        # else:
            #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len > 0:
            # print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -997
            # and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len > 0:
            test.loc[test_series.isnull(), test_name] = -997

    # print('find denominator ...')
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


    ######################################################
    print (train.shape, test.shape)
    # print ('Normalizing')
    #
    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.fit_transform(test)
    print ('Finish Creating Features')

    train = np.array(train)
    test = np.array(test)

    # object array to float
    # train = train.astype(float)
    # test = test.astype(float)

    print ('Train Test Split')
    X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.3)

    print ('running fit')

    clf = xgb.XGBClassifier(max_depth=11,
                            n_estimators=1500,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.40,
                            colsample_bylevel=0.40,
                            min_child_weight = 1,
                            objective='binary:logistic',
                            nthread=4,
                            seed=1313)

    clf.fit(X_train, y_train, early_stopping_rounds=200, eval_metric="logloss", eval_set=[(X_test, y_test)])

    #local: 0.467156 | real:  0.46102
    #CV: 0.465076


    print('CV: {}'.format(clf.best_score))

    if(clf.best_score < 0.46):
        print ('real fiting')
        clf.fit(train, target)
        print ('start predicting')

        y_pred = clf.predict_proba(test, ntree_limit=clf.best_iteration)[:, 1]
        submission = pd.DataFrame({"ID": id_test, "PredictedProb": y_pred})
        submission.to_csv("data/submission_xgb2.csv", index=False)

    print ("Success")
    #########################################################
