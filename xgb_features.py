import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.preprocessing import Binarizer, scale, StandardScaler, OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures
import pickle

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
    print (train.shape, test.shape)
    shapeTrain = train.shape[0]
    shapeTest = test.shape[0]
    train = train.append(test)
    print (train.shape, test.shape)
    for f in train.columns:

        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

        # if test[f].dtype == 'object':
        #     lbl = preprocessing.LabelEncoder()
        #     lbl.fit(list(test[f].values))
        #     test[f] = lbl.transform(list(test[f].values))

    test = train[shapeTrain:shapeTrain+shapeTest]
    train = train[0:shapeTrain]

    print (train.shape, test.shape)

    vs = pd.concat([train, test])
    for c in num_vars:
        if c not in train.columns:
            continue

        train.loc[train[c].round(5) == 0, c] = 0
        test.loc[test[c].round(5) == 0, c] = 0

        denominator = find_denominator(vs, c)
        train[c] *= 1/denominator
        test[c] *= 1/denominator

    for f in train.columns:
        if(train[f].max() == test[f].max() and train[f].max() < 1000):

            train_dummies = pd.get_dummies(train[f]).astype(np.int16)
            test_dummies = pd.get_dummies(test[f]).astype(np.int16)
            if(train_dummies.shape[1] == test_dummies.shape[1]):
                columns_train = train_dummies.columns.tolist() # get the columns
                columns_test = test_dummies.columns.tolist() # get the columns

                cols_to_use_train = columns_train[:len(columns_train)-1] # drop the last one
                cols_to_use_test = columns_test[:len(columns_test)-1] # drop the last one

                train = pd.concat([train, train_dummies[cols_to_use_train]], axis=1)
                test = pd.concat([test, test_dummies[cols_to_use_test]], axis=1)

                train.drop([f], inplace=True, axis=1)
                test.drop([f], inplace=True, axis=1)


    # test_enc = []
    # for f in test.columns:
    #     if test[f].dtype == 'object':
    #         test_enc.append(f)
    #         lbl = preprocessing.LabelEncoder()
    #         lbl.fit(list(test[f].values))
    #         test[f] = lbl.transform(list(test[f].values))
    #         if(test[f].max() <= 10):
    #             just_dummies = pd.get_dummies(test[f]).astype(np.int8)
    #             columns = just_dummies.columns.tolist() # get the columns
    #             cols_to_use = columns[:len(columns)-1] # drop the last one
    #             test = pd.concat([test, just_dummies[cols_to_use]], axis=1)
    #             test.drop([f], inplace=True, axis=1)
    #         print test.shape



    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
        if train_series.dtype != 'O':
            #for int or float: fill NaN
            tmp_len = len(train[train_series.isnull()])
            if tmp_len>0:
                #print "mean", train_series.mean()
                train.loc[train_series.isnull(), train_name] = -997
            #and Test
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = -997

    ######################################################
    print (train.shape, test.shape)
    # print ('Normalizing')
    #
    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.fit_transform(test)
    print ('Creating Features')

    train = np.array(train)
    test = np.array(test)

    # object array to float
    train = train.astype(float)
    test = test.astype(float)

    label_log = np.log1p(target)

    # poly = PolynomialFeatures()
    # train = poly.fit_transform(train)
    # print('X_train Shape {}'.format(train.shape))
    # test = poly.fit_transform(test)
    # print('X_test Shape {}'.format(test.shape))

    # print ('Start CV')
    #
    # xgtrain = xgb.DMatrix(train_poly, target)
    param = {'max_depth': 11,
             'n_estimators': 1000,
             'learning_rate': 0.02,
             'subsample': 1,
             'colsample_bytree': 0.90,
             'objective': 'binary:logistic',
             'nthread': 4,
             'seed': 1313
             }

    # print ('running cross validation')
    # # do cross validation, this will print result out as
    # # [iteration]  metric_name:mean_value+std_value
    # # std_value is standard deviation of the metric
    #
    # xgb.cv(param, xgtrain, nfold=5, num_boost_round=5,
    #        metrics={'logloss'}, seed = 1301, show_progress=True)
    #
    # exit()

    print ('Train Test Split')
    X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.3)

    # print ('creating matrices')
    # plst = list(params.items())
    #
    # xgtrain = xgb.DMatrix(train, label=target)
    #
    # print ('running cross validation')
    # num_round = 10
    # xgb.cv(param, xgtrain, num_round, nfold=10, metrics={'logloss'}, seed=1331)

    print ('running fit')

    # xgtest = xgb.DMatrix(test)
    #
    # plst = list(params.items())
    #watchlist = [(xgtrain, 'train'),(xgtest, 'test')]
    # evals = {'eval_metric':'logloss'}
    # print ('Testing loss')
    # num_rounds = 150
    clf = xgb.XGBClassifier(max_depth=11,
                            n_estimators=1000,
                            learning_rate=0.05,
                            subsample=0.96,
                            colsample_bytree=0.40,
                            colsample_bylevel=0.40,
                            objective='binary:logistic',
                            nthread=4,
                            seed=1313)

    clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_test, y_test)])

    #CV: 0.463517

    print('CV: {}'.format(clf.best_score))

    if(clf.best_score < 0.47):
        print ('real fiting')
        clf.fit(train, target)
        print ('start predicting')

        # num_rounds = 1000
        #
        # model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
        # preds1 = model.predict(xgtest)
        # model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
        # preds2 = model.predict(xgtest)
        #
        # preds = np.expm1((preds1+preds2)/2)
        #
        # submission = pd.DataFrame({"ID":id_test, "PredictedProb":preds})
        # submission.to_csv("data/submission_xgb_nopoly_matrix.csv", index=False)

        #
        #
        # clf = xgb.XGBClassifier(objective='binary:logistic',
        #                         max_depth=11,
        #                         n_estimators=1000,
        #                         learning_rate=0.05,
        #                         subsample=0.96,
        #                         colsample_bytree=0.45,
        #                         seed=1301)
        #
        # # fitting
        # clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_test, y_test)])
        #
        #
        # # print y_pred
        y_pred = clf.predict_proba(test, ntree_limit=clf.best_iteration)[:, 1]
        submission = pd.DataFrame({"ID": id_test, "PredictedProb": y_pred})
        submission.to_csv("data/submission_xgb_poly.csv", index=False)

    print ("Success")
    #########################################################
