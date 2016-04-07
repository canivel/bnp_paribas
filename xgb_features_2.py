import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.preprocessing import Binarizer, scale, StandardScaler, OneHotEncoder
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures, Imputer
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel
import operator
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #Needed to save figures

def az_to_int(az,nanVal=None):
    if az==az:  #catch NaN
        hv = 0
        for i in range(len(az)):
            hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)
        return hv
    else:
        if nanVal is not None:
            return nanVal
        else:
            return az

def impute_most_frequent(data):
    clf = Imputer(missing_values='NaN', strategy='most_frequent', axis=0).fit(data)
    data = clf.transform(data)
    return pd.DataFrame(data)

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax()


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

if __name__ == '__main__':

    print('Load data...')
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    top_prediction = pd.read_csv("data/submission_final_xgb.csv")

    df_target = train['target']
    target = train['target'].values
    id_test = test['ID'].values
    top_preds = top_prediction['PredictedProb']

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

    shapeTrain = train.shape[0]
    shapeTest = test.shape[0]
    shapeTarget = target.shape[0]

    train = train.append(test)
    full_target = df_target.append(top_preds)
    all_values_full_target = full_target.values.astype(np.int8)

    # print(full_target.tail())
    # exit()

    print('Enginneering...')

    print('v22 encode')

    train['v22'] = train['v22'].apply(az_to_int)

    print('fill the means ...')
    train['v50'] = train['v50'].fillna(train['v50'].mean())
    train['v55'] = train['v55'].fillna(train['v55'].mean())
    train['v16'] = train['v16'].fillna(train['v16'].mean())
    #train['v34'] = train['v34'].fillna(train['v34'].mean())
    #train['v21'] = train['v21'].fillna(train['v21'].mean())
    #train['v12'] = train['v12'].fillna(train['v12'].mean())
    #train['v38'] = train['v38'].fillna(train['v38'].mean())
    # train['v78'] = train['v78'].fillna(train['v78'].mean())
    #train['v69'] = train['v69'].fillna(train['v69'].mean())
    #train['v78'] = train['v78'].fillna(train['v78'].mean())
    # train['v129'] = train['v129'].fillna(train['v129'].mean())

    # for i in num_vars:
    #     train[i] = train[i].fillna(train[i].mean())


    # train['v22'].fillna('C0', inplace=True)
    # v22_freqs = dict(train['v22'].value_counts())
    # train.loc[:, 'v22'] = [('C%d' % v22_freqs[s]) for s in train['v22'].values]

    train['v50'] = train['v50'] ** 0.125

    train = train.fillna(-9999)
    encode_columns = []
    print('Encoding...')
    for f in train.columns:
        if (train[f].dtype == 'object'):
            newf = f+'_new_f'
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[newf] = lbl.transform(list(train[f].values))
            encode_columns.append(newf)
            train = train.drop([f], axis=1)


    print('Dummy time...')
    for f in train[encode_columns].columns:
        train_dummies = pd.get_dummies(train[f], prefix=f, prefix_sep='_', dummy_na=True).astype(np.int16)
        columns_train = train_dummies.columns.tolist()  # get the columns
        cols_to_use_train = columns_train[:len(columns_train) - 1]  # drop the last one
        train = pd.concat([train, train_dummies[cols_to_use_train]], axis=1)
        #
        train.drop([f], inplace=True, axis=1)


    #train = train.fillna(-977)
    print('creating dummies for v38 and v55 and v129')
    for f in train.columns:
        new_c = f+'_new_vs_v55'
        new_c3 = f+'_new_vs_v38'
        new_c2 = f+'_new_vs_v129'
        new_c4 = f+'_new_vs_v78'
        new_c5 = f+'_new_vs_v3'
        new_c6 = f+'_new_vs_v74'
        new_c8 = f+'_new_vs_v14'

        train[new_c] = train[f]*(train['v38']+train['v55'])
        train[new_c3] = train[f]*(train['v38']+train['v129'])
        train[new_c2] = train[f]*(train['v129']+train['v55'])
        #val-logloss:0.460323

        train[new_c4] = train[f]*train['v38']+(train[f]*train['v55'])+(train[f]*train['v129'])
        #train[new_c5] = (train[f]/train['v38'])*(train[f]/train['v55'])*(train[f]/train['v129'])
        # train[new_c6] = train[f]*(train['v22']+train['v55'])
        # train[new_c8] = train[f]*(train['v14']+train['v38'])




    #print(train.tail())
    # print('Generating Polynomial...')
    # etc = ExtraTreesClassifier(n_estimators=150,
    #                            max_features=50,
    #                            criterion='entropy',
    #                            min_samples_split=2,
    #                            max_depth=35,
    #                            min_samples_leaf=2,
    #                            n_jobs=-1,
    #                            verbose=2)


    # etc.fit(train, all_values_full_target)
    # model = SelectFromModel(etc, prefit=True)
    # train_new = model.transform(train)
    # print('--------- train_new', train_new.shape)
    # poly = PolynomialFeatures(3)
    # train_poly = poly.fit_transform(train_new)
    # train = pd.DataFrame(train_poly)
    #
    # print('Shape after poly', train.shape)

    train = train.fillna(-9999)

    # p = 50
    #
    # X_bin = Binarizer().fit_transform(scale(train))
    # selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, all_values_full_target)
    # selectF_classif = SelectPercentile(f_classif, percentile=p).fit(train, all_values_full_target)
    #
    # chi2_selected = selectChi2.get_support()
    # chi2_selected_features = [ f for i,f in enumerate(train.columns) if chi2_selected[i]]
    # print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
    #    chi2_selected_features))
    # f_classif_selected = selectF_classif.get_support()
    # f_classif_selected_features = [ f for i,f in enumerate(train.columns) if f_classif_selected[i]]
    # print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
    #    f_classif_selected_features))
    # selected = chi2_selected & f_classif_selected
    # print('Chi2 & F_classif selected {} features'.format(selected.sum()))
    # features = [ f for f,s in zip(train.columns, selected) if s]
    # print (features)
    # train = train[features]

    print('Reshaping Train and test ...')
    test = train[shapeTrain:shapeTrain + shapeTest]
    train = train[0:shapeTrain]

    print (train.shape, test.shape)

    features = train.columns.tolist()
    ceate_feature_map(features)

    # for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
    #     # if train_series.dtype == 'O':
    #     #     #for objects: factorize
    #     #     train[train_name], tmp_indexer = pd.factorize(train[train_name])
    #     #     test[test_name] = tmp_indexer.get_indexer(test[test_name])
    #     #
    #     #     #but now we have -1 values (NaN)
    #     # else:
    #     # for int or float: fill NaN
    #     tmp_len = len(train[train_series.isnull()])
    #     if tmp_len > 0:
    #         # print "mean", train_series.mean()
    #         train.loc[train_series.isnull(), train_name] = -997
    #         # and Test
    #     tmp_len = len(test[test_series.isnull()])
    #     if tmp_len > 0:
    #         test.loc[test_series.isnull(), test_name] = -997

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
    # print ('Normalizing')
    #
    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.fit_transform(test)

    # merge numerical and categorical sets
    trainend = int(0.75 * len(train))
    valid_inds = list(train[trainend:].index.values)
    train_inds = list(train.loc[~train.index.isin(valid_inds)].index.values)

    X_valid = train.iloc[valid_inds]
    X_train = train.iloc[train_inds]

    validlabels = target[trainend:]
    trainlabels = target[:trainend]

    xgtrain = xgb.DMatrix(X_train, label=trainlabels)
    xgval = xgb.DMatrix(X_valid, label=validlabels)
    xgtest = xgb.DMatrix(test)

    ROUNDS = 600
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]

    yevalpred_list = []
    ypred_list = []

    seeds_list = [2017]
    #2017 best seed
    print('Start Bagging')
    for seed in seeds_list:
        params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.05,
            #'eta':0.05,
            #'n_estimators': 1500,
            'subsample': 1,
            'colsample_bytree': 0.4,
            'colsample_bylevel': 0.4,
            'max_depth': 11,
            'eval_metric': 'logloss',
            'silent': 0,
            'nthread': 4,
            'seed': seed
        }

        print('Training', seed)
        model = xgb.train(params, xgtrain, 3000, watchlist, early_stopping_rounds=100)
        y_cv_pred = model.predict(xgval)
        y_pred = model.predict(xgtest)

        yevalpred_list.append(y_cv_pred)
        ypred_list.append(y_pred)

    # print('Cross Validation')
    # cv = xgb.cv(params, xgtrain, ROUNDS, nfold=5, metrics={'logloss'}, show_progress=True, as_pandas=True, seed=4242)
    #
    # print ('Best Round')
    # print ('__________________________')
    # print (cv['test-logloss-mean'].idxmin(), cv['test-logloss-mean'].min())
    #
    # print ('__________________________')
    # print('Training')
    #
    #
    # num_boost_round=cv['test-logloss-mean'].idxmin()
    # print('Training')
    # model = xgb.train(params, xgtrain, 300, watchlist, early_stopping_rounds=50)

    # print('Predict')
    # y_cv_pred = model.predict(xgval)
    pred_eval = np.mean(np.array(yevalpred_list), axis = 0)
    print('CV:', log_loss(validlabels, np.clip(pred_eval, 0.01, 0.99)))
    #('CV:', 0.45882495188313527) >>>> 0.46259 ....
    # y_pred = model.predict(xgtest)
    pred_final = np.mean(np.array(ypred_list), axis = 0)

    # pd.DataFrame({"ID": id_test, "PredictedProb": np.clip(y_pred, 0.01, 0.99)}).to_csv('submission_xgb_compact_1.csv',
    #                                                                                    index=False)

    submission = pd.DataFrame({"ID":id_test, "PredictedProb":pred_final})
    submission.to_csv("data/submission_final_xgb.csv", index=False)


    # importance = model.get_fscore(fmap='xgb.fmap')
    # importance = sorted(importance.items(), key=operator.itemgetter(1))
    #
    # df = pd.DataFrame(importance[:30], columns=['feature', 'fscore'])
    # df['fscore'] = df['fscore'] / df['fscore'].sum()
    #
    # plt.figure()
    # df.plot()
    # df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 15))
    # plt.title('XGBoost Feature Importance')
    # plt.xlabel('relative importance')
    # plt.gcf().savefig('feature_importance_xgb.png')

    #     y_pred = clf.predict_proba(test, ntree_limit=clf.best_iteration)[:, 1]
    #     submission = pd.DataFrame({"ID": id_test, "PredictedProb": y_pred})
    #     submission.to_csv("data/submission_xgb2.csv", index=False)

    # object array to float
    # train = train.astype(float)
    # test = test.astype(float)

    # print ('Train Test Split')
    # X_train, X_test, y_train, y_test = train_test_split(train, target, random_state=1301, stratify=target, test_size=0.3)
    #
    # print ('running fit')
    #
    # clf = xgb.XGBClassifier(max_depth=11,
    #                         n_estimators=1500,
    #                         learning_rate=0.05,
    #                         subsample=0.9,
    #                         colsample_bytree=0.40,
    #                         colsample_bylevel=0.40,
    #                         min_child_weight = 1,
    #                         objective='binary:logistic',
    #                         nthread=4,
    #                         seed=1313)
    #
    # clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_test, y_test)])
    #
    # #local: 0.467156 | real:  0.46102
    # #CV: 0.463081
    #
    #
    # print('CV: {}'.format(clf.best_score))
    #
    # if(clf.best_score < 0.46):
    #     print ('real fiting')
    #     clf.fit(train, target)
    #     print ('start predicting')
    #
    #     y_pred = clf.predict_proba(test, ntree_limit=clf.best_iteration)[:, 1]
    #     submission = pd.DataFrame({"ID": id_test, "PredictedProb": y_pred})
    #     submission.to_csv("data/submission_xgb2.csv", index=False)
    #
    # print ("Success")
    #########################################################
