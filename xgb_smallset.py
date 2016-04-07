import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


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

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax()


def PrepareData(train, test):
    trainids = train.ID.values
    testids = test.ID.values
    targets = train['target'].values
    #55
    tokeep = ['v3', 'v10', 'v12', 'v14', 'v16', 'v21', 'v22', 'v24',
              'v30', 'v31', 'v34', 'v38', 'v40', 'v47',
              'v50', 'v52', 'v55', 'v56', 'v62', 'v66',
              'v71', 'v72', 'v74', 'v75', 'v79', 'v91',  # 'v107',
              'v112', 'v113', 'v114', 'v125', 'v129']

    num_vars = ['v10', 'v12', 'v14', 'v16', 'v21', 'v34', 'v38','v40', 'v50', 'v55', 'v62', 'v72', 'v129']


    features = train.columns[2:]
    todrop = list(set(features).difference(tokeep))
    train.drop(todrop, inplace=True, axis=1)
    test.drop(todrop, inplace=True, axis=1)
    print(train.columns)

    vs = pd.concat([train, test])
    for c in num_vars:
        if c not in train.columns:
            continue

        train.loc[train[c].round(5) == 0, c] = 0
        test.loc[test[c].round(5) == 0, c] = 0

        denominator = find_denominator(vs, c)
        train[c] *= 1 / denominator
        test[c] *= 1 / denominator

    print('Merging train test to categorize v22 and means')

    shapeTrain = train.shape[0]
    shapeTest = test.shape[0]
    train = train.append(test)

    train['v22'] = train['v22'].apply(az_to_int)

    print('fill the means ...')
    train['v50'] = train['v50'].fillna(train['v50'].mean())
    train['v55'] = train['v55'].fillna(train['v55'].mean())
    train['v16'] = train['v16'].fillna(train['v16'].mean())

    print('Reshaping back Train and test ...')
    test = train[shapeTrain:shapeTrain + shapeTest]
    train = train[0:shapeTrain]

    print(train.shape, test.shape)

    features = train.columns[2:]
    for col in features:
        print(col)
        if((train[col].dtype == 'object')):
            train.loc[~train[col].isin(test[col]), col] = 'Orphans'
            test.loc[~test[col].isin(train[col]), col] = 'Orphans'

            train[col].fillna('Missing', inplace=True)
            test[col].fillna('Missing', inplace=True)

            train[col], tmp_indexer = pd.factorize(train[col])
            test[col] = tmp_indexer.get_indexer(test[col])

            traincounts = train[col].value_counts().reset_index()
            traincounts.rename(columns={'index': col, col: col+'_count'}, inplace=True)
            traincounts = traincounts[traincounts[col+'_count'] >= 50]

            # train = train.merge(traincounts, how='left', on=col)
            # test = test.merge(traincounts, how='left', on=col)
            g = train[[col, 'target']].copy().groupby(col).mean().reset_index()
            g = g[g[col].isin(traincounts[col])]
            g.rename(columns={'target': col+'_avg'}, inplace=True)

            train = train.merge(g, how='left', on=col)
            test = test.merge(g, how='left', on=col)

            h = train[[col, 'target']].copy().groupby(col).std().reset_index()
            h = h[h[col].isin(traincounts[col])]
            h.rename(columns={'target': col+'_std'}, inplace=True)

            train = train.merge(h, how='left', on=col)
            test = test.merge(h, how='left', on=col)

            train.drop(col, inplace=True, axis=1)
            test.drop(col, inplace=True, axis=1)

    features = train.columns[2:]
    train.fillna(-997, inplace=True)
    test.fillna(-997, inplace=True)
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
    ss = StandardScaler()
    train[features] = np.round(ss.fit_transform(train[features].values), 6)
    test[features] = np.round(ss.transform(test[features].values), 6)
    gptrain = pd.DataFrame()
    gptest = pd.DataFrame()
    gptrain.insert(0, 'ID', trainids)
    gptest.insert(0, 'ID', testids)
    gptrain = pd.merge(gptrain, train[list(['ID'])+list(features)], on='ID')
    gptest = pd.merge(gptest, test[list(['ID'])+list(features)], on='ID')
    gptrain['TARGET'] = targets
    del train
    del test
    gc.collect()
    return gptrain, gptest


if __name__ == "__main__":
    print('Started!')
    ss = StandardScaler()
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    gptrain, gptest = PrepareData(train, test)


    target = gptrain['TARGET']
    ids = gptest['ID']
    gptrain = gptrain.drop(['ID', 'TARGET'], axis=1)
    gptest = gptest.drop(['ID'], axis=1)


    # print(gptrain.shape, gptest.shape)
    # print(gptrain.columns)
    # print(gptest.columns)
    # exit()

    # merge numerical and categorical sets
    trainend = int(0.75 * len(gptrain))
    valid_inds = list(gptrain[trainend:].index.values)
    train_inds = list(gptrain.loc[~gptrain.index.isin(valid_inds)].index.values)

    X_valid = gptrain.iloc[valid_inds]
    X_train = gptrain.iloc[train_inds]

    validlabels = target[trainend:]
    trainlabels = target[:trainend]

    xgtrain = xgb.DMatrix(X_train, label=trainlabels)
    xgval = xgb.DMatrix(X_valid, label=validlabels)
    xgtest = xgb.DMatrix(gptest)

    # dfulltrain = xgb.DMatrix(gptrain[features], gptrain.TARGET.values)
    # dfulltest = xgb.DMatrix(gptest[features])

    num_rounds = 500

    seed = 2017
    params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.05,
            #'eta':0.05,
            #'n_estimators': 1500,
            'subsample': 0.96,
            'colsample_bytree': 0.45,
            'colsample_bylevel': 0.45,
            'max_depth': 11,
            'eval_metric': 'logloss',
            'nthread': 4,
            'seed': seed
        }


    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    #clf = xgb.train(params, dfulltrain, num_rounds)
    clf = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

    train_preds = clf.predict(xgval)
    #
    # pred_eval = np.mean(np.array(train_preds), axis = 0)
    # print('CV:', log_loss(target.values, np.clip(pred_eval, 0.01, 0.99)))

    # submission = pd.DataFrame({"ID": gptrain.ID,
    #                            "TARGET": gptrain.TARGET,
    #                            "PREDICTION": train_preds})
    #
    # submission.to_csv("data/ordinalxgbtrain.csv", index=False)

    test_preds = clf.predict(xgtest)
    submission = pd.DataFrame({"ID": ids,
                               "PredictedProb": test_preds})
    submission.to_csv("data/xgb_smallset_1.csv", index=False)
    print('Finished!')