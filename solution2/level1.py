'''this file contains:

- split train data to l1-train and l2-train
- train level 1 base learners
    level 1 contains:
    - random forest
    - extra trees
    - xgb
    - xgb
    - knn
    - neural network
    - naive bayes
- produce train data pack for level 2
- produce test data pack for level 2

'''

import numpy as np
from scipy import sparse
import pandas as pd
import theano
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet

def train_data_from_level1(df, train_split):
    '''
    1. train level 1 base learners
    2. then load training data pack for level 2
    '''
    #X = df.values.copy()
    # labels = df['target'].copy()
    # id_test = df['ID'].copy().values
    #
    # X = df.drop(['target', 'ID'], axis=1)
    # test = df.drop(['target', 'ID'], axis=1)
    df = df.iloc[np.random.permutation(len(df))]
    y = df['target'].copy()
    X = df.drop(['target', 'ID'], axis=1).copy()
    Xids = df.drop(['target'], axis=1).copy()
    train_ids = df['ID'].copy()

    # # random shuffle the training data
    # np.random.shuffle(X)
    # X, y = X[:, 1:-1].astype(np.float32), X[:, -1]
    # print (X[:10])
    # print (y[:10])
    # exit()
    
    # label encoding
    # encoder = LabelEncoder()
    # y = encoder.fit_transform(labels).astype(np.int32)

    # tf-idf transforming
    # tfidf_trans = TfidfTransformer()
    # X_tfidf = tfidf_trans.fit_transform(X).toarray().astype(np.float32)

    # 0-1 standardization
    standard_trans = StandardScaler()
    X_standard = standard_trans.fit_transform(X).astype(np.float32)


    # nn on 0-1 standardized data
    num_classes = len(np.unique(y))
    # num_features = X_standard.shape[1]
    # layers5 = [('input', InputLayer),
    #            ('dropoutf', DropoutLayer),
    #            ('dense0', DenseLayer),
    #            ('dropout', DropoutLayer),
    #            ('dense1', DenseLayer),
    #            ('dropout2', DropoutLayer),
    #            ('output', DenseLayer)]
    # clf5 = NeuralNet(layers=layers5,
    #                  input_shape=(None, num_features),
    #                  dropoutf_p=0.15,
    #                  dense0_num_units=1000,
    #                  dropout_p=0.25,
    #                  dense1_num_units=500,
    #                  dropout2_p=0.25,
    #                  output_num_units=10,
    #                  output_nonlinearity=softmax,
    #
    #                  update=adagrad,
    #                  update_learning_rate=theano.shared(np.float32(0.01)),
    #                  max_epochs=50,
    #                  eval_size=0.2,
    #                  verbose=1,
    #                  )
    # clf5.fit(X_standard[:train_split], y[:train_split])
    # pred5 = clf5.predict_proba(X_standard[train_split:]).astype(np.float32)

    
    # random forest on raw
    clf1 = RandomForestClassifier(
        n_estimators=1000,
        bootstrap=True,
        criterion='entropy',
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=50,
        max_depth=35,
        n_jobs=4,
        oob_score=False,
        random_state=1301,
        verbose=2
    )
    clf1.fit(X[:train_split], y[:train_split])
    pred1 = clf1.predict_proba(X[train_split:]).astype(np.float32)

    print(pred1)
    
    # extra trees on tfidf
    clf2 = ExtraTreesClassifier(
        n_estimators=1000,
        max_features=50,
        criterion='entropy',
        min_samples_split=4,
        max_depth=35,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=2017,
        verbose=2
    )
    clf2.fit(X[:train_split], y[:train_split])
    pred2 = clf2.predict_proba(X[train_split:]).astype(np.float32)

    print(pred2)

    clf3 = ExtraTreesClassifier(n_estimators=1000,
                            max_features=50,
                            criterion='gini',
                            min_samples_split=4,
                            max_depth=35,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=9527,
                            verbose=2)

    clf3.fit(X[:train_split], y[:train_split])
    pred3 = clf3.predict_proba(X[train_split:]).astype(np.float32)

    print(pred3)
    
    # xgb on raw
    dtrain4 = xgb.DMatrix(X[:int(train_split*0.8)], y[:int(train_split*0.8)])
    deval4 = xgb.DMatrix(X[int(train_split*0.8):train_split], y[int(train_split*0.8):train_split])
    dtest4 = xgb.DMatrix(X[train_split:])
    watchlist4 = [(dtrain4,'train'), (deval4,'eval')]

    param3 = {
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        # 'eta':0.05,
        # 'n_estimators': 1500,
        'subsample': 1,
        'colsample_bytree': 0.4,
        'colsample_bylevel': 0.4,
        'max_depth': 11,
        'eval_metric': 'logloss',
        'silent': 0,
        'nthread': 4,
        'seed': 2017
    }
    num_rounds4 = 2000
    clf4 = xgb.train(param3, dtrain4, num_rounds4, watchlist4, early_stopping_rounds=15)
    pred4 = clf4.predict(dtest4, ntree_limit=clf4.best_iteration).astype(np.float32)

    print(pred4)
    
    # xgb on tfidf
    # dtrain4 = xgb.DMatrix(X_tfidf[:int(train_split*0.8)], y[:int(train_split*0.8)])
    # deval4 = xgb.DMatrix(X_tfidf[int(train_split*0.8):train_split], y[int(train_split*0.8):train_split])
    # dtest4 = xgb.DMatrix(X_tfidf[train_split:])
    # watchlist4 = [(dtrain4,'train'), (deval4,'eval')]
    # param4 = {'max_depth':10, 'eta':0.0825, 'subsample':0.85, 'colsample_bytree':0.8, 'min_child_weight':5.2475 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':9}
    # num_rounds4 = 2000
    # clf4 = xgb.train(param4, dtrain4, num_rounds4, watchlist4, early_stopping_rounds=15)
    # pred4 = clf4.predict(dtest4, ntree_limit=clf4.best_iteration).astype(np.float32)
    
    # naive bayes on tfidf
    # clf5 = MultinomialNB()
    # clf5.fit(X_tfidf[:train_split], y[:train_split])
    # pred5 = clf5.predict_proba(X_tfidf[train_split:]).astype(np.float32)
    #
    # # knn on tfidf with cosine
    # clf6 = KNeighborsClassifier(n_neighbors=380, metric='cosine', algorithm='brute')
    # clf6.fit(X_tfidf[:train_split], y[:train_split])
    # pred6 = clf6.predict_proba(X[train_split:]).astype(np.float32)

    mainX = Xids[train_split:]
    mainX1 = pd.DataFrame({"ID": train_ids[train_split:], "Pred1": pred1[:, 1]})
    mainX2 = pd.DataFrame({"ID": train_ids[train_split:], "Pred2": pred2[:, 1]})
    mainX3 = pd.DataFrame({"ID": train_ids[train_split:], "Pred3": pred3[:, 1]})
    mainX4 = pd.DataFrame({"ID": train_ids[train_split:], "Pred4": pred4})
    #
    all_dfs = mainX.merge(mainX1,on='ID').merge(mainX2,on='ID').merge(mainX3,on='ID').merge(mainX4,on='ID')
    # print(all_dfs.head())
    #print(X[train_split:].shape, pred1.shape, pred2.shape, pred3.shape, pred4.shape)
    # combine raw with meta
    #feat_pack = sparse.hstack((X[train_split:], pred1, pred2, pred3, pred4))

    all_dfs = all_dfs.drop(['ID'], axis=1).values

    return all_dfs, y[train_split:], clf1, clf2, clf3, clf4

def test_data_from_level1(df, clf1, clf2, clf3, clf4):
    '''
    1. load test data pack from level 1
    '''

    y = df['target'].copy()
    ids = df['ID'].copy()
    Xids = df.drop(['target'], axis=1).copy()
    X = df.drop(['target', 'ID'], axis=1).copy()
    X= X.astype(np.float32)
    # transform to tfidf
    #X_standard = standard_trans.transform(X).astype(np.float32)

    # pred proba with clf1 
    pred1 = clf1.predict_proba(X).astype(np.float32)
    # pred proba with clf2 
    pred2 = clf2.predict_proba(X).astype(np.float32)
    # pred proba with clf2
    pred3 = clf3.predict_proba(X).astype(np.float32)
    # pred proba with clf3
    pred4 = clf4.predict(xgb.DMatrix(X), ntree_limit=clf4.best_iteration).astype(np.float32)
    # pred proba with clf4
    # pred5 = clf5.predict_proba(X_standard).astype(np.float32)

    mainX = Xids
    mainX1 = pd.DataFrame({"ID": ids, "Pred1": pred1[:, 1]})
    mainX2 = pd.DataFrame({"ID": ids, "Pred2": pred2[:, 1]})
    mainX3 = pd.DataFrame({"ID": ids, "Pred3": pred3[:, 1]})
    mainX4 = pd.DataFrame({"ID": ids, "Pred4": pred4})
    #
    # combine raw with meta
    all_dfs = mainX.merge(mainX1,on='ID').merge(mainX2,on='ID').merge(mainX3,on='ID').merge(mainX4,on='ID')
    #feat_pack = sparse.hstack((X, pred1, pred2, pred3, pred4))
    all_dfs = all_dfs.drop(['ID'], axis=1).values
    return all_dfs