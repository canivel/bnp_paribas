'''this file contains:

- xgb meta learner
- nn meta learner
- predict probabilities for the test data

'''

import numpy as np
import theano
import xgboost as xgb
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet
import random

def xgb_level2(train_x, train_y, test_x):
    '''
    xgb proba predict for level 2
    '''
    # set the train/validate ratio
    l2_row_num = train_x.shape[0]
    l2_row_spl = int(l2_row_num * 0.8)
    
    dtrain = xgb.DMatrix(train_x[:l2_row_spl], train_y[:l2_row_spl])
    deval = xgb.DMatrix(train_x[l2_row_spl:], train_y[l2_row_spl:])
    dtest = xgb.DMatrix(test_x)
    watchlist = [(dtrain,'train'), (deval,'eval')]
    evals_result = {}
    seed = int('%04i'%random.randint(1,9999))
    param = {
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
        'seed': seed
    }
    num_rounds = 10000
    
    bst = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15, evals_result=evals_result)
    pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
    print(pred.shape)
    return pred

def nn_level2(train_x, train_y, test_x):
    '''
    neural net proba predict for level 2
    '''
    num_classes = len(np.unique(train_y))
    num_features = train_x.shape[1]
    layers0 = [('input', InputLayer),
               ('dropoutf', DropoutLayer),
               ('dense0', DenseLayer),
               ('dropout', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout2', DropoutLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropoutf_p=0.15,
                     dense0_num_units=1000,
                     dropout_p=0.25,
                     dense1_num_units=500,
                     dropout2_p=0.25,
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,

                     update=adagrad,
                     update_learning_rate=theano.shared(np.float32(0.01)),
                     # on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.02, stop=0.016)],
                     max_epochs=18,
                     eval_size=0.2,
                     verbose=1,
                     )
    
    net0.fit(train_x, train_y)
    pred = net0.predict_proba(test_x).astype(np.float32)
    
    return pred