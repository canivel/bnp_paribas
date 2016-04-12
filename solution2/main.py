'''this file contains:

a 2-level stacking
- xgb in 30 runs
- nn in 30 runs
author: apex

'''
import numpy as np
from level1 import *
from level2 import *
from utils import *

if __name__ == '__main__':

    # set the l1_train/l2_train split ratio
    train_split = 30000
    # train_x, train_y, test_x = load_data('../data/test.csv', '../data/train.csv', train_split)

    print('Load data...')

    tot = pd.read_csv('../ensemble/data/new_train.csv')
    #tot = tot.drop(['n0'], axis=1)
    train = tot[tot['target']!=-1].copy()
    test = tot[tot['target']==-1].copy()
    ids = test['ID'].values

    print('Split data')
    train_x, train_y, test_x = load_data_by_df(train, test, train_split)

    print('Num classes')
    num_classes = len(np.unique(train_y))
    num_tests = test_x.shape[0]

    pred1 = np.zeros(num_tests).astype(np.float32)
    print(pred1.shape)
    print(pred1)
    #pred2 = np.zeros((num_tests, num_classes)).astype(np.float32)

    seeds = [2017, 3249, 5705, 5310, 6271, 1430, 4102, 9071, 6855, 9313, 7516, 8512, 8044, 3383, 777,
             5540, 9313, 9266, 5759, 873, 9216, 6743, 9330, 8479, 2195, 939, 4817, 6938, 6356, 4312]
    # level in 30 runs
    print('level in 30 runs')
    for i in range(30):
        # seed = int('%04i'%random.randint(1,9999))
        # print('Seed {}'.format(seed))
        print('Run {}'.format(i))
        r = xgb_level2(train_x, train_y, test_x, seeds[i])
        e = etc_level2(train_x, train_y, test_x, seeds[i])
        print (r.shape, r)

        pred1 = pred1 + r
        #pred2 += nn_level2(train_x, train_y, test_x)

    # combine by averaging
    pred2 = (pred1/30)
    pred3 = pred2/30
    pred = (pred1 + pred2)/2
    
    submit(pred, ids)