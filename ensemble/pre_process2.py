import pandas as pd
from utils import *
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import predict_nans

'''*****************************************************************'''
'''****************************load data****************************'''
'''*****************************************************************'''
#load data set
X = pd.read_csv('../data/train.csv')
tst = pd.read_csv('../data/test.csv')
y = X['target'].copy()
tot = X.append(tst)
tot['target'] = tot['target'].fillna(-1)
tot.index = list(range(tot.shape[0]))
#load data info
data_info = pd.read_csv('data/data_info.csv', index_col=0)
groupc = eval(data_info.loc['groupc', 'info'])


tot['nNAN'] = tot.isnull().sum(axis=1)

'''*****************************************************************'''
'''************create a catfeats-based logistic feature*************'''
'''*****************************************************************'''
from sklearn.linear_model import LogisticRegression
#make one hot cols
oh = to_onehot(tot[groupc])
oh.drop('v22', axis=1, inplace=True)
xoh = oh.loc[X.index].copy()

#compute the pearson correlation and choose the top 200 as input
pcoh = compute_pearsonr(xoh, y).abs().sort_values(ascending=False)
pcoh_top200 = pcoh[:200].index
oh = oh[pcoh_top200]
xoh = xoh[pcoh_top200]

#logistic regression
logit = LogisticRegression(C=10, solver='lbfgs', max_iter=300)
logit.fit(xoh, y)
logit_feat = logit.predict_log_proba(oh)[:, 1]
tot['logit_feat'] = logit_feat

'''*****************************************************************'''
'''****************************factorize****************************'''
'''*****************************************************************'''
for col in groupc:
    tot[col] = pd.factorize(tot[col])[0]


'''*****************************************************************'''
'''************************numeric transform************************'''
'''*****************************************************************'''
tot['v50'] = tot['v50'] ** 0.125
#tot['v62'] = np.log(tot['v62'] + 0.1)
tot.to_csv('data/new_train_with_nans.csv', index=False)
exit()
# tot.drop(['v8', 'v25', 'v46', 'v54', 'v63', 'v89'], axis=1, inplace=True) #group1 feats
# tot.drop(['v107', 'v79', 'v75', 'v110'], axis=1, inplace=True) #cat feats

partial_train = tot[tot['target']!=-1].copy()
partial_test = tot[tot['target']==-1].copy()
estimator = KNeighborsClassifier()

not_null_features = []
has_null_features = []
for f in tot.columns:
    if (tot[f].isnull().sum() == 0):
        not_null_features.append(f)
    else:
        has_null_features.append(f)

categories = groupc
print categories

for f in has_null_features:
    print('Predict {}'.format(f))
    feature_train = partial_train[not_null_features].loc[(partial_train[f].notnull())]
    feature_train_test = partial_train[not_null_features].loc[(partial_train[f].isnull())]
    label_train = partial_train['target'].loc[(partial_train[f].notnull())].values

    feature_train = feature_train.drop(['ID', 'target'], axis=1).values
    feature_train_test = feature_train_test.drop(['ID', 'target'], axis=1).values

    nan_features_test = partial_test[not_null_features].loc[(partial_test[f].notnull())]
    y_test = partial_test[not_null_features].loc[(partial_test[f].isnull())]
    nan_labels_test = partial_test['target'].loc[(partial_test[f].notnull())].values

    nan_features_test = nan_features_test.drop(['ID', 'target'], axis=1).values
    y_test = y_test.drop(['ID', 'target'], axis=1).values

    new_feature_train, new_feature_test = predict_nans.predict_nans(feature_train,
                                                                    label_train,
                                                                    feature_train_test,
                                                                    nan_features_test,
                                                                    nan_labels_test,
                                                                    y_test,
                                                                    estimator)

    partial_train.loc[partial_train[f].isnull(), f] = new_feature_train
    partial_test.loc[(partial_test[f].isnull()), f] = new_feature_test

    print('Predict {} Done!'.format(f))


tot = partial_train.append(partial_test)

tot = tot.fillna(-9999)


'''*****************************************************************'''
'''***********************drop some features************************'''
'''*****************************************************************'''


'''*****************************************************************'''
'''*****************************output******************************'''
'''*****************************************************************'''
tot.to_csv('data/new_train_pred_nans2.csv', index=False)

print tot.head()
