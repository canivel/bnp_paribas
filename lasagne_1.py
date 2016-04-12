# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:01:21 2016

@author: Ouranos
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.init import Uniform
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()       
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


        
def getDummiesInplace(columnList, train, test = None):
    #Takes in a list of column names and one or two pandas dataframes
    #One-hot encodes all indicated columns inplace
    columns = []
    
    if test is not None:
        df = pd.concat([train,test], axis= 0)
    else:
        df = train
        
    for columnName in df.columns:
        index = df.columns.get_loc(columnName)
        if columnName in columnList:
            dummies = pd.get_dummies(df.ix[:,index], prefix = columnName, prefix_sep = ".")
            columns.append(dummies)
        else:
            columns.append(df.ix[:,index])
    df = pd.concat(columns, axis = 1)
    
    if test is not None:
        train = df[:train.shape[0]]
        test = df[train.shape[0]:]
        return train, test
    else:
        train = df
        return train
        
def pdFillNAN(df, strategy = "mean"):
    #Fills empty values with either the mean value of each feature, or an indicated number
    if strategy == "mean":
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)


def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax()


def az_to_int(az, nanVal=None):
    if az == az:  # catch NaN
        hv = 0
        for i in range(len(az)):
            hv += (ord(az[i].lower()) - ord('a') + 1) * 26 ** (len(az) - 1 - i)
        return hv
    else:
        if nanVal is not None:
            return nanVal
        else:
            return az


# tot = pd.read_csv('ensemble/data/new_train.csv')
#
# train = tot[tot['target']!=-1].copy()
# test = tot[tot['target']==-1].copy()
# labels = train['target'].copy()
# trainId = train["ID"]
# testId = test['ID'].copy().values


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

np.random.seed(3210)
train = train.iloc[np.random.permutation(len(train))]

#Drop target, ID, and v22(due to too many levels), and high correlated columns
labels = train["target"]
trainId = train["ID"]
testId = test["ID"]

#train.drop(labels = ["ID","target","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
train = train.drop(
    ['ID', 'target', 'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79',
     'v81', 'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123',
     'v124', 'v128'], axis=1)
#test.drop(labels = ["ID","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
test = test.drop(
    ['ID', 'v8', 'v23', 'v25', 'v31', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81',
     'v82', 'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124',
     'v128'], axis=1)

num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
            'v21', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
            'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50',
            'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
            'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84',
            'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98',
            'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
            'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']


np.random.seed(3210)
train = train.iloc[np.random.permutation(len(train))]

# train = train.drop(['target', 'ID'], axis=1)
# test = test.drop(['target', 'ID'], axis=1)
train['nNAN'] = train.isnull().sum(axis=1)
test['nNAN'] = test.isnull().sum(axis=1)

vs = pd.concat([train, test])
for c in num_vars:
    if c not in train.columns:
        continue

    train.loc[train[c].round(5) == 0, c] = 0
    test.loc[test[c].round(5) == 0, c] = 0

    denominator = find_denominator(vs, c)
    train[c] *= 1 / denominator
    test[c] *= 1 / denominator


shapeTrain = train.shape[0]
shapeTest = test.shape[0]
shapeTarget = labels.shape[0]

train = train.append(test)

train['v22'] = train['v22'].apply(az_to_int)

print('fill the means ...')
train['v50'] = train['v50'].fillna(train['v50'].mean())
train['v55'] = train['v55'].fillna(train['v55'].mean())
train['v16'] = train['v16'].fillna(train['v16'].mean())

train['v50'] = train['v50'] ** 0.125

train = train.fillna(-9999)
encode_columns = []
print('Encoding...')
for f in train.columns:
    if (train[f].dtype == 'object'):
        newf = f + '_new_f'
        lbl = LabelEncoder()
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

train = train.fillna(-9999)
print('Reshaping Train and test ...')
test = train[shapeTrain:shapeTrain + shapeTest]
train = train[0:shapeTrain]

print (train.shape, test.shape)

# print ("Generating dummies...")
# train, test = getDummiesInplace(categoricalVariables, train, test)
#
# #Remove sparse columns
# cls = train.sum(axis=0)
# train = train.drop(train.columns[cls<10], axis=1)
# test = test.drop(test.columns[cls<10], axis=1)
#
# print ("Filling in missing values...")
# # fillNANStrategy = -1
# # #fillNANStrategy = "mean"
# # train = pdFillNAN(train, fillNANStrategy)
# # test = pdFillNAN(test, fillNANStrategy)
#
# #tot['v62'] = np.log(tot['v62'] + 0.1)
# train = train.fillna(-9999)
# test = test.fillna(-9999)
#
#
print ("Scaling...")
train, scaler = preprocess_data(train)
test, scaler = preprocess_data(test, scaler)


train = np.asarray(train, dtype=np.float32)        
labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)

net = NeuralNet(
    layers=[  
        ('input', InputLayer),
        ('dropout0', DropoutLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('output', DenseLayer),
        ],

    input_shape=(None, len(train[1])),
    dropout0_p=0.4,
    hidden1_num_units=100,
    hidden1_W=Uniform(),
    dropout1_p=0.3,
    hidden2_num_units=50,
    #hidden2_W=Uniform(),

    output_nonlinearity=sigmoid,
    output_num_units=1, 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(np.float32(0.01)),
    update_momentum=theano.shared(np.float32(0.9)),    
    # Decay the learning rate
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
                       AdjustVariable('update_momentum', start=0.9, stop=0.99),
                       ],
    regression=True,
    y_tensor_type = T.imatrix,                   
    objective_loss_function = binary_crossentropy,
    #batch_iterator_train = BatchIterator(batch_size = 256),
    max_epochs=40,
    eval_size=0.2,
    #train_split =0.0,
    verbose=2,
    )


seednumber=1235
np.random.seed(seednumber)
net.fit(train, labels)

preds = net.predict_proba(test)[:,0]

pd.DataFrame({"ID": testId, "PredictedProb": preds}).to_csv('data/subs/nn1.csv', index=False)
