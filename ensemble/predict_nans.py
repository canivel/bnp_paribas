import pandas as pd


def predict_nans(features_train,
                 labels_train,
                 feature_train_test,
                 nan_features_test,
                 nan_labels_test,
                 y_test,
                 estimator):


    estimator.fit(features_train, labels_train)
    predictedNans = estimator.predict(feature_train_test)

    estimator.fit(nan_features_test, nan_labels_test)
    predictedNansTest = estimator.predict(y_test)

    return predictedNans, predictedNansTest