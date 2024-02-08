import numpy as np
import pickle as pkl
from sklearn.model_selection import GridSearchCV
#from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import multiprocessing
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb
from dask_jobqueue import PBSCluster
from sklearn.model_selection import cross_val_score
import scipy


if __name__ == "__main__":
    fdir = "/username/folder/file.pkl"
    with open(fdir,'rb') as pkfile:
        datamap = pkl.load(pkfile)

    with open(fdir,'rb') as pkfile:
        X = pkl.load(pkfile)

    clf = xgb.XGBClassifier(objective="binary:logistic")
    y = datamap['y']
    g = datamap['groups']

    xgparams = {
            'n_estimators':[50, 100, 150, 200],
            'learning_rate':[0.01, 0.1, 0.2, 0.3],
            'max_depth':range(3, 10),
            'colsample_bytree':[i/10.0 for i in range(1,3)],
            'gamma':[i/10.0 for i in range(3)],
            'n_jobs':[-1],
            'use_label_encoder':[False],
            'eval_metric':['error']
        }

    outercv = GroupKFold(n_splits=22)
    innercv = GroupKFold(n_splits=21)

    cval_scores = []
    acc_scores = []
    ypreds = []
    ytests = []
    validation_list = np.zeros((352,2))
    counter = 1
    print('Starting cross-validations...')
    for trainid, testid in outercv.split(X, y, groups=g):
        x_train, x_test = X[trainid,:], X[testid,:]
        y_train, y_test = y[trainid], y[testid]
        validation_list[testid,0] = y_test
        print('Grid searching for the best estimator...')
        grid = GridSearchCV(clf, xgparams, cv=innercv, n_jobs=-1)
        grid.fit(x_train, y_train, groups=g[trainid])
        final_model = grid.best_estimator_
        print('Best classifier selected is:', grid.best_estimator_)
        print('Best score after the grid search is:', grid.best_score_)
        cvalscore = cross_val_score(final_model, x_train, y_train, groups=g[trainid], cv=innercv, n_jobs=-1)
        cval_scores.append(cvalscore)
        print('Crossvalscore of the best estimator:', cvalscore)
        y_pred = final_model.predict(x_test)
        validation_list[testid,1] = y_pred
        ypreds.append(y_pred)
        ytests.append(y_test)
        accscore = accuracy_score(y_pred, y_test)
        acc_scores.append(accscore)
        print('Accuracy score for the best estimator is:', accscore)

    with open('XGBcval_scores.pkl', 'wb') as pkfile:
        pkl.dump(cval_scores, pkfile)

    with open('XGBacc_scores.pkl', 'wb') as pkfile:
        pkl.dump(acc_scores, pkfile)

    with open('XGBypreds.pkl', 'wb') as pkfile:
        pkl.dump(ypreds, pkfile)

    with open('XGBytests.pkl', 'wb') as pkfile:
        pkl.dump(ytests, pkfile)

    with open('XGBvalidation_list.pkl', 'wb') as pkfile:
        pkl.dump(validation_list, pkfile)