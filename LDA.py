import numpy as np
import pickle as pkl
#from sklearn.model_selection import GridSearchCV
from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import multiprocessing
from sklearn.metrics import accuracy_score
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dask_jobqueue import PBSCluster

if __name__ == "__main__":
    clf = LinearDiscriminantAnalysis()
    LDAparams = [
        {'solver':['svd','lsqr']},
        {'solver':['svd'], 'store_covariance':[True, False]},
        {'solver':['lsqr'], 'shrinkage':[None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    ]
    fdir = "/username/folder/file.pkl"
    datasets = pkl.load(open(fdir, 'rb'))
    #X = datasets['X']
    X = pkl.load(open(fdir, 'rb'))
    #print(X.shape)
    
    y = datasets['y']
    g = datasets['groups']

    outercv = GroupKFold(n_splits=22)
    innercv = GroupKFold(n_splits=21)
    acc_scores = []
    counter = 1
    print('Starting cross-validations...')
    for trainid, testid in outercv.split(X, y, groups=g):
        print('The', counter, 'split is under tests')
        counter += 1
        x_train, x_test = X[trainid,:], X[testid,:]
        y_train, y_test = y[trainid], y[testid]
        #print(x_train.shape)
        grid = GridSearchCV(clf, LDAparams, cv=innercv, n_jobs=-1, return_train_score=True)
        grid.fit(x_train, y_train, groups=g[trainid])
        y_pred = grid.predict(x_test)
        acc_scores.append(np.mean(accuracy_score(y_pred, y_test)))
        print('Scores for this set of parameters',acc_scores[-1])
        print('Set of parameters:', grid.best_params_)
    print('Entire scoring of this training is:', np.mean(np.array(acc_scores)))