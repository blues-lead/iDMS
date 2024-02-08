import numpy as np
import pickle as pkl
from sklearn.model_selection import GridSearchCV
#from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import multiprocessing
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from dask_jobqueue import PBSCluster
from sklearn.model_selection import cross_val_score

fdir = "/username/folder/file.pkl"
with open(fdir,'rb') as pkfile:
    datamap = pkl.load(pkfile)

with open(fdir,'rb') as pkfile:
    X = pkl.load(pkfile)

clf = GradientBoostingClassifier()
y = datamap['y']
g = datamap['groups']

GBparams = {
    'n_estimators':[50,100,150,200,250,300,350,400],
    'learning_rate':[1/i for i in range(10,100,10)],
    'loss':['deviance','exponential'],
    'criterion':['friedman_mse', 'mse', 'mae'],
    'max_features':['auto', 'sqrt', 'log2'],
    'warm_start':[True, False]
    
}

outercv = GroupKFold(n_splits=22)
innercv = GroupKFold(n_splits=21)

cval_scores = []
acc_scores = []
ypreds = []
ytests = []
counter = 1
print('Starting cross-validations...')
for trainid, testid in outercv.split(X, y, groups=g):
    x_train, x_test = X[trainid,:], X[testid,:]
    y_train, y_test = y[trainid], y[testid]
    print('Grid searching for the best estimator...')
    grid = GridSearchCV(clf, GBparams, cv=innercv, n_jobs=-1)
    grid.fit(x_train, y_train, groups=g[trainid])
    final_model = grid.best_estimator_
    print('Best classifier selected is:', grid.best_estimator_)
    print('Best score after the grid search is:', grid.best_score_)
    cvalscore = cross_val_score(final_model, x_train, y_train, groups=g[trainid], cv=innercv, n_jobs=-1)
    cval_scores.append(cvalscore)
    print('Crossvalscore of the best estimator:', cvalscore)
    y_pred = final_model.predict(x_test)
    ypreds.append(y_pred)
    ytests.append(y_test)
    accscore = accuracy_score(y_pred, y_test)
    acc_scores.append(accscore)
    print('Accuracy score for the best estimator is:', accscore)

with open('GBcval_scores.pkl', 'wb') as pkfile:
    pkl.dump(cval_scores, pkfile)

with open('GBacc_scores.pkl', 'wb') as pkfile:
    pkl.dump(acc_scores, pkfile)

with open('GBypreds.pkl', 'wb') as pkfile:
    pkl.dump(ypreds, pkfile)

with open('GBytests.pkl', 'wb') as pkfile:
    pkl.dump(ytests, pkfile)