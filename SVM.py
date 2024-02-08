import numpy as np
import pickle as pkl
from sklearn.model_selection import GridSearchCV
#from dask_ml.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import multiprocessing
from sklearn.metrics import accuracy_score
import joblib
from sklearn.svm import SVC
from dask_jobqueue import PBSCluster
from sklearn.model_selection import cross_val_score

fdir = "/username/folder/file.pkl"
with open(fdir,'rb') as pkfile:
    datamap = pkl.load(pkfile)

with open(fdir,'rb') as pkfile:
    X = pkl.load(pkfile)

clf = SVC()
y = datamap['y']
g = datamap['groups']

SVCparams = {'kernel':['rbf','linear','poly', 'sigmoid'],
            'gamma':['scale','auto'],
            'shrinking':[False, True],
            'decision_function_shape':['ovr','ovo'], # TO BE REMOVED IF PUBLISHED
            'degree':[1,2,3,4]}

outercv = GroupKFold(n_splits=22)
innercv = GroupKFold(n_splits=21)

cval_scores = []
acc_scores = []
ypreds = []
ytests = []
counter = 1
validation_list = np.zeros((352,2))

print('Starting cross-validations...')
for trainid, testid in outercv.split(X, y, groups=g):
    x_train, x_test = X[trainid,:], X[testid,:]
    y_train, y_test = y[trainid], y[testid]
    # validation list
    validation_list[testid,0] = y_test
    # ===============
    print('Grid searching for the best estimator...')
    grid = GridSearchCV(clf, SVCparams, cv=innercv, n_jobs=-1)
    grid.fit(x_train, y_train, groups=g[trainid])
    final_model = grid.best_estimator_
    print('Best classifier selected is:', grid.best_estimator_)
    print('Best score after the grid search is:', grid.best_score_)
    cvalscore = cross_val_score(final_model, x_train, y_train, groups=g[trainid], cv=innercv, n_jobs=-1)
    cval_scores.append(cvalscore)
    print('Crossvalscore of the best estimator:', cvalscore)
    y_pred = final_model.predict(x_test)
     # validation list
    validation_list[testid,1] = y_pred
    # ===============
    ypreds.append(y_pred)
    ytests.append(y_test)
    accscore = accuracy_score(y_pred, y_test)
    acc_scores.append(accscore)
    print('Accuracy score for the best estimator is:', accscore)

with open('SVMcval_scores.pkl', 'wb') as pkfile:
    pkl.dump(cval_scores, pkfile)

with open('SVMacc_scores.pkl', 'wb') as pkfile:
    pkl.dump(acc_scores, pkfile)

with open('SVMypreds.pkl', 'wb') as pkfile:
    pkl.dump(ypreds, pkfile)

with open('SVMytests.pkl', 'wb') as pkfile:
    pkl.dump(ytests, pkfile)

with open('SVMvalidation_list.pkl', 'wb') as pkfile:
    pkl.dump(validation_list, pkfile)