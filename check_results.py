import numpy as np
import pickle as pkl
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

with open('LDAcval_scores.pkl', 'rb') as pkfile:
    cscores = np.array(pkl.load(pkfile))

with open('LDAacc_scores.pkl', 'rb') as pkfile:
    ascores = np.array(pkl.load(pkfile))

with open('SVMypreds.pkl', 'rb') as pkfile:
    ypreds = pkl.load(pkfile)

with open('SVMytests.pkl', 'rb') as pkfile:
    ytests = pkl.load(pkfile)


scores = 0
lens = [len(i) for i in ypreds]
weights = [lens[i]/sum(lens) for i in range(len(lens))]
for i in range(len(ypreds)):
    scores += accuracy_score(ypreds[i], ytests[i])*len(ypreds[i])
print(scores/sum(lens))
