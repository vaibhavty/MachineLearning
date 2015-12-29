__author__ = 'vaibhavtyagi'

import math
import numpy as np
from random import shuffle
from collections import Counter
import pylab as p





inputFile = '../Dataset2/spambase.data'
dataSet = list()
lam = 0.01
K=10


# read input file
fp = np.loadtxt('../Dataset2/spambase.data',delimiter=',',dtype='float')
dPoints = fp[:, range(fp.shape[1] - 1)]
tPoints = fp[:, [-1]]



# Calculate accuracy for testing point
def calcAccuray(W, X, Y):
    Acc = 0
    tp=0
    tn=0

    for dPoint, actual in zip(X, Y):
        pred = np.vdot(dPoint, W)
        temp = 1 if pred > 0.0 else 0
        if actual[0] == temp:
            Acc += 1

        if (actual[0] == temp) & (actual[0]==0):
            tn+=1
        if (actual[0] == temp) & (actual[0]==1):
            tp+=1

    Acc = Acc * 100 / float(Y.size)
    return Acc,tp,tn


"""
def calcROC(W, X, Y, text,Acc):
    pLabel = []
    aLabel = []
    for dPoint, y in zip(X,Y):
        pred = np.vdot(dPoint,W)
        pLabel.append(pred)
        aLabel.append(y[0])

    roc = ROC(pLabel, aLabel, text)
    roc.compute()
    roc.plot(Acc)
"""


# calculate value for the sigmoid function
def calcSigmoid(W, dPoint):
    temp = 1 / float(1 + np.exp(-1 * np.vdot(W, dPoint)))
    return temp



# Calculate gradient descend using batch mode as data set is not very big
# use stochastic approach for a bigger dataset
def calculateCDW(X, Y):
    global lam
    rows, cols = X.shape
    W = np.random.random(cols)
    grad = [(Y[i] - calcSigmoid(W, X[i])) * X[i] for i in xrange(rows)]
    W +=  lam * np.sum(grad, axis=0)
    SSEold = np.sum([(Y[i] - calcSigmoid(W, X[i])) ** 2 for i in xrange(rows)], axis=0)
    j = 0
    while True:
        grad = [(Y[i] - calcSigmoid(W, X[i])) * X[i] for i in xrange(rows)]
        W +=  lam * np.sum(grad, axis=0)
        SSE = np.sum([(Y[i] - calcSigmoid(W, X[i])) ** 2 for i in xrange(rows)], axis=0)
        if SSEold - SSE <= 0.001:
            break
        SSEold = SSE
        j+=1
        #print SSEold,SSE,j
        if j == 500:
            break
    return W


# generate kfolds for input data
# split data into testing and training points
def genrateKFolds(X, Y):
    global K
    X = [i for i in xrange(len(Y))]
    X = list(X)
    shuffle(X)
    for fold in xrange(K):
        xTrain = [a for i, a in enumerate(X) if i % K != fold]
        yValid = [a for i, a in enumerate(X) if i % K == fold]
        yield xTrain,yValid



# calculate mean and variance for normalization of input data
def getParam(X):
    mean = X.mean(axis=0)
    var = X.std(axis=0)
    return mean,var

# normalize data
def normalize(X,mean,var):
    X = (X-mean)/var
    return X


i=0
tTrain = 0
tTest = 0

# run for all the data folds
for (m, n) in genrateKFolds(dPoints,tPoints):
    print "Fold:",i
    xTrain, xTest = m,n
    X= dPoints[xTrain]
    mean, var = getParam(X)
    X = normalize(X,mean,var)
    Y = tPoints[xTrain]
    X = np.c_[X, np.ones(Y.size)]
    print Y.shape
    print X.shape
    exit()
    W = calculateCDW(X, Y)
    acc,tp,tn = calcAccuray(W,X,Y)
    tTrain+=acc
    print "Training Acc Is:", acc
    X = dPoints[xTest]
    X = normalize(X,mean,var)
    Y = tPoints[xTest]
    X = np.c_[X, np.ones(Y.size)]
    acc,tp,tn= calcAccuray(W,X,Y)
    tTest +=acc
    print "Testing Acc is:", acc
    i+=1
    #calcROC(W, X, Y, "Logsistic Regression",acc)


# print average accuracy for training and testing data over all folds
print "Average Training Accuracy:",tTrain/float(K)
print "Average Testing Accuracy:",tTest/float(K)
