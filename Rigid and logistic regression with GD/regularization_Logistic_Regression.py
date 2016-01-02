# This applies regularization to logistic regression

__author__ = 'vaibhavtyagi'

import math
import numpy as np
from random import shuffle
from collections import Counter
import pylab as p
from sklearn import decomposition





dataSet = list()
lam = 0.1
K=10
reg_factor = 0.002

trainFeature = '../spam_polluted/train_feature.txt'
trainLabel = '../spam_polluted/train_label.txt'

testFeature = '../spam_polluted/test_feature.txt'
testLabel = '../spam_polluted/test_label.txt'
bCount = 4


# Read input data file
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=" ")
    return dataSet



xTrain = readFile(trainFeature)
yTrain = readFile(trainLabel)
yTrain = np.array([[i] for i in yTrain])



trainLen = len(xTrain)
print 'No of Training Data',trainLen
xTest = readFile(testFeature)
yTest = readFile(testLabel)
yTest = np.array([[i] for i in yTest])

print 'No. of testing Data',yTest.size


dataSet = np.concatenate((xTrain,xTest),axis=0)
pca = decomposition.PCA(n_components=200)
pca.fit(dataSet)
dataSet = pca.transform(dataSet)
xTrain = dataSet[0:trainLen]
xTest = dataSet[trainLen:]



# Calculate accuracy on the testing data
def calcAccuray(W, X, Y):
    Acc = 0
    tp=0
    tn=0

    for dPoint, actual in zip(X, Y):
        pred = np.vdot(dPoint, W)
        temp = 1 if pred > 1 else 0
        if actual[0] == temp:
            Acc += 1

        if (actual[0] == temp) & (actual[0]==0):
            tn+=1
        if (actual[0] == temp) & (actual[0]==1):
            tp+=1

    Acc = Acc * 100 / float(Y.size)
    return Acc,tp,tn



# Calculate value for the activation function sigmoid
def calcSigmoid(W, dPoint):
    temp = 1 / float(1 + np.exp(-1 * np.vdot(W, dPoint)))
    return temp



# Update weights for each feature, use batch mode for update
# we should use stochastic for bigger data set

def calculateCDW(X, Y):
    global lam
    rows, cols = X.shape
    W = np.random.random(cols)
    grad = [(Y[i] - calcSigmoid(W, X[i])) * X[i] for i in xrange(rows)]
    # Introduce regularization factor
    W =  W * (1 - (lam * reg_factor)) + lam * np.sum(grad, axis=0)
    SSEold = np.sum([(Y[i] - calcSigmoid(W, X[i])) ** 2 for i in xrange(rows)], axis=0)
    j = 0
    while True:
        grad = [(Y[i] - calcSigmoid(W, X[i])) * X[i] for i in xrange(rows)]
        W =  W * (1 - (lam * reg_factor)) +  lam * np.sum(grad, axis=0)
        SSE = np.sum([(Y[i] - calcSigmoid(W, X[i])) ** 2 for i in xrange(rows)], axis=0)
        if SSEold - SSE <= 0.001:
            break
        SSEold = SSE
        j+=1
        #print SSEold,SSE,j
        if j == 500:
            break
    return W


# Get mean and variance for normalization
def getParam(X):
    mean = X.mean(axis=0)
    var = X.std(axis=0)
    return mean,var

# Normalize data
def normalize(X,mean,var):
    X = (X-mean)/var
    return X






mean, var = getParam(xTrain)
xTrain = normalize(xTrain,mean,var)
xTrain = np.c_[xTrain, np.ones(yTrain.size)]


W = calculateCDW(xTrain, yTrain)


acc,tp,tn = calcAccuray(W,xTrain,yTrain)
print "Training Acc Is:", acc

xTest = normalize(xTest,mean,var)
xTest = np.c_[xTest, np.ones(yTest.size)]
acc,tp,tn= calcAccuray(W,xTest,yTest)
print "Testing Acc is:", acc


