# This program applies SMO algorithm using custom MySMO class implementation on spambase dataset

__author__ = 'vaibhavtyagi'

import numpy as np
from sklearn import svm,preprocessing,metrics
import random
import MySMO
np.random.seed(100)

inputFile = '../Dataset2/spambase.data'
dataSet = list()
k=2


# Load data to python lists
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    # normalize data
    feature_data = preprocessing.scale(dataSet[:, range(dataSet.shape[1] - 1)])
    dataSet = np.c_[feature_data, dataSet[:, -1]]
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        if data[-1] == 0:
            yList.append(-1)
        else:
            yList.append(1)
    return xList,yList




# Returns output from SVM function
def runSVMFunction(alpha,b,X,Y,dPoint):
    val = 0
    i=0
    dPoint = dPoint.T

    for point in X:
        part1 = alpha[i] * Y[i]
        part2 = X[i] * dPoint
        out = part1 * part2
        val+=out
        i+=1
    return (val + b)



# Predict accuracy on a testing data
def getAcc(alpha,b,X,Y,xTest,yTest):
    i=0
    acc=0
    for val in yTest:
        pred = runSVMFunction(alpha,b,X,Y,np.matrix(xTest[i]))
        print
        print
        print "******"
        print 'Predicted Value',pred
        if pred>=0:
            pred=1
        else:
            pred=-1
        print 'Predicted Label',pred,'Actual Label',val
        if pred == val:
            acc+=1
        i+=1
    return acc/float(len(yTest))





dataSet,targetSet=readFile(inputFile)


total = len(targetSet)
reqd = int(total * 0.05)


xTrain = np.array(dataSet[0:reqd])
yTrain = np.array(targetSet[0:reqd])
xTest = np.array(dataSet[reqd:reqd+500])
yTest = np.array(targetSet[reqd:reqd+500])


# Create object of MySMO class and get b and alpha list
Obj = MySMO.MySMO()
b,alpha = Obj.smo_simple(xTrain, yTrain, 0.75, 0.0001, 30)
print alpha,b
Acc = getAcc(alpha,b,xTrain,yTrain,xTest,yTest)


print Acc
