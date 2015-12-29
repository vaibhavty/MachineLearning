__author__ = 'vaibhavtyagi'



import math
import numpy as np


trainingFile = '../Dataset1/housing_train.txt'
testingFile = '../Dataset1/housing_test.txt'


featureList = ['CRIM','ZN','INDUS','CHAS','NOX',
               'RM','AGE','DIS','RAD',
               'TAX','PTRATIO','B','LSTAT']
trainVal = dict()
testVal = dict()

trainTarget = dict()
testTarget = dict()


# Read datafile
def readFile(fileName):
    fp=open(fileName,'r')
    return fp

#Load data to python lists
def loadDataSet(fp):
    dataSet = list()
    targetSet = list()
    i=1

    for line in fp.readlines():
        valList = line.split()
        if len(valList) != 0:
            dataSet.append([float(j) for j in valList[:-1]])
            targetSet.append([float(valList[-1])])
            i+=1
    return dataSet,targetSet



# Normalize data using shift and scale across all testing and training points
def normalizeData():
    global trainVal
    global trainTarget
    global testVal
    global testTarget
    global featureList
    i=0
    for feature in featureList:
        allVal = list()
        j=0
        for val in trainVal:
            allVal.append(val[i])
            j+=1

        j=0
        for val in testVal:
            allVal.append(val[i])
            j+=1

        minVal = min(allVal)


        j=0
        for val in trainVal:
            trainVal[j][i] -=  minVal
            j+=1


        j=0
        for val in testVal:
            testVal[j][i] -=  minVal
            j+=1

        allVal = list()

        j=0
        for val in trainVal:
            allVal.append(val[i])
            j+=1

        j=0
        for val in testVal:
            allVal.append(val[i])
            j+=1

        maxVal = max(allVal)

        j=0
        for val in trainVal:
            trainVal[j][i] = trainVal[j][i]/float(maxVal)
            j+=1

        j=0
        for val in testVal:
            testVal[j][i] = testVal[j][i]/ float(maxVal)
            j+=1

        i+=1



# Calculate W on training dataset
def calculateW(X,Y):
    X=np.matrix(X)
    X=np.c_[X,np.ones(len(Y))]
    Y=np.matrix(Y)
    XT=X.transpose()
    val = XT * X
    #print val[0]
    val = val.I
    val = val * XT
    val = val * Y
    return X,val



# Predict testlabels for testing dataset
def getTestLabels(X,W):
    X=np.c_[X,np.ones(len(X))]
    X=np.matrix(X)
    W=np.matrix(W)
    val = X*W
    return val



# Calculate mean square error on actual and testing data based
# for testing data
def calcMSE(yPred,testTarget):
    MSE=0
    SE=0
    tLen=len(testTarget)
    yPred = np.array(yPred)
    for i in range(len(testTarget)):
        val = testTarget[i][0] - yPred[i][0]
        val = math.pow(val,2)
        SE+=val

    MSE = SE/float(tLen)
    return MSE




fp=readFile(trainingFile)
trainVal,trainTarget = loadDataSet(fp)


fp=readFile(testingFile)
testVal,testTarget = loadDataSet(fp)


normalizeData()

X,W=calculateW(trainVal,trainTarget)

#yPred = getTestLabels(trainVal,W)
yPred = getTestLabels(testVal,W)

#MSE=calcMSE(yPred,trainTarget)
MSE= calcMSE(yPred,testTarget)

# Output mean square error across all testing dataset
print MSE