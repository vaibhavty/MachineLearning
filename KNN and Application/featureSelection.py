# Implementation of feature selection using Relief algorithm and kNN
__author__ = 'vaibhavtyagi'


import numpy as np
from sklearn import preprocessing
import KNN

inputFile = '../Dataset2/spambase.data'
dataSet = list()



# Load data to python lists
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    feature_data = preprocessing.scale(dataSet[:, range(dataSet.shape[1] - 1)])
    dataSet = np.c_[feature_data, dataSet[:, -1]]
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])
        '''
        if data[-1] == 0:
            yList.append(-1)
        else:
            yList.append(1)
        '''
    return xList,yList


# Update W using nearest neighbour in same and opposite class
def updateW(w,dPoint,X,sameIndex,diffIndex):
    sameX = X[sameIndex]
    diffX = X[diffIndex]
    for i in range(len(w)):
        w[i] +=  ((dPoint[i] - diffX[i])**2 - (dPoint[i] - sameX[i])**2)
    return w



# Run over all data points
# Returns a list of W with feature importance
def getBestFeatures(X,Y):
    total = len(Y)
    w= np.zeros(len(xTrain[0]))
    i=0
    for dPoint in X:
        minDistSame=np.inf
        sameIndex = 0
        minDistDiff=np.inf
        diffIndex=0
        j=0
        for dPoint1 in X:
            dist = np.linalg.norm(dPoint - dPoint1)
            if (Y[j] == Y[i]) & (minDistSame > dist):
                minDistSame = dist
                sameIndex = j
            if (Y[j] != Y[i]) & (minDistDiff > dist):
                minDistDiff = dist
                diffIndex = j

            j+=1
        w = updateW(w,dPoint,X,sameIndex,diffIndex)
        i+=1

    #Normalize W
    for i in range(len(w)):
        w[i]/=float(total)
    return w


def getDataSet(dataSet,reqdIndex):
    tempList = list()
    for index in reqdIndex:
        feature = dataSet[:,index]
        try:
            tempList = np.c_[feature,tempList]
        except:
            tempList = feature
        #print tempList

    return tempList



# Predict accuracy on testing data using most important feature selected using relief algorithm
def getAccuracyK(k,xTrain,yTrain,X,Y):


    Acc=[0,0,0]
    i=0
    obj = KNN.knn(k)
    for point in X:
        #print i
        dataList = obj.Euclidian(xTrain,yTrain,point)
        predList = obj.returnTopK(dataList,'Eucl')

        j=0
        for val in Acc:
            if predList[j]==Y[i]:
                Acc[j]+=1
            j+=1

        i+=1


    j=0
    for val in Acc:
        Acc[j]/=float(len(Y))
        j+=1


    return Acc





dataSet,targetSet=readFile(inputFile)

total = len(targetSet)
reqd = int(total * 0.7)

#print targetSet[460:100]
xTrain = np.array(dataSet[0:reqd])
yTrain = np.array(targetSet[0:reqd])
#xTest = dataSet[reqd:reqd+200]
#yTest = targetSet[reqd:reqd+200]

xTest = dataSet[reqd:]
yTest = targetSet[reqd:]


w = getBestFeatures(xTrain,yTrain)

reqdIndex = np.array(w).argsort()[-5:][::-1]
print reqdIndex
dataSet = getDataSet(np.array(dataSet),reqdIndex)

xTrain = np.array(dataSet[0:reqd])
yTrain = np.array(targetSet[0:reqd])
#xTest = dataSet[reqd:reqd+200]
#yTest = targetSet[reqd:reqd+200]
xTest = dataSet[reqd:]
yTest = targetSet[reqd:]


k=[1,3,7]

print getAccuracyK(k,xTrain,yTrain,xTest,yTest)