# This program applies custom KNN iplementation on spambase dataset using Euclidian distance 
__author__ = 'vaibhavtyagi'


import numpy as np
from sklearn import preprocessing

import KNN

inputFile = '../Dataset2/spambase.data'
dataSet = list()
k=2


# Load input data to python lists
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



# Predict accuracy for k neighbours on testing data
def getAccuracyK(k,xTrain,yTrain,X,Y):


    Acc=[0,0,0]
    i=0
    obj = KNN.knn(k)
    for point in X:
        print i
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



# Predict accuracy for range on testing data
def getAccuracyWindow(xTrain,yTrain,X,Y):
    k=0
    Acc=0
    i=0
    obj = KNN.knn(k)
    for point in X:
        print i
        dataList = obj.Euclidian(xTrain,yTrain,point)
        pred = obj.returnWindow(dataList,-0.7)
        if pred == Y[i]:
            Acc+=1
        i+=1

    Acc/=float(len(Y))

    return Acc


dataSet,targetSet=readFile(inputFile)


total = len(targetSet)
reqd = int(total * 0.7)

#print targetSet[460:100]
xTrain = dataSet[0:reqd]
yTrain = targetSet[0:reqd]
#xTest = dataSet[reqd:reqd+100]
#yTest = targetSet[reqd:reqd+100]

xTest = dataSet[reqd:]
yTest = targetSet[reqd:]


k=[1,3,7]

#print getAccuracyK(k,xTrain,yTrain,xTest,yTest)

print getAccuracyWindow(xTrain,yTrain,xTest,yTest)