
# Applies dual perceptron algorithm using linear and gaussian kernels
__author__ = 'vaibhavtyagi'

import numpy as np
from sklearn import preprocessing
import KNN

#inputFile = "perceptronData.txt"
inputFile = "twoSpirals.txt"


# Load data to python lists
def readFile(file,normalize):
    dataSet = np.loadtxt(file,dtype=float,delimiter="\t")
    xList = list()
    yList = list()
    if (normalize):
        feature_data = preprocessing.scale(dataSet[:, range(dataSet.shape[1] - 1)])
        dataSet = np.c_[feature_data, dataSet[:, -1]]
    #np.random.shuffle(dataSet)
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


# Generate K matrix
def generatekmatrix(dataSet,targetSet,model):
    obj = KNN.knn(0)
    size = len(dataSet)
    kMatrix = np.zeros((size,size))
    i=0
    predList = list()
    for data in dataSet:
        if model == 'Gaus':
            predList = obj.Gaussian(dataSet,targetSet,data,0.125)
        if model == 'Dot':
            predList = obj.Dot(dataSet,targetSet,data)
        #print predList
        #exit()
        j=0
        for val in predList:
            kMatrix[i][j] = val[0]
            j+=1
        i+=1
    return kMatrix



# Run dual perceptron algorithm using linear kernel based on model

def runDualPerceptron(dataSet,targetSet,model):
    total = len(targetSet)
    m= np.zeros(total)
    itr=0

    kMatrix = generatekmatrix(dataSet,targetSet,model)
    misclass = True
    while misclass:
        print 'Iteration',itr
        misclass = False
        misCount=0
        for i in range(total):
            pred=0
            for j in range(total):
                pred+=kMatrix[i][j] * m[j]
            pred = np.sign(pred)

            label = targetSet[i]
            if pred!= label:
                m[i]+=label
                misclass = True
                misCount+=1
        itr+=1
        print 'Misclass',misCount



# Dot
#dataSet,targetSet=readFile(inputFile,True)

# Gaussian
dataSet,targetSet=readFile(inputFile,False)

#runDualPerceptron(dataSet,targetSet,'Dot')
runDualPerceptron(dataSet,targetSet,'Gaus')

