__author__ = 'I849196'

from random import shuffle
import numpy as np

inputFile = '../Dataset2/spambase.data'
k=10


# Read Input data file
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList


# Generate k folds i.e testing and traing data
def generateKSets(data,label):
    global k
    splitFactor = int(round(len(data)/k,0))
    counter = 0
    i=0
    kFoldData=dict()
    kFoldLabel=dict()
    for i in range(k):
        kFoldData[i] = data[counter:counter+splitFactor]
        kFoldLabel[i] = label[counter:counter+splitFactor]
        counter+=splitFactor
    kFoldData[i]+=data[counter:len(data)]
    kFoldLabel[i]+=label[counter:len(data)]
    return kFoldData,kFoldLabel



# Generate combinations of testing and training from folds
def createCombination(kFoldData,kFoldLabel):
    combinationSet = list()
    global k
    for i in range(k):
        trainSet = list()
        trainLabel = list()
        testSet = list()
        testLabel = list()
        for j in range(k):
            if (j==i):
                testSet+=kFoldData[j]
                testLabel+=kFoldLabel[j]
            else:
                trainSet+=kFoldData[j]
                trainLabel+=kFoldLabel[j]
        combinationSet.append((trainSet,trainLabel,testSet,testLabel))
    return combinationSet




# Calculte U0 , U1 and covariance matrix for trainging data
def calculateParam(X,Y):
    meanList1 = [0] * len(X[0])
    meanList0 = [0] * len(X[0])
    cnt1 =0
    cnt0 =0
    i=0
    for val in Y:
        if val==1:
            meanList1 =np.add(meanList1,X[i])
            cnt1+=1
        if val==0:
            meanList0 = np.add(meanList0,X[i])
            cnt0+=1
        i+=1

    meanList1 /= cnt1
    meanList0 /= cnt0

    abs = list()
    i=0
    for val in Y:
       j=0
       temp = list()
       if val == 1:
           temp = np.subtract(X[i],meanList1)
       if val == 0:
           temp = np.subtract(X[i],meanList0)

       temp = np.array(temp)
       abs.append(temp)
       i+=1

    abs = np.matrix(abs)
    absT = abs.T
    abs = np.dot(absT,abs)/float(len(X))
    #print abs.shape
    return abs,meanList0,meanList1




# Calculate accuracy on the testing data
# If prob of class 0 is more predict output as 0 else 1

def getAccuracy(abs,meanList0,meanList1,X,Y):
    n = len(X[0])
    det = np.linalg.det(abs)
    invAbs = np.linalg.inv(abs)
    Acc=0
    i=0
    for dPoint in X:
        part1 = 1 / float( (2 * np.pi)** (n/float(2)) * (det)**0.5)
        part2_0 = 0.5 * (np.subtract(dPoint,meanList0).T)  * invAbs * np.matrix(np.subtract(dPoint,meanList0)).T
        part2_1 = 0.5 * (np.subtract(dPoint,meanList1).T)  * invAbs * np.matrix(np.subtract(dPoint,meanList1)).T
        val1 = part1 * np.exp(-1 * part2_0)
        val2 = part1 * np.exp(-1 * part2_1)

        if val1 >= val2:
            pred = 0
        else:
            pred = 1

        if pred == Y[i]:
            Acc+=1
        i+=1

    Acc = Acc/float(len(Y)) * 100
    return Acc



xList,yList = readFile(inputFile)

kFoldData,kFoldLabel=generateKSets(xList,yList)
combinationSet=createCombination(kFoldData,kFoldLabel)



# Run for all the folds
i=0
total=0
for combination in combinationSet:
    xTrain=combination[0]
    yTrain=combination[1]
    xTest = combination[2]
    yTest =combination[3]
    abs,meanList0,meanList1=calculateParam(xTrain,yTrain)
    Acc = getAccuracy(abs,meanList0,meanList1,xTest,yTest)
    print "Accuracy for fold",i, "is:",Acc
    total +=Acc
    i+=1


# Average accuracy over all the folds
print "Average Accuracy over all folds is:",total/float(k)
