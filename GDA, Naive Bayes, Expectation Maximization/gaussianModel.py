__author__ = 'Vaibhav_Tyagi'

from random import shuffle
import numpy as np

import pylab as py
np.seterr(divide='ignore')

inputFile = '../Dataset2/spambase.data'
k=10


# read input file into numpy array
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList



# Generate k data folds
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


# Create combinations over data folds
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



def initializeList(rows, columns):
    dataList = list()
    for r in range(rows):
        temp = list()
        for c in range(columns):
            temp.append(0)
        dataList.append(temp)
    if rows==1:
        return dataList[0]
    else:
        return dataList



# Calculate parameter on training data
# mean for both classes over all features
# variance of both classes over all features
def calculateParam(X,Y):
    meanList1 = initializeList(1,len(X[0]))
    meanList0 = initializeList(1,len(X[0]))
    meanList = initializeList(1,len(X[0]))
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
        meanList = np.add(meanList,X[i])
        i+=1

    meanList1 /= float(cnt1)
    meanList0 /= float(cnt0)
    meanList /= float(cnt1+cnt0)

    PY1 = cnt1/float(len(Y))
    PY0 = cnt0/float(len(Y))

    variance1 = initializeList(1,len(X[0]))
    variance0 = initializeList(1,len(X[0]))
    variance = initializeList(1,len(X[0]))
    i=0
    for dPoint in X:
        j=0
        for val in dPoint:
            if Y[i]==0:
                variance0[j] += (val - meanList0[j]) ** 2

            if Y[i]==1:
                variance1[j] += (val - meanList1[j]) ** 2

            variance[j] += (val - meanList[j]) ** 2
            j+=1
        i+=1

    variance0 = np.array(variance0)/float(cnt0)
    variance1 = np.array(variance1)/float(cnt1)
    variance = np.array(variance)/float(cnt0 + cnt1)

    #print (variance)
    #exit()
    return meanList0,meanList1,variance0,variance1,PY1,PY0,variance



# Calcultae accuracy over all testing data points
# use naive bayes for prediction
def getAccuracy(meanList0,meanList1,variance0,variance1,PY1,PY0,X,Y,variance,thresh):
    Acc=0
    i=0
    FN=0
    FP=0
    TP=0
    TN=0
    threshList = list()
    n = len(meanList0)
    for dPoint in X:
        j=0
        prob0 = 1
        prob1 = 1
        for data in dPoint:
            if variance0[j] ==0:
                variance0[j] = 0.1
            if variance1[j] ==0:
                variance1[j]=0.1
            if variance[j] ==0:
                variance[j]==0.1

            #print variance[j]
            part1 = 1/float(((2 * np.pi)**0.5) * variance[j])
            part2 = ((data - meanList0[j]) ** 2)/float(2* (variance[j])**2)
            part2 = np.exp(-1 * part2)
            temp = (part1 * part2)
            prob0 *= temp

            part1 = 1/float(((2 * np.pi)**0.5) * variance[j])
            part2 = ((data - meanList1[j]) ** 2)/float(2* (variance[j])**2)
            part2 = np.exp(-1 * part2)
            temp = part1 * part2
            prob1 *= temp

            j+=1

            #if j==10:
            #    exit()

        prob0 *= PY0
        prob1 *= PY1

        temp = (prob1/float(prob0))
        threshList.append(temp)

        if temp >= thresh:
            pred = 1
        else:
            pred = 0

        if pred == Y[i]:
            Acc+=1

        # code for roc curve
        if (pred==1) & (Y[i] == 1):
            TP+=1

        if (pred==0) & (Y[i] == 1):
            FN+=1

        if (pred==1) & (Y[i] == 0):
            FP+=1

        if (pred==0) & (Y[i] == 0):
            TN+=1

        i+=1

    Err = (FP+FN)/float(len(Y))  * 100
    Acc = Acc/float(len(Y)) * 100
    return Acc,Err,TP,FN,FP,TN,threshList



xList,yList = readFile(inputFile)

kFoldData,kFoldLabel=generateKSets(xList,yList)
combinationSet=createCombination(kFoldData,kFoldLabel)



# Run over all k folds
i=0
total=0
totalE=0
for combination in combinationSet:
    xTrain=combination[0]
    yTrain=combination[1]
    xTest = combination[2]
    yTest =combination[3]
    meanList0,meanList1,variance0,variance1,PY1,PY0,variance=calculateParam(xTrain,yTrain)
    Acc1,Err1,TP,FN,FP,TN,threshList = getAccuracy(meanList0,meanList1,variance0,variance1,PY1,PY0,xTest,yTest,variance,1)
    """
    threshList = sorted(threshList)
    fprList = list()
    tprList = list()

    for thresh in threshList:
        Acc,Err,TP,FN,FP,TN,threshList = getAccuracy(meanList0,meanList1,variance0,variance1,PY1,PY0,xTest,yTest,variance,thresh)
        fprList.append(FP/float(FP+TN))
        tprList.append(TP/float(TP+FN))

    print np.trapz(tprList,fprList)
    k=0
    for val in fprList:
        print val,tprList[k]
        k+=1
    py.plot(fprList,tprList)
    py.show()
    """
    print "Fold:",i
    print "Accuracy:",Acc1
    #print "FP:",FP,"FN:",FN,"Err:",Err
    total +=Acc1
    totalE+=Err1
    i+=1
    #exit()



# Output avergae accuracy and error over all the k folds
print "Average Accuracy over all folds is:",total/float(k)
print "Average Error over all folds is:",totalE/float(k)

