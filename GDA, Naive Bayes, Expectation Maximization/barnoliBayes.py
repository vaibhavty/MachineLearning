# This applies simple bernoulli to classify data as spam or non spam
# compares mean for each feature in training set with testing data for prediction

__author__ = 'I849196'

from random import shuffle
import numpy as np
import math
import pylab
import pylab as py

np.seterr(divide='ignore')

inputFile = '../Dataset2/spambase.data'
k=10


# Read Inputfile
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList



# Generate k folds from input data
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


# Create testing and training combinations
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




# Calculate parameneter (mean, class probability) on training data
def calculateParam(X,Y):
    meanList1 = initializeList(1,len(X[0]))
    meanList0 = initializeList(1,len(X[0]))
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

    meanList1 /= float(cnt1)
    meanList0 /= float(cnt0)


    PY1 = cnt1/float(len(Y))
    PY0 = cnt0/float(len(Y))

    probLess1 = initializeList(1,len(X[0]))
    probMore1 = initializeList(1,len(X[0]))

    probLess0 = initializeList(1,len(X[0]))
    probMore0 = initializeList(1,len(X[0]))

    i=0
    for dPoint in X:
        j=0
        for val in dPoint:
            if Y[i]==0:
                if val <= meanList0[j]:
                    probLess0[j]+=1
                else:
                    probMore0[j]+=1

            if Y[i]==1:
                if val <= meanList1[j]:
                    probLess1[j]+=1
                else:
                    probMore1[j]+=1
            j+=1
        i+=1



    probLess0 = np.array(probLess0)/float(cnt0)
    probMore0 = np.array(probMore0)/float(cnt0)
    probLess1 = np.array(probLess1)/float(cnt1)
    probMore1 = np.array(probMore1)/float(cnt1)

    return meanList0,meanList1,probLess0,probMore0,probLess1,probMore1,PY1,PY0





# Use naive bayes to predict output label and return accuracy over all training points
def getAccuracy(meanList0,meanList1,probLess0,probMore0,probLess1,probMore1,PY1,PY0,X,Y,thresh):
    Acc=0
    FN=0
    FP=0
    TP=0
    TN=0
    i=0
    n = len(meanList0)
    threshList=list()
    for dPoint in X:
        j=0
        prob0 = 1
        prob1 = 1
        for data in dPoint:
            if data <= meanList0[j]:
                PXY = probLess0[j]
            else:
                PXY = probMore0[j]

            prob0 *= PXY



            if data <= meanList1[j]:
                PXY = probLess1[j]
            else:
                PXY = probMore1[j]

            prob1 *= PXY
            j+=1


        prob0 *=PY0
        prob1 *=PY1

        temp = (prob1/float(prob0))
        threshList.append(temp)

        if temp >= thresh:
            pred = 1
        else:
            pred = 0


        if pred == Y[i]:
            Acc+=1

        # Code for roc curve generation
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



# Run over all the data folds
i=0
total=0
totalE =0
for combination in combinationSet:
    xTrain=combination[0]
    yTrain=combination[1]
    xTest = combination[2]
    yTest =combination[3]
    meanList0,meanList1,probLess0,probMore0,probLess1,probMore1,PY1,PY0=calculateParam(xTrain,yTrain)
    Acc1,Err1,TP,FN,FP,TN,threshList = getAccuracy(meanList0,meanList1,probLess0,probMore0,probLess1,probMore1,PY1,PY0,xTest,yTest,1)
    threshList = sorted(threshList)
    fprList = list()
    tprList = list()

    for thresh in threshList:
        Acc,Err,TP,FN,FP,TN,threshList = getAccuracy(meanList0,meanList1,probLess0,probMore0,probLess1,probMore1,PY1,PY0,xTest,yTest,thresh)
        fprList.append(FP/float(FP+TN))
        tprList.append(TP/float(TP+FN))

    print np.trapz(tprList,fprList)
    k=0
    for val in fprList:
        print val,tprList[k]
        k+=1
    py.plot(fprList,tprList)
    py.show()

    print "Fold:",i
    print "Accuracy:",Acc1
    #print "FP:",FP,"FN:",FN,"Err:",Err
    total +=Acc1
    totalE+=Err1
    i+=1
    exit()


# Print average accuracy and error over all the folds
print "Average Accuracy over all folds is:",total/float(k)
print "Average Error over all folds is:",totalE/float(k)
