__author__ = 'I849196'

from random import shuffle
import numpy as np
import pylab as py

np.seterr(divide='ignore')
inputFile = '../Dataset2/spambase.data'
#inputFile= '../Assignment 1/Assignment 1/Dataset2/spambase.data'
k=10
bCount=4



# Read input file
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList


# Generate k data folds for the input data
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


# Create combination of training and testing data on folds
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



# Get bucket for a feature in a datapoint
def getBucket(dPoint,bucket):
    global bCount
    for k in range(bCount):
        if  bucket[k] <= dPoint <=  bucket[k+1]:
            return True,k

    return False,0


# Calculate parameters
# mean per class
# probability for each bucket
# & class probabilities
def calculateParam(X,Y):
    global bCount
    meanList1 = initializeList(1,len(X[0]))
    meanList0 = initializeList(1,len(X[0]))
    meanList = initializeList(1,len(X[0]))
    cnt1 =0
    cnt0 =0
    i=0
    minList = np.amin(X,axis=0)
    maxList = np.amax(X,axis=0)

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
    meanList /= float(cnt1 + cnt0)

    PY1 = cnt1/float(len(Y))
    PY0 = cnt0/float(len(Y))


    bucketList = list()
    for i in range(len(minList)):
        tempList = list()
        tempList += [minList[i],maxList[i],meanList1[i],meanList0[i],meanList[i]]
        bucketList.append(sorted(tempList))


    probMatrix0 = initializeList(len(X[0]),bCount)
    probMatrix1 = initializeList(len(X[0]),bCount)

    i=0
    for val in Y:
        j=0
        for dPoint in X[i]:
            if val == 0:
                flag,n = getBucket(dPoint,bucketList[j])
                probMatrix0[j][n]+=1
            if val==1:
                flag,n = getBucket(dPoint,bucketList[j])
                probMatrix1[j][n] +=1
            j+=1
        i+=1

    for i in range(len(probMatrix0)):
        for j in range(len(probMatrix0[i])):
            probMatrix0[i][j] = (probMatrix0[i][j] + 1)/float(cnt0 + bCount)

    for i in range(len(probMatrix1)):
        for j in range(len(probMatrix1[i])):
            probMatrix1[i][j] = (probMatrix1[i][j] + 1)/float(cnt1 + bCount)


    #print probMatrix1
    return probMatrix0,probMatrix1,bucketList,PY1,PY0




#  Caclulate accuracy over all testing datapoint
# Use naive base for same
def getAccuracy(probMatrix0,probMatrix1,bucketList,PY1,PY0,X,Y,thresh):
    Acc=0
    i=0
    FN=0
    FP=0
    TP=0
    TN=0
    threshList = list()
    for dPoint in X:
        j=0
        prob0 = 1
        prob1 = 1
        for data in dPoint:
            flag,n = getBucket(data,bucketList[j])
            if flag:
                PXY0= probMatrix0[j][n]
                PXY1=probMatrix1[j][n]
            else:
                if data > bucketList[j][-1]:
                    n=len(bucketList[j])-2
                if data < bucketList[j][0]:
                    n=0

                PXY0= probMatrix0[j][n]
                PXY1= probMatrix1[j][n]

            prob0 *= PXY0
            prob1 *= PXY1



            j+=1
            #if j==1:
            #    exit()

        prob0 *= PY0
        prob1 *= PY1

        #temp = np.log(prob1/float(prob0))
        temp = (prob1/float(prob0))
        threshList.append(temp)

        if temp >= thresh:
            pred = 1
        else:
            pred = 0

        if pred == Y[i]:
            Acc+=1

        # logic for ROC curve generation
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



# Run over all the folds
i=0
total=0
totalE=0
for combination in combinationSet:
    xTrain=combination[0]
    yTrain=combination[1]
    xTest = combination[2]
    yTest =combination[3]
    probMatrix0,probMatrix1,bucketList,PY1,PY0=calculateParam(xTrain,yTrain)
    Acc1,Err1,TP,FN,FP,TN,threshList = getAccuracy(probMatrix0,probMatrix1,bucketList,PY1,PY0,xTest,yTest,1)
    threshList = sorted(threshList)
    fprList = list()
    tprList = list()

    for thresh in threshList:
        Acc,Err,TP,FN,FP,TN,threshList = getAccuracy(probMatrix0,probMatrix1,bucketList,PY1,PY0,xTest,yTest,thresh)
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


# Average and output error & accuracy over all the folds 
print "Average Accuracy over all folds is:",total/float(k)
print "Average Error over all folds is:",totalE/float(k)

