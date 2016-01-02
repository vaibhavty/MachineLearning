# This program takes an input dataset.
# Train random boosting model by randomly choosing a number of feature threshold combination
# Get misclassifications on each step and update weights as per their errors.
# On testing data validate for all weak classifiers choosen during training and predict label

__author__ = 'vaibhavtyagi'

from random import randint,sample
import numpy as np
import pylab as py
import DecisionStumps

k=10
bCount=4
inputFile = '../Dataset2/spambase.data'


# read input file
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList


# Generate k folds on the input data
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


# Create testing and training combinations on the folds
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



# Get total misclassifications for a decision stump
def getBestThresh(node,val):
    err=0
    for dPoint in node:
        if (dPoint[0] <= val) & (dPoint[2]==1):
            err+=dPoint[1]
        if (dPoint[0] > val) & (dPoint[2]==0):
            err+=dPoint[1]
    return err


# Randomly choose a feature and threshold value for decision stump
def getDecisionStump(X,Y,W):
    X = np.array(X)
    featureLen = len(X[0])
    index = randint(0,51)
    vals = X[:,index]
    val = sample(vals,1)[0]
    err= getBestThresh(zip(vals,W,Y),val)
    print 'Local Error:',err
    return (index,val,err)



# Update weight vector as per error and alpha
def updateWeight(tpl,X,Y,W):
    X = np.array(X)
    feature = tpl[0]
    thresh = tpl[1]
    error = tpl[2]

    #Calculate alpha for the error
    alpha =  0.5 * np.log((1 - error)/float(error))

    vals = X[:,feature]
    print 'Alpha:',alpha
    #Update all Ws
    i=0
    for val in vals:
        temp=0
        if ((val <= thresh) & (Y[i]==1)) | ((val > thresh) & (Y[i] == 0)):
            temp = np.exp(alpha)
            prev = W[i]
            W[i] *= temp
            #if prev > W[i]:
            #    print "Error"
        else:
            temp = np.exp(-1 * alpha)
            prev = W[i]
            W[i] *= temp
            #if prev < W[i]:
            #    print "Error"

        i+=1


    #Normalize Weights
    total = np.sum(W)
    W /= float(total)
    return W,alpha




# Get error on the testing data
# Predict label for all the decision stumps selected
# The error will go down as number of decision stumps i.e. classifiers are increased
def getError(X,Y,hList,thresh):
    i=0
    err=0
    threshList = list()
    TP=0
    FP=0
    TN=0
    FN=0
    for dPoint in X:
        HX=0
        for obj in hList:
            feature = obj.feature
            threshold = obj.threshold
            alpha = obj.alpha
            if dPoint[feature] <= threshold:
                pred = -1
            else:
                pred =1

            HX += alpha * pred


        threshList.append(HX)
        #print HX
        if HX <= thresh:
            pred=0
        else:
            pred=1

        if pred != Y[i]:
            err+=1

        # For generating ROC curves
        if (pred==1) & (Y[i] == 1):
            TP+=1

        if (pred==0) & (Y[i] == 1):
            FN+=1

        if (pred==1) & (Y[i] == 0):
            FP+=1

        if (pred==0) & (Y[i] == 0):
            TN+=1



        i+=1

    err = err /float(len(Y))
    return err,threshList,FP,TP,FN,TN

xList,yList = readFile(inputFile)

kFoldData,kFoldLabel=generateKSets(xList,yList)
combinationSet=createCombination(kFoldData,kFoldLabel)




i=0
total=0
totalE =0



# Run for all data folds
for combination in combinationSet:
    localError = list()
    trainList = list()
    testList = list()
    itrList = list()
    aucList = list()

    #fetch data for a fold
    xTrain = combination[0]
    yTrain = combination[1]
    xTest = combination[2]
    yTest = combination[3]

    #Generate initial weights as equal over all sets
    weight = np.empty(len(xTrain))
    a = 1/float(len(xTrain))
    weight.fill(a)

    hList = list()
    itr=0
    
    # For every data fold create a pool of 1000 random classifiers and check improvement on each step
    while True:
        #print sum(weight)
        tpl = getDecisionStump(xTrain,yTrain,weight)
        localError.append(tpl[2])

        weight,alpha = updateWeight(tpl,xTrain,yTrain,weight)

        obj = DecisionStumps.DecisionStumps(tpl[0],tpl[1],alpha)
        hList.append(obj)

        trainError,threshList,FP,TP,FN,TN = getError(xTrain,yTrain,hList,0)
        trainList.append(trainError)
        print 'trainError',trainError

        testError,threshList,FP,TP,FN,TN = getError(xTest,yTest,hList,0)
        testList.append(testError)
        print 'testError',testError


        threshList = sorted(threshList)
        fprList = list()
        tprList = list()
        for thresh in threshList:
            testError,th,FP,TP,FN,TN = getError(xTest,yTest,hList,thresh)
            fprList.append(FP/float(FP+TN))
            tprList.append(TP/float(TP+FN))

        AUC = np.trapz(tprList,fprList)
        aucList.append(AUC)
        print 'AUC:',AUC
        print "****"

        itrList.append(itr)
        if itr==1000:
            break
        itr+=1




