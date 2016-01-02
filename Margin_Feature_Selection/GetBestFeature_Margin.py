# Feature Selection using margin by adaptive boosting

__author__ = 'vaibhavtyagi'

from random import shuffle
import numpy as np
import pylab as py
import DecisionStumps
from random import sample
k=10
bCount=4
inputFile = '../Dataset2/spambase.data'


# read datafile to python list
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList


# generate k data folds over testing data
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


# create combinations of testing and training data
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


# Returns total error for a threshold value choosen
def getBestThresh(node,val):
    #print 'In Get Best thresh'
    err=0
    for dPoint in node:
        if (dPoint[0] <= val) & (dPoint[2]==1):
            err+=dPoint[1]
        if (dPoint[0] > val) & (dPoint[2]==0):
            err+=dPoint[1]
    return err



# Greedily select a threshold/ feature combination
def getDecisionStump(X,Y,W):
    #print 'In Get Decision stump'
    X = np.array(X)
    featureLen = len(X[0])
    featureList = list()
    for i in range(featureLen):
        vals = X[:,i]
        val = sample(vals,1)[0]
        err= getBestThresh(zip(vals,W,Y),val)


        threshList = list()
        data = zip(vals,W,Y)
        valSet = set(vals)
        for val in valSet:
            print 'a'
            err= getBestThresh(data,val)
            merr = abs(0.5 - err)
            threshList.append((val,merr,err))
        threshList.sort(key=lambda tup: tup[1])
        #print threshList
        tpl = threshList[-1]
        tpl = (i,tpl[0],tpl[1],tpl[2])
        
        tpl = (i,val,abs(0.5 - err),err)
        featureList.append(tpl)
        #if i==20:
        #    break
    featureList.sort(key=lambda tup: tup[2])
    #print 'Local Error:',featureList[-1][3]
    return featureList[-1]



# Update weights across all datapoints based on error from feature/  threshold selection
def updateWeight(tpl,X,Y,W):
    #print 'In update Weight'
    X = np.array(X)
    feature = tpl[0]
    thresh = tpl[1]
    error = tpl[3]

    #Calculate alpha for the error
    alpha =  0.5 * np.log((1 - error)/float(error))

    vals = X[:,feature]
    #print 'Alpha:',alpha
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


    #Normalize W
    total = np.sum(W)
    W /= float(total)
    return W,alpha


# Get testing error across all testing points
def getError(X,Y,hList,thresh):
    i=0
    err=0
    TP=0
    FP=0
    TN=0
    FN=0
    threshList = list()
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




# Make a feature list from the selected decision stumps during adaptive boosting
def getFeatureList(hList):
    featureList = set()
    for obj in hList:
        featureList.add(obj.feature)
    return featureList



# Get feature importance using margin .
# Run across all data points by comparing threshold for every decision stump
# add all margin values
# normalize margin values for all feature
# return features with top margin as most important features
def getFeatureImportance(X,Y,hList):
    i=0
    featureList = getFeatureList(hList)
    marginList = np.zeros(len(featureList))
    for dpoint in X:

        if Y[i] == 1:
            lx = 1
        else:
            lx = -1

        tempList = np.zeros(len(featureList))
        j=0
        for feature in featureList:
            temp=0
            for obj in hList:
                if obj.feature == feature:
                    alpha = obj.alpha
                    if dpoint[feature] <= obj.threshold:
                        HX = -1
                    else:
                        HX = 1

                    temp += lx * alpha * HX
            tempList[j]+= temp
            j+=1
        marginList = np.add(marginList,tempList)
        i+=1
    #print marginList
    total =  np.sum(marginList)
    marginList /= total
    #print marginList
    featureList = list(featureList)
    reqdList = zip(featureList,marginList)
    reqdList.sort(key=lambda tup: tup[1],reverse=True)
    return reqdList




xList,yList = readFile(inputFile)

kFoldData,kFoldLabel=generateKSets(xList,yList)
combinationSet=createCombination(kFoldData,kFoldLabel)




i=0
total=0
totalE =0



# Run for all data set combinations
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

    #Calculate best H(x) and update weights
    hList = list()
    itr=0
    while True:
        print 'Running Iteration:',itr
        tpl = getDecisionStump(xTrain,yTrain,weight)
        localError.append(tpl[3])

        weight,alpha = updateWeight(tpl,xTrain,yTrain,weight)

        obj = DecisionStumps.DecisionStumps(tpl[0],tpl[1],alpha)
        hList.append(obj)

        '''
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
        '''
        itrList.append(itr)
        if itr==1000:
            break
        itr+=1


    reqdList = getFeatureImportance(xTrain,yTrain,hList)
    featureList = [i for (i,j) in reqdList]
    print featureList

    exit()