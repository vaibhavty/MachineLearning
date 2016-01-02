# This applies ECOC algorithm using adaptive boosting to predict for multiclass classification problem
# Generate ECOC code table based on number of classes and number of functions(20 here)
# For every function
#   update training output lables with the class and function mapping
#   run adaptive boosting to generate set of weak classifiers as a function model

# For prediction run over all the function models and generate an output code for a datapoint
# Predict the class whose ECOC code is closest to the output code generated

__author__ = 'vaibhavtyagi'

import numpy as np
import DecisionStumps
from random import randint,sample



numFeatures=1754
numFunctions = 20

trainFile = '8newsgroup/train.trec/feature_matrix.txt'
testFile = '8newsgroup/test.trec/feature_matrix.txt'


# Read datafile into python lists
def readFile(fileName):
    global numFeatures
    fp = open(fileName,'r')
    dataList = fp.readlines()
    dataLen = len(dataList)
    #print dataLen
    dataSet = np.zeros((dataLen,numFeatures))
    labelList = list()
    i=0
    for line in dataList:
        line = line.split()
        label = int(line[0])
        labelList.append(label)
        for data in line[1:]:
            data = data.split(':')
            index = int(data[0])
            val = float(data[1])
            dataSet[i][index] = val
        i+=1
    #print dataSet[1][1210]
    return labelList,dataSet


# Genereate ECOC code table based on number of classes and number of functions
def generateErrorCodes(classCount,numFunctions):
    reqd = True
    while reqd:
        errCode = np.random.randint(2, size=(classCount,numFunctions))
        #print errCode[0]
        for code in errCode:
            if (code==np.ones(numFeatures)):
                reqd = True
            elif (code==np.zeros(numFeatures)):
                reqd= True
            elif ((code==1).sum() == classCount):
                reqd =True
            elif ((code==0).sum() == classCount):
                reqd = True
            else:
                reqd =False

    return errCode


# Update data labels based on class and function mapping in ECOC code table
def updateDataLabel(labelList,errColumn):
    i=0
    #print labelList
    for label in labelList:
        val = errColumn[label]
        labelList[i] = val
        i+=1
    #print labelList
    return labelList




# Returns error for a decison stump
def getBestThresh(node,val):
    err=0
    for dPoint in node:
        if (dPoint[0] <= val) & (dPoint[2]==1):
            err+=dPoint[1]
        if (dPoint[0] > val) & (dPoint[2]==0):
            err+=dPoint[1]
    return err


# Randomly select a feature/threshold combination as decision stump
def getDecisionStump(X,Y,W):
    X = np.array(X)
    featureLen = len(X[0])
    index = randint(0,numFeatures-1)
    vals = X[:,index]
    val = sample(vals,1)[0]
    #print Y
    err= getBestThresh(zip(vals,W,Y),val)
    #print 'Local Error:',err
    return (index,val,err)




# Update weights over all data points based on decison stump selected
def updateWeight(tpl,X,Y,W):
    X = np.array(X)
    feature = tpl[0]
    thresh = tpl[1]
    error = tpl[2]

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




# Calculate error for a function for a data point
def getCode(dPoint,hList,thresh):
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

        #print HX
        if HX <= thresh:
            pred=0
        else:
            pred=1

    return pred


# Predict nearest class for an output code
def getNearestClass(codeList):
    global errCode
    tempList = list()
    print codeList
    for row in errCode:
        i=0
        temp = 0
        for item in row:
            temp += abs(item - codeList[i])
            i+=1
        tempList.append(temp)
    print tempList
    return tempList.index(min(tempList))



# Calculate total error in prediction
def getError(dataSet,labelList,allHList):
    i =0
    err=0
    for dPoint in dataSet:
        codeList = list()
        for row in allHList:
            code = getCode(dPoint,row,0)
            codeList.append(code)

        cls = getNearestClass(codeList)
        print cls,labelList[i]
        if cls != labelList[i]:
            err+=1
        i+=1
        #if i==5:

    err = err /float(len(dataSet))
    return err

labelList,dataSet = readFile(trainFile)

classes = sorted(set(labelList))
classCount = len(classes)


errCode = generateErrorCodes(classCount,numFunctions)
print errCode



# Run adaptive boosting over all functions
allHList = list()
for func in range(numFunctions):
    labelList,dataSet = readFile(trainFile)
    errColumn = errCode[:,func]
    labelList = updateDataLabel(labelList,errColumn)

    #Generate initial weights as equal over all sets
    weight = np.empty(len(dataSet))
    a = 1/float(len(dataSet))
    weight.fill(a)

    #Calculate best H(x) and update weights
    hList = list()
    itr=0
    while True:
        #print sum(weight)
        tpl = getDecisionStump(dataSet,labelList,weight)

        weight,alpha = updateWeight(tpl,dataSet,labelList,weight)
        obj = DecisionStumps.DecisionStumps(tpl[0],tpl[1],alpha)
        hList.append(obj)
        print itr
        if itr==1000:
            break
        itr+=1

    allHList.append(hList)




labelList,dataSet = readFile(testFile)
#print dataSet[1][1154]


err = getError(dataSet,labelList,allHList)

print 'err is:',err
