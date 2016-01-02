# This progam applies active learning concept to train model and predict error
# Randomly choose 5% datapoints
# Run adaptive boosting on these datapoints
# Predict for testing set
# Choose next 5% and add testing point that are close to seprator to training data
# Repeat again
# Observation: error on testing data decreases with increase in training data

__author__ = 'vaibhavtyagi'
from random import randint,sample
import numpy as np
import pylab as py
import DecisionStumps


initPerct = 5
itrPerct = 2


inputFile = '../Dataset2/spambase.data'


# Read data into python list from input files
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList




# Get total error for a feature/threshold combination
def getBestThresh(node,val):
    err=0
    for dPoint in node:
        if (dPoint[0] <= val) & (dPoint[2]==1):
            err+=dPoint[1]
        if (dPoint[0] > val) & (dPoint[2]==0):
            err+=dPoint[1]
    return err




# Get best decision stump using optimum boosting
def getDecisionStump(X,Y,W):
    X = np.array(X)
    featureLen = len(X[0])
    featureList = list()
    for i in range(featureLen):
        vals = X[:,i]
        threshList = list()
        data = zip(vals,W,Y)
        valSet = set(vals)
        for val in valSet:
            err= getBestThresh(data,val)
            merr = abs(0.5 - err)
            threshList.append((val,merr,err))
        threshList.sort(key=lambda tup: tup[1])
        #print threshList
        tpl = threshList[-1]
        tpl = (i,tpl[0],tpl[1],tpl[2])
        featureList.append(tpl)
        #if i==20:
        #    break
    featureList.sort(key=lambda tup: tup[2])
    #print 'Local Error:',featureList[-1][3]
    return featureList[-1]



# Update weights for all datapoints based on the error for a feature/ threshold combination
def updateWeight(tpl,X,Y,W):
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


    #Normalize Weight
    total = np.sum(W)
    W /= float(total)
    return W,alpha


# calculate HX for all test points and return new training and testing sets
# add points close to separator to training set
def getFXList(X,Y,hList):
    global next
    err=0
    i=0
    dataList = list()
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


        #print HX
        if HX <= 0:
            pred=0
        else:
            pred=1

        if pred != Y[i]:
            err+=1

        temp = dPoint + [Y[i]] + [abs(HX)]
        dataList.append(temp)

    dataList.sort(key=lambda temp: temp[-1])
    trainSet = dataList[0:next+1]
    testSet = dataList[next+1:len(dataList)]
    err = err /float(len(Y))
    return err,trainSet,testSet





#Read input file data
xList,yList = readFile(inputFile)


#Calculate required percentage
total =  len(xList)
reqd = (total * initPerct)/100

next = (total * itrPerct)/100

#Get initial testing and training data
xTrain = xList[0:reqd+1]
xTest = xList[reqd+1:total]

#Get initial train and test labels
yTrain = yList[0:reqd+1]
yTest = yList[reqd+1:total]


#Iterate until training data reaches 50% of actual data
errList = list()
reqdList = list()
cnt=5
while True:
    print cnt
    itr=0
    hList = list()


    #Assign equal weights initially
    weight = np.empty(len(xTrain))
    a = 1/float(len(xTrain))
    weight.fill(a)


    # Train using ADA boosting for 100 iteration
    while True:
        tpl = getDecisionStump(xTrain,yTrain,weight)
        weight,alpha = updateWeight(tpl,xTrain,yTrain,weight)
        obj = DecisionStumps.DecisionStumps(tpl[0],tpl[1],alpha)
        hList.append(obj)
        if itr==5:
            break
        itr+=1


    err,trainSet,testSet = getFXList(xTest,yTest,hList)
    print err
    errList.append(err)
    reqdList.append(cnt)
    # add 2% data to train set
    for data in trainSet:
        xTrain.append(data[0:-2])
        yTrain.append(data[-2])

    #Generate new testing data
    xTest = list()
    yTest = list()
    np.random.shuffle(testSet)
    for data in testSet:
        xTest.append(data[0:-2])
        yTest.append(data[-2])


    cnt += itrPerct
    if cnt >= 50:
        break

print errList
print reqdList
print "******"
