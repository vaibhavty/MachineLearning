# Gradient boosted regression tree for house price prediction
# Generate 10 regression trees with 2 levels each
# After each tree generation, update new output label as label - predicted label
# For final prediction take some over predictions from all the labels

__author__ = 'vaibhavtyagi'

import random
import numpy as np
import pylab as py

import TreeNode


trainFile = '../Dataset1/housing_train.txt'
testFile = '../Dataset1/housing_test.txt'

#trainFile= '../Assignment 1/Assignment 1/Dataset1/housing_train.txt'
#testFile = '../Assignment 1/Assignment 1/Dataset1/housing_test.txt'

# Read input datafile into python list
def readFile(fileName):
    fp=open(fileName,'r')
    dataSet = list()
    for line in fp.readlines():
        valList = line.split()
        if len(valList) != 0:
            dataSet.append([float(j) for j in valList])
    return dataSet




# calculate randomness in a set of data
def calculateVariance(labelRow):
    total  = sum(labelRow)
    l =  len(labelRow)
    mean = total/float(l)
    variance = 0
    for val in labelRow:
        variance+= (mean - val) ** 2

    return variance/float(l)


# Returns best threshold for a feature by checking degree on randomness upon split
def getBestThresh(f,trainSet):
    fRow = trainSet[:,f]
    labelRow = trainSet[:,-1]
    parentVar= calculateVariance(labelRow)

    currVar = 0
    currVal = 0
    currLeft = list()
    currRight = list()

    lParent = len(trainSet)
    for val in set(fRow):
        left = list()
        right = list()
        for dPoint in trainSet:
            if dPoint[f] <= val:
                left.append(dPoint)
            else:
                right.append(dPoint)


        lLeft = len(left)
        if lLeft ==0:
            leftVar=0
        else:
            left = np.array(left)
            labelRow = left[:,-1]
            leftVar = calculateVariance(labelRow)

        lRight = len(right)
        if lRight ==0:
            rightVar = 0
        else:
            right = np.array(right)
            labelRow = right[:,-1]
            rightVar = calculateVariance(labelRow)


        #print parentIG

        var = parentVar - (((lLeft)/float(lParent)) *leftVar + (((lRight)/float(lParent)) * rightVar))
        #print IG
        if currVar < var :
            currVal = val
            currVar = var
            currLeft = left
            currRight = right

    return currVal,currVar,currLeft,currRight



# Greedily choose best feature - threshold combination for a split
def getBestFeature(trainSet):
    trainSet = np.array(trainSet)
    fCount = len(trainSet[0])-5
    # for all features
    currVar =0
    currF = 0
    currVal = 0
    currLeft = list()
    currRight = list()

    for f in range(fCount):
        val,var,left,right = getBestThresh(f,trainSet)
        if currVar < var :
            currVal = val
            currVar =var
            currLeft = left
            currRight = right
            currF = f
    #print currF,currVal,currVar
    #exit()
    return currF,currVal,currVar,currLeft,currRight



# Generate 2 level regression tree
def getRegressionTree(trainSet,depth,text):
    #print depth
    #print text
    if depth>=2:
        return ""

    if len(trainSet) <= 30:
        node = TreeNode.TreeNode()
        labels =  [i[-1] for i in trainSet]
        val  = sum(labels)/len(labels)
        #print 'val',val
        node.isLeafNode(True,val)
        return node

    currF,currVal,currVar,currLeft,currRight = getBestFeature(trainSet)
    #print len(currLeft),len(currRight),currF,currVal,currIG,len(trainSet)
    node = TreeNode.TreeNode()
    #print currF,currVal
    node.addParam(currF,currVal)
    node.isLeafNode(False,0)

    if (len(currLeft)==0) & (len(currRight) ==0):
        #print '****'
        labels =  [i[-1] for i in trainSet]
        val  = sum(labels)/len(labels)
        #print 'val',val
        node.isLeafNode(True,val)
        return node
    else:
        left = getRegressionTree(currLeft,depth+1,'left')
        right = getRegressionTree(currRight,depth+1,'right')

        if (left=="") & (right==""):
            #print 'Both Null'
            labels =  [i[-1] for i in trainSet]
            val  = sum(labels)/len(labels)
            #print 'val',val
            node.isLeafNode(True,val)

        else:
            node.addChild(left,right)
        return node



# Normalize training and testing data
def generate(trainSet,testSet,normalize):
    dataSet = trainSet + testSet
    features = len(dataSet[0])-1
    dataSet = np.array(dataSet)
    np.random.shuffle(dataSet)
    if normalize:
        for i in range(features):
            col = dataSet[:,i]
            min = np.amin(col)
            for j in range(len(dataSet)):
                dataSet[j][i] -= min

            col = dataSet[:,i]
            max = np.max(col)
            for j in range(len(dataSet)):
                dataSet[j][i] /= float(max)

    trainSet = dataSet[0:len(trainSet)]
    testSet = dataSet[len(trainSet):len(dataSet)]
    return  trainSet,testSet



# Predict output over all datapoints in testing data
# Sum over outputs from all regression trees created during training
def getAccuracy(dataSet,TreeList):
    MSE=0
    for point in dataSet:
        valList = list()
        for tree in TreeList:
            isLeaf = tree.isLeaf
            #print tree
            while(isLeaf != True):
                feature = tree.feature
                thresh = tree.threshold

                pointVal = point[feature]

                if pointVal> thresh:
                    tree = tree.rightChild
                else:
                    tree = tree.leftChild

                isLeaf = tree.isLeaf
            valList.append(tree.val)

        total = sum(valList)
        #print 'total',total
        MSE += (total - point[-1]) ** 2

    return (MSE/float(len(dataSet)))



# Update new label for the next regression tree creation
def updateLabel(dataSet,DT):
    i=0
    for point in dataSet:
        tree = DT
        isLeaf = tree.isLeaf
        #print tree
        while(isLeaf != True):
            feature = tree.feature
            thresh = tree.threshold

            pointVal = point[feature]

            if pointVal> thresh:
                #print 'left'
                tree = tree.rightChild
            else:
                #print 'right'
                tree = tree.leftChild

            isLeaf = tree.isLeaf


        #print tree.val
        dataSet[i][-1] = point[-1] - tree.val
        i+=1
    return dataSet







trainSet = readFile(trainFile)
testSet = readFile(testFile)


trainSet,testSet = generate(trainSet,testSet,True)
print 'Training Count:',len(trainSet)
print 'Testing Count:',len(testSet)
#exit()


# Make 100 decision trees
TreeList = list()
itr=0

# Generate 10 regression trees
while True:
    # take a sample from entire dataset
    #print dataSet[1][-1],dataSet[5][-1]
    print 'Generating Tree Node:',itr,'....'
    DT = getRegressionTree(trainSet,0,'root')
    TreeList.append(DT)
    trainSet = updateLabel(trainSet,DT)
    itr+=1
    if itr==10:
        break


trainSet = readFile(trainFile)
testSet = readFile(testFile)
trainSet,testSet = generate(trainSet,testSet,True)


# Calculate overall errror over training and testing data
err=getAccuracy(trainSet,TreeList)
print 'Error is:',err
err=getAccuracy(testSet,TreeList)
print 'Error is:',err
#print 'Accuracy is:',100-err


