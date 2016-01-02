# This program applies bagging using decision trees
# Randomly choose 100 samples with replacement from training set
# Generate a decision tree using these samples
# Repeat above steps and generate multiple decision trees
# For testing predict over combination of all the decision trees


__author__ = 'vaibhavtyagi'

import random
import numpy as np
import pylab as py
import DecisionStumps
import TreeNode


inputFile = '../Dataset2/spambase.data'


# Read input file into python list
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    np.random.shuffle(dataSet)

    return dataSet
    """
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList
    """



# calculate randomness in a set of data
def calculateIG(labelRow):
    total  = len(labelRow)

    p0 = list(labelRow).count(0)
    p1 = list(labelRow).count(1)

    if p0 !=0:
        p0 = p0/float(total)
        p0 = p0 * np.log2(p0)

    if p0 !=0:
        p1 = p1/float(total)
        p1 = p1 * np.log2(p1)

    p = -1 * (p0 + p1)

    return p


# Returns best threshold for split for a feature
def getBestThresh(f,trainSet):
    fRow = trainSet[:,f]
    labelRow = trainSet[:,-1]
    parentIG= calculateIG(labelRow)
    lParent = len(trainSet)

    currIG = 0
    currVal = 0
    currLeft = list()
    currRight = list()


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
            leftIG=0
        else:
            left = np.array(left)
            labelRow = left[:,-1]
            leftIG = calculateIG(labelRow)

        lRight = len(right)
        if lRight ==0:
            rightIG = 0
        else:
            right = np.array(right)
            labelRow = right[:,-1]
            rightIG = calculateIG(labelRow)


        #print parentIG

        IG = parentIG - (((lLeft)/float(lParent)) * leftIG) - (((lRight)/float(lParent)) * rightIG)
        #print IG
        if currIG < IG:
            currVal = val
            currIG =IG
            currLeft = left
            currRight = right

    return currVal,currIG,currLeft,currRight



# Returns best threshold/Feature combination for a split
def getBestFeature(trainSet):
    trainSet = np.array(trainSet)
    fCount = len(trainSet[0])-5
    # for all features
    currIG =0
    currF = 0
    currVal = 0
    currLeft = list()
    currRight = list()

    for f in range(fCount):
        val,IG,left,right = getBestThresh(f,trainSet)
        if currIG < IG:
            currVal = val
            currIG =IG
            currLeft = left
            currRight = right
            currF = f
    return currF,currVal,currIG,currLeft,currRight



# Generate decision tree
def getDecisionTree(trainSet,depth):
    #print depth
    if depth>=5:
        return ""

    if len(trainSet) <= 10:
        node = TreeNode.TreeNode()
        labels =  np.array(trainSet)
        labels = labels[:,-1]
        cnt0 = list(labels).count(0)
        cnt1 = list(labels).count(1)
        if cnt0 > cnt1:
            cnt=0
        else:
            cnt=1
        node.isLeafNode(True,cnt)
        return node


    currF,currVal,currIG,currLeft,currRight = getBestFeature(trainSet)
    #print len(currLeft),len(currRight),currF,currVal,currIG,len(trainSet)
    node = TreeNode.TreeNode()
    node.addParam(currF,currVal)
    node.isLeafNode(False,0)
    if (len(currLeft)==0) & (len(currRight) ==0):
        labels =  np.array(trainSet)
        labels = labels[:,-1]
        cnt0 = list(labels).count(0)
        cnt1 = list(labels).count(1)
        if cnt0 > cnt1:
            cnt=0
        else:
            cnt=1
        node.isLeafNode(True,cnt)
        return node
    else:
        left = getDecisionTree(currLeft,depth+1)
        right = getDecisionTree(currRight,depth+1)

        if (left=="") & (right==""):
            labels =  np.array(trainSet)
            labels = labels[:,-1]
            cnt0 = list(labels).count(0)
            cnt1 = list(labels).count(1)
            if cnt0 > cnt1:
                cnt=0
            else:
                cnt=1
            node.isLeafNode(True,cnt)
        else:
            node.addChild(left,right)
        return node



# Predict accuracy over testing points using all the decision trees generated
def getAccuracy(dataSet,TreeList):
    err=0
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
        cnt0 = list(valList).count(0)
        cnt1 = list(valList).count(1)

        if cnt0 > cnt1:
            cnt=0
        else:
            cnt=1

        if cnt != point[-1]:
            err+=1

    return (err/float(len(dataSet))) * 100

dataSet = readFile(inputFile)

print 'Data Size:',len(dataSet)


# Make 50 decision trees
TreeList = list()
itr=0
while True:
    print 'Creating tree number:',itr,'....'
    # take a sample from entire dataset
    trainSet = random.sample(dataSet,100)
    #get a decision tree
    DT = getDecisionTree(trainSet,0)
    TreeList.append(DT)
    itr+=1
    if itr==50:
        break



err=getAccuracy(dataSet,TreeList)

print '***'
print 'Err:',err
print 'Acc:',100-err
