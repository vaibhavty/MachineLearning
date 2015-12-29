__author__ = 'vaibhavtyagi'



import TreeNode
import math


trainingFile = '../Dataset1/housing_train.txt'
testingFile = '../Dataset1/housing_test.txt'


featureList = ['CRIM','ZN','INDUS','CHAS','NOX',
               'RM','AGE','DIS','RAD',
               'TAX','PTRATIO','B','LSTAT']
trainVal = dict()
testVal = dict()

trainTarget = dict()
testTarget = dict()



# Read data file
def readFile(fileName):
    fp=open(fileName,'r')
    return fp



# Load data into python dictionaries
def loadDataSet(fp):
    dataSet = dict()
    targetSet = dict()
    i=1
    for line in fp.readlines():
        valList = line.split()
        if len(valList) != 0:
            dataSet[i] = [float(j) for j in valList[:-1]]
            targetSet[i]= valList[-1]
            i+=1
    return dataSet,targetSet



# Normalize data using shift and scale normalization across testing and training data
def normalizeData():
    global trainVal
    global trainTarget
    global testVal
    global testTarget
    global featureList
    i=0
    for feature in featureList:
        allVal = list()
        for val in trainVal:
            allVal.append(trainVal[val][i])

        for val in testVal:
            allVal.append(testVal[val][i])

        minVal = min(allVal)

        for val in trainVal:
            trainVal[val][i] -=  minVal

        for val in testVal:
            testVal[val][i] -=  minVal

        allVal = list()
        for val in trainVal:
            allVal.append(trainVal[val][i])

        for val in testVal:
            allVal.append(testVal[val][i])

        maxVal = max(allVal)

        for val in trainVal:
            trainVal[val][i] = trainVal[val][i]/float(maxVal)

        for val in testVal:
            testVal[val][i] = testVal[val][i]/ float(maxVal)


        i+=1




# Calculate error at a node
def calcVariance(instFeature):
    uParent=0
    varParent=0
    valList=[k for (j,k) in instFeature]
    parentLen = len(valList)
    for val in valList:
        uParent += val

    try:
        uParent = uParent/parentLen
    except:
        uParent = 0

    for val in valList:
        varParent += math.pow((uParent - val),2)

    return varParent



# Find best threshold for a feature and return threshold feature tuple
def threshFeatureVal(threshSet,instFeature,varParent):
    dataList=list()
    varTotal = varParent

    for thresh in threshSet:
        left = list()
        right = list()
        for pair in instFeature:
            if pair[1] <= thresh:
                left.append(pair)
            else:
                right.append(pair)

        if len(left) == 0:
            varleft =0
        else:
            varleft=calcVariance(left)

        if len(right) == 0:
            varRight=0
        else:
            varRight=calcVariance(right)
        varParent = varleft+varRight


        totalVariance = varTotal - varParent
        dataList.append((totalVariance,left,right,thresh))



    tpl = sorted(dataList, key=lambda tup: tup[0])[len(dataList)-1]
    #tpl = sorted(dataList, key=lambda tup: tup[0])[0]
    return tpl




# Get best feature threshold pair across all features
def getNextFeature(instanceList):
    global trainVal
    dataList = list()
    i=0

    for feature in featureList:
        instFeature = list()
        threshSet = set()
        for instance in instanceList:
            instFeature.append((instance,trainVal[instance][i]))
            threshSet.add(trainVal[instance][i])
        instFeature = sorted(instFeature, key=lambda tup: tup[1])

        #get varience for parent node
        if len(instFeature)==0:
            varParent=0
        else:
            varParent=calcVariance(instFeature)


        tpl=threshFeatureVal(threshSet,instFeature,varParent)
        if round(tpl[0],4) == round(varParent,4):
            pass
        else:
            dataList.append((tpl,feature))

        #if i==5:
        #    break
        i+=1

    tpl = sorted(dataList, key=lambda tup: tup[0])[len(dataList)-1]
    #tpl = sorted(dataList, key=lambda tup: tup[0])[0]
    return tpl



# Generate regression tree on the testing data
# Find best threshold feature combination at each node for split
def generateTree(instanceList,parent,depth):
    global root
    global featureList
    val=0
    if depth==25:
        return ""

    if len(instanceList) < 6:
        return ""

    tpl=getNextFeature(instanceList)
    threshold = tpl[0][3]
    feature = tpl[1]
    left = tpl[0][1]
    right = tpl[0][2]

    #print feature,depth,threshold,len(instanceList)
    index = featureList.index(feature)

    for instance in instanceList:
        val+= float(trainTarget[instance])

    val = val/float(len(instanceList))

    parentNode = TreeNode.TreeNode(instanceList,parent,feature,threshold,val)
    left = [i for (i,j) in left]
    right = [i for (i,j) in right]
    depth+=1

    leftChild=generateTree(left,parentNode,depth)
    rightChild=generateTree(right,parentNode,depth)
    parentNode.addChild(leftChild,rightChild)
    if (leftChild=="") & (rightChild==""):
        parentNode.isLeafNode(True)
    else:
        parentNode.isLeafNode(False)
    return parentNode



fp=readFile(trainingFile)
trainVal,trainTarget = loadDataSet(fp)


fp=readFile(testingFile)
testVal,testTarget = loadDataSet(fp)


normalizeData()


root=generateTree(trainVal.keys(),"",0)



#Testing data
# Traverse tree by comapring threshold- feature combination at every node
# return leaf value as predicted value for the data point

def getLeafValue(featureData,node):
    while(node.isLeaf!=True):
        feature = node.feature
        index = featureList.index(feature)
        dataPoint = featureData[index]
        threshold = node.threshold
        print dataPoint, threshold
        if dataPoint <= threshold:
            if node.leftChild == "":
                break
            else:
                node = node.leftChild
                print 'Going Left'
        else:
            if node.rightChild == "":
                break
            else:
                node = node.rightChild
                print 'Going Right'

    return node.val


# Calculate error for all testing point
# Return error for predicted and expected label value
def getError(instanceList,root):
    err = 0
    for instance in instanceList:
        featureData = testVal[instance]
        leafVal=getLeafValue(featureData,root)
        print leafVal
        err += math.pow((float(testTarget[instance]) - leafVal),2)
    err = err/ len(instanceList)
    return err


#MSE=getError(testVal.keys(),root)
MSE=getError(trainVal.keys(),root)
print MSE




