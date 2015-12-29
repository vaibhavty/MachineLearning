__author__ = 'vaibhavtyagi'

import TreeNode
import math
from collections import Counter
import pylab as plot
import numpy as np
from random import shuffle

inputFile = '../Dataset2/spambase.data'


featureList = ['word_freq_make', 'word_freq_address', 'word_freq_all',
               'word_freq_3d', 'word_freq_our', 'word_freq_over',
               'word_freq_remove', 'word_freq_internet', 'word_freq_order',
               'word_freq_mail', 'word_freq_receive', 'word_freq_will',
               'word_freq_people', 'word_freq_report', 'word_freq_addresses',
               'word_freq_free', 'word_freq_business', 'word_freq_email',
               'word_freq_you', 'word_freq_credit', 'word_freq_your',
               'word_freq_font', 'word_freq_000', 'word_freq_money',
               'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
               'word_freq_650', 'word_freq_lab', 'word_freq_labs',
               'word_freq_telnet', 'word_freq_857', 'word_freq_data',
               'word_freq_415', 'word_freq_85', 'word_freq_technology',
               'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
               'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
               'word_freq_original', 'word_freq_project', 'word_freq_re',
               'word_freq_edu', 'word_freq_table', 'word_freq_conference',
               'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
               'char_freq_$', 'char_freq_#', 'capital_run_length_average',
               'capital_run_length_longest', 'capital_run_length_total']




trainVal = dict()
testVal = dict()

trainTarget = dict()
testTarget = dict()
k=10

# Read file from disk
def readFile(fileName):
    fp=open(fileName,'r')
    return fp



# Load dataset(features and labels) into python dictionaries
def loadDataSet(fp):
    dataSet = dict()
    targetSet = dict()
    i=1
    data = list()
    for line in fp.readlines():
        data.append(line)
    shuffle(data)
    for line in data:
        valList = line.split(",")
        if len(valList) != 0:
            dataSet[i] = [float(j) for j in valList[:-1]]
            targetSet[i]= valList[-1]
            i+=1
    return dataSet,targetSet



# Generate k sets for k fold validation for global k value
def generateKSets(instanceList):
    global k
    kFoldData = dict()
    splitFactor = int(round(len(instanceList)/k,0))
    counter = 0
    i=0
    for i in range(k):
        kFoldData[i] = instanceList[counter:counter+splitFactor]
        counter+=splitFactor
    kFoldData[i]+=instanceList[counter:len(instanceList)]
    return kFoldData



# Create training and testing set combinations
def createCombination(kFoldData):
    combinationSet = list()
    global k
    for i in range(k):
        trainSet = list()
        testSet = list()
        for j in range(k):
            if (j==i):
                testSet+=kFoldData[j]
            else:
                trainSet+=kFoldData[j]
        combinationSet.append((trainSet,testSet))
    return combinationSet



# Calulate information gain for a split
def calcIG(instFeature):
    entParent=0
    #print instFeature
    valList=[k for (j,k) in instFeature]
    parentLen = len(valList)
    tempList = Counter(valList)
    for val in tempList:
        count = tempList[val]
        prob = count/float(parentLen)
        entParent += prob * math.log(prob,2)

    return (-1 * entParent)


# get best threshold for a feature and return feature threshold tuple
def threshFeatureVal(threshSet,instFeature,entParent):
    dataList=list()
    igList=list()
    lenParent = len(instFeature)
    for thresh in threshSet:
        #print "here"
        left = list()
        right = list()
        for pair in instFeature:
            if pair[1] <= thresh:
                left.append(pair)
            else:
                right.append(pair)

        lenLeft = len(left)
        if  lenLeft == 0:
            entleft =0
        else:
            entleft=calcIG(left)


        lenRight = len(right)
        if  lenRight == 0:
            entRight=0
        else:
            entRight=calcIG(right)


        IG = entParent - ( ((lenLeft/float(lenParent)) * entleft) + ((lenRight/float(lenParent)) * entRight) )

        dataList.append((IG,left,right,thresh))
        igList.append(IG)
    index = igList.index(max(igList))
    tpl = dataList[index]
    #print tpl
    return tpl




# Iterate on all features and return best feature threshold tuple
def getNextFeature(instanceList):
    global trainVal
    dataList = list()
    igList = list()
    currFeature=""
    i=0
    for feature in featureList:
        #print feature
        instFeature = list()
        threshSet = set()
        for instance in instanceList:
            instFeature.append((instance,trainVal[instance][i]))
            threshSet.add(trainVal[instance][i])
        #instFeature = sorted(instFeature, key=lambda tup: tup[1])

        #get varience for parent node
        if len(instFeature)==0:
            igParent=0
        else:
            igParent=calcIG(instFeature)

        tpl=threshFeatureVal(threshSet,instFeature,igParent)
        #if round(tpl[0],4) == round(igParent,4):
        if (igParent == tpl[0]):
            pass
        else:
            dataList.append(tpl)
            igList.append(tpl[0])

        i+=1
        #if i==2:
        #    exit()

    if len(igList) == 0:
        #print 'leaf'
        return ""
    else:
        index = igList.index(max(igList))
        tpl = (dataList[index],featureList[index])
        #print tpl
        return tpl



# Generate decision tree by spliting data based on best threshold feature combination
# use greedy approach for every split
# choose feature threshold for the best information gain
def generateTree(instanceList,parent,depth):
    #print depth
    global root
    global featureList
    val=0
    if depth== 25:
        return ""


    #print len(instanceList)
    if len(instanceList) < 20:
        return ""

    tpl=getNextFeature(instanceList)
    if tpl == "":
        return ""
    threshold = tpl[0][3]
    feature = tpl[1]
    left = tpl[0][1]
    right = tpl[0][2]

    #print feature,depth,threshold,len(instanceList)
    index = featureList.index(feature)
    tempList=list()
    for instance in instanceList:
        tempList.append(trainTarget[instance])
    val = Counter(tempList).most_common()[0][0]



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

    #print parentNode.feature
    return parentNode



# Iterate tree from root for a testing point
# move through the nodes by comapring feature and threshold values
# return the value stored for the leaf node
def getLeafValue(featureData,node):
    while(node.isLeaf!=True):
        feature = node.feature
        index = featureList.index(feature)
        dataPoint = featureData[index]
        threshold = node.threshold
        #print dataPoint, threshold
        if dataPoint <= threshold:
            if node.leftChild == "":
                break
            else:
                node = node.leftChild
                #print 'Going Left'
        else:
            if node.rightChild == "":
                break
            else:
                node = node.rightChild
                #print 'Going Right'

    return node.val



# Generate ROC plot for FP and TP comparision
def generatePlot(rocList,accuracy):
    Y = [i for (i,j) in rocList]
    X = [j for (i,j) in rocList]
    AUC = np.trapz(Y,X)
    plot.plot(Y,X)
    plot.xlabel('False Positive')
    plot.ylabel('True Positive')
    plot.title('ROC curve(Decision Tree) Accuracy:'+str(accuracy)+' AUC:'+str(AUC))
    plot.grid(True)
    plot.savefig("test.png")
    plot.show()




# Gete accuracy for set of testing data and generate ROC curve
def getAccuracy(instanceList,root):
    accuracy = 0
    TP=0
    FP=0
    TN=0
    FN=0
    rocList=list()
    #print instanceList
    for instance in instanceList:
        featureData = testVal[instance]
        leafVal=getLeafValue(featureData,root)
        #print 'Actual Value',testTarget[instance],'Predicted Value',leafVal
        val = testTarget[instance]
        val = int(val)
        leafVal = int(leafVal)
        if val == leafVal:
            accuracy+=1

        print val,leafVal

        # Code for confusion matrics and plot
        if (val == leafVal) &  (leafVal== 1):
            #print 'TP'
            TP+=1
        if (val == leafVal) &  (leafVal == 0):
            #print 'TN'
            TN+=1
        if (val == 0) & (leafVal == 1):
            #print 'FP'
            FP+=1
        if (val == 1) & (leafVal == 0):
            #print 'FN'
            FN+=1

        rocList.append((TP,FP))
    accuracy = (accuracy/ float(len(instanceList)))*100
    print TP,FN,FP,TN
    generatePlot(rocList,accuracy)
    return accuracy





fp=readFile(inputFile)
dataSet,targetSet=loadDataSet(fp)
kFoldData=generateKSets(sorted(dataSet.keys()))
combinationSet=createCombination(kFoldData)



#print calcIG(combinationSet[0][1])

j=0
totalAccuracy =0

#combinationSet = [combinationSet[7]]


#Execute for all k fold combinations for training-testing points
# Generate decision tree on training data
# predict accuracy on testing data
for combination in combinationSet:
    testVal=dict()
    trainVal=dict()
    testTarget=dict()
    trainTarget=dict()
    for train in combination[0]:
        trainVal[train]= dataSet[train]
        trainTarget[train]=targetSet[train]

    for test in combination[1]:
        testVal[test] = dataSet[test]
        testTarget[test] = targetSet[test]

    root=generateTree(trainVal.keys(),"",0)
    Percentage=getAccuracy(testVal.keys(),root)
    print 'Accuracy % for combination '+str(j)+' is:',Percentage
    totalAccuracy+=Percentage
    j+=1
    #if j==1:
    #    break



# output accuracy as average over all the folds
print totalAccuracy/len(combinationSet)











