__author__ = 'vaibhavtyagi'
import TreeNode
import math
import numpy as np

from collections import Counter
import pylab as plot
from random import shuffle

inputFile = '../Dataset2/spambase.data'
dataSet = list()

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




trainVal = list()
testVal = list()

trainTarget = list()
testTarget = list()
k=10



# read datafile
def readFile(fileName):
    fp=open(fileName,'r')
    return fp


# load data to python list
def loadDataSet(fp):
    dataSet = list()
    targetSet = list()
    i=1
    for line in fp.readlines():
        valList = line.split(",")
        if len(valList) != 0:

            dataSet.append(valList)
            #dataSet.append([float(j) for j in valList[:-1]])
            #targetSet.append([float(valList[-1])])
            #i+=1

    shuffle(dataSet)
    i=0
    for data in dataSet:
        dataSet[i] = [float(j) for j in data[:-1]]
        targetSet.append([float(data[-1])])
        i+=1
    return dataSet,targetSet


# Normalize input data using shift and scale
def normalizeData():
    global dataSet
    i=0
    for feature in featureList:
        allVal = list()

        for val in dataSet:
            allVal.append(val[i])

        minVal = min(allVal)
        #print minVal

        j=0
        for val in dataSet:
            dataSet[j][i] -=  minVal
            j+1

        allVal = list()

        for val in dataSet:
            allVal.append(val[i])


        maxVal = max(allVal)
        j=0
        for val in dataSet:
            dataSet[j][i] = dataSet[j][i]/float(maxVal)
            j+=1

        i+=1


# Generate training and testing sets using k-fold validation
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




# Create k training and testing combinations
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




#Calculate weights for training data
def calculateW(X,Y):
    X=np.matrix(X)
    X=np.c_[X,np.ones(len(Y))]
    Y=np.matrix(Y)
    XT=X.transpose()
    val = XT * X
    #print val[0]
    val = val.I
    val = val * XT
    val = val * Y
    return X,val



# Calculate test labels based on W from traing data
def getTestLabels(X,W):
    X=np.c_[X,np.ones(len(X))]
    X=np.matrix(X)
    W=np.matrix(W)
    val = X*W
    return val



# Generate ROC plot for the output
def generatePlot(rocList,accuracy):
    Y = [i for (i,j) in rocList]
    X = [j for (i,j) in rocList]
    AUC = np.trapz(Y,X)
    plot.plot(Y,X)
    plot.xlabel('False Positive')
    plot.ylabel('True Positive')
    plot.title('ROC curve(Linear Regression) Accuracy:'+str(accuracy)+' AUC:'+str(AUC))
    plot.grid(True)
    plot.savefig("test.png")
    plot.show()






# Calculate error in prediction using mean square error approach
def calcMSE(yPred,testTarget):
    ACC=0
    SE=0
    tLen=len(testTarget)
    yPred = np.array(yPred)
    TP=0
    FP=0
    TN=0
    FN=0
    rocList=list()
    for i in range(len(testTarget)):
        if yPred[i][0] <= 0.5:
            pred = 0
        else:
            pred = 1
        if pred == testTarget[i][0]:
            SE+=1



        # For plot
        val = testTarget[i][0]
        leafVal = pred

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


        #val = testTarget[i][0] - yPred[i][0]
        #val = math.pow(val,2)
        #SE+=val

    ACC = (SE/float(tLen))*100
    print TP,FN,FP,TN
    generatePlot(rocList,ACC)
    return ACC





fp=readFile(inputFile)
dataSet,targetSet=loadDataSet(fp)
normalizeData()

kFoldData,kFoldLabel=generateKSets(dataSet,targetSet)
combinationSet=createCombination(kFoldData,kFoldLabel)
total=0
i=0
combinationSet = [combinationSet[5]]


# Run for all the k folds
for combination in combinationSet:
    X,W=calculateW(combination[0],combination[1])

    yPred = getTestLabels(combination[2],W)
    MSE= calcMSE(yPred,combination[3])
    #yPred = getTestLabels(combination[0],W)
    #MSE = calcMSE(yPred,combination[1])

    print 'ACC for combination',i,'is :',MSE
    total+=MSE
    i+=1


#Average error over all the folds
print total/float(10)