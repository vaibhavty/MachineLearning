# This program applies sklearn's svm on digit images dataset
__author__ = 'vaibhavtyagi'

from pylab import *
from numpy import *
import mnist
from sklearn import svm,preprocessing

import warnings
warnings.filterwarnings('ignore')


# Load data into python lists
def loadData(text):
    xData = list()
    yData = list()
    for i in range(6):
        images, labels = mnist.load_mnist(text,digits=[i])
        total = len(labels)
        if text=="training":
            reqd = int(total * 0.2)
        else:
            reqd = total
        if i==0:
            xData = images[:reqd]
            yData = labels[:reqd]
        else:
            xData= np.concatenate((xData,images[:reqd]),axis=0)
            yData= np.concatenate((yData,labels[:reqd]),axis=0)

        #print yData
    return xData,yData



# Generate random size rectangles on images for feature
# creation using HAAR feature selection
def generateRectangleList():
    rectList = list()
    i=0
    while i<100:
        init=np.random.randint(0,27,2)
        end=np.random.randint(0,27,2)
        topX = min(init[0],end[0])
        botX = max(init[0],end[0])
        topY = min(init[1],end[1])
        botY = max(init[1],end[1])

        area = abs(topX - botX) * abs(topY - botY)
        if (area >=130) & (area <= 170):
            temp = [topX,topY,botX,botY]
            if temp not in rectList:
                rectList.append(temp)
                i+=1
        #if i==2:
        #    break
    return rectList



# Generate features from rectangle using HAAR feature selection
def generateFeatures(rectList,xData):
    fList = list()
    for image in xData:
        temp = list()
        for rect in rectList:
            #print image
            #print rect
            chunk = image[rect[0]:rect[2],rect[1]:rect[3]]
            h = (rect[2] - rect[0])/2
            v = (rect[3] - rect[1])/2

            upperSum = int(np.sum(chunk[0:,0:h]))
            lowerSum = int(np.sum(chunk[0:,h:]))

            leftSum = int(np.sum(chunk[0:v,0:]))
            rightSum = int(np.sum(chunk[v:,0:]))

            #print upperSum,lowerSum,leftSum,rightSum
            f1 =  upperSum-lowerSum
            f2 = leftSum-rightSum
            temp+=[f1,f2]

        fList.append(temp)
    return fList


# Predict accuracy on testing data
def getAccuracy(clf,xTrain,yTrain,xTest,yTest):
    clf.fit(xTrain,yTrain)
    Acc=0
    pred = clf.predict(xTest)
    i=0
    for val in pred:
        if val==yTest[i]:
            Acc+=1
        i+=1
    Acc/=float(len(yTest))
    return Acc


# Generate training features and labels
xData,yLabel = loadData('training')
yTrain = [i[0] for i in yLabel]
rectList = generateRectangleList()
xTrain = generateFeatures(rectList,xData)


#Generate testing features and labels
xData,yTest = loadData('testing')
yTest = [i[0] for i in yTest]
xTest = generateFeatures(rectList,xData)


testCount = len(xTest)
trainCount = len(xTrain)


#Normalize Data- Added
dataSet = xTrain + xTest
print 'Total Data Considered',len(dataSet)
dataSet = preprocessing.scale(dataSet)
xTrain = dataSet[0:trainCount]
xTest = dataSet[trainCount:]

print 'Data for Training',len(xTrain)
print 'Data for testing',len(xTest)
#exit()


# Create class object and predict accuracy 
multiClass_clf = svm.LinearSVC()
acc= getAccuracy(multiClass_clf,xTrain,yTrain,xTest,yTest)


print 'Accuracy',acc


