# This program applies custom MySMO class implementation on digits dataset
__author__ = 'vaibhavtyagi'

from pylab import *
from numpy import *
import mnist
from sklearn import svm,preprocessing
import MySMO
import cPickle


import warnings
warnings.filterwarnings('ignore')


# Load data to python lists
def loadData(text):
    xData = list()
    yData = list()
    for i in range(10):
        images, labels = mnist.load_mnist(text,digits=[i])
        total = len(labels)
        if text=="training":
            reqd = int(total * 0.2)
        else:
            reqd = total * 0.2
        if i==0:
            xData = images[:reqd]
            yData = labels[:reqd]
        else:
            xData= np.concatenate((xData,images[:reqd]),axis=0)
            yData= np.concatenate((yData,labels[:reqd]),axis=0)

        #if text=="training":
            #print yData
            #dataset = np.c_(xData,yData)
            #np.ndarray.dump(dataset, "datasetter/train_data/haar_features_smo_"+i+".mat")

    #if text = "training":
        #dataset = np.c_(xData,yData)
        #np.ndarray.dump(dataset, "datasetter/haar_features_fi_test_smo.mat")
    return xData,yData



# Generate random rectangles from image matrix for feature creation
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


# Genereate fearures from rectangle using HAAR feature selection
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



# Returns output of SVM function on a testing data point
def runSVMFunction(alpha,b,X,Y,dPoint):
    val = 0
    i=0
    dPoint = dPoint.T

    for point in X:
        part1 = alpha[i] * Y[i]
        part2 = X[i] * dPoint
        out = part1 * part2
        val+=out
        i+=1
    return (val + b)



# Predict accuracy on testing data points
def getAcc(alpha,b,X,Y,xTest,yTest):
    acc=0
    i=0
    for val in yTest:
        predList=list()
        for label in range(10):
            pred = runSVMFunction(alpha[label],b[label],X,Y,np.matrix(xTest[i]))

            #print "******"
            #print 'Predicted Value',pred
            predList.append(pred)
        pred = predList.index(max(predList))
        print 'Predicted Label',pred,'Actual Label',val

        if pred == val:
            acc+=1
        i+=1
    return acc/float(len(yTest))


def updateLabelList(Y,label):
    labels = [1 if i==label else -1 for i in Y]
    return labels



# Generate training features and labels
xData,yLabel = loadData('training')
yTrain = np.array([i[0] for i in yLabel])
rectList = generateRectangleList()
xTrain = generateFeatures(rectList,xData)


#Generate testing features and labels
xData,yTest = loadData('testing')
yTest = np.array([i[0] for i in yTest])
xTest = generateFeatures(rectList,xData)


testCount = len(xTest)
trainCount = len(xTrain)

dataSet = xTrain + xTest
print 'Total Data Considered',len(dataSet)
dataSet = preprocessing.scale(dataSet)
xTrain = np.array(dataSet[0:trainCount])
xTest = np.array(dataSet[trainCount:])

print 'Data for Training',len(xTrain)
print 'Data for testing',len(xTest)



# Run SMO algorithm using MySMO class implementation
alphaList = list()
bList = list()
models = list()
for label in range(10):
    labels = np.array(updateLabelList(yTrain,label))
    Obj = MySMO.MySMO()
    b,alpha = Obj.smo_simple(xTrain, labels, 0.75, 0.0001, 30)
    print alpha,b
    alphaList.append(alpha)
    bList.append(b)
    models.append((label,b,alpha))

#cPickle.dump(models, open("datasetter/o_v_r_models.pkl", "wb"))


Acc = getAcc(alphaList,bList,xTrain,yTrain,xTest,yTest)
print Acc