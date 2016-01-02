# This progam applies KNN on digits dataset using custom KNN class implementation
# It uses Polynomial, cosine and Gaussian models
__author__ = 'vaibhavtyagi'


import mnist
import KNN
import numpy as np



# Load data into python lists
def loadData(text):
    xData = list()
    yData = list()
    for i in range(10):
        images, labels = mnist.load_mnist(text,digits=[i])
        total = len(labels)
        if text=="training":
            reqd = int(total * 0.5)
        else:
            reqd = int(total * 0.01)
        if i==0:
            xData = images[:reqd]
            yData = labels[:reqd]
        else:
            xData= np.concatenate((xData,images[:reqd]),axis=0)
            yData= np.concatenate((yData,labels[:reqd]),axis=0)

        #print yData
    return xData,yData



# Generate random rectangles sizes for feature creation on an image
def generateRectangleList():
    rectList = list()
    i=0
    while i<99:
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



# Generate features from rectangles using HAAR feature selection
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



# Predict accuracy on the testing data for K neighbours
def getAccuracy(k,xTrain,yTrain,X,Y,model):

    Acc=[0]*len(k)
    i=0
    obj = KNN.knn(k)
    for point in X:
        #print i
        if model == 'Cosine':
            dataList = obj.Cosine(xTrain,yTrain,point)
        if model == 'Poly':
            dataList = obj.Polynomial(xTrain,yTrain,point)
        if model == 'Gaus':
            dataList = obj.Gaussian(xTrain,yTrain,point,10e-10)
        predList = obj.returnTopK(dataList,model)

        j=0
        for val in Acc:
            if predList[j]==Y[i]:
                Acc[j]+=1
            j+=1
        i+=1
    j=0
    for val in Acc:
        Acc[j]/=float(len(Y))
        j+=1
    return Acc



# Predict accuracy for window range approach
def getAccuracyWindow(xTrain,yTrain,X,Y):
    k=0
    Acc=0
    i=0
    oldPred=0
    obj = KNN.knn(k)
    for point in X:
        print i
        dataList = obj.Cosine(xTrain,yTrain,point)
        pred = obj.returnWindow(dataList,-1.5)
        if pred=='NoData':
            pred = oldPred

        if pred == Y[i]:
            Acc+=1
        i+=1
        oldPred = pred
    Acc/=float(len(Y))

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


k=[1,3,7]
model = 'Cosine'
#model = 'Poly'
#model = 'Gaus'
#print getAccuracy(k,xTrain,yTrain,xTest,yTest,model)


print getAccuracyWindow(xTrain,yTrain,xTest,yTest)
