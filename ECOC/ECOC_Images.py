# ECOC with adaptive boosting to predict image labels in digits dataset
# Generate ECOC code table based on number of classes and number of functions(20 here)
# For every function
#   update training output lables with the class and function mapping
#   run adaptive boosting to generate set of weak classifiers as a function model

# For prediction run over all the function models and generate an output code for a datapoint
# Predict the class whose ECOC code is closest to the output code generated


__author__ = 'vaibhavtyagi'

from pylab import *
from numpy import *
import mnist
import DecisionStumps
from random import randint,sample


# Load data from digits dataset using minst library
def loadData(text):
    xData = list()
    yData = list()
    for i in range(10):
        images, labels = mnist.load_mnist(text,digits=[i])
        total = len(labels)
        if text=="training":
            reqd = int(total * 0.1)
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



# Randomly generate a feature as rectangle with area between 130-170
# The rectangle should be unique in the list i.e not being generated twice
def generateRectangleList():
    rectList = list()
    i=0
    while i<50:
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



# Use HAAR feature selection to generate features from rectangle
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



# Genereate ECOC code table based on number of classes and number of functions
def generateErrorCodes(classCount,numFunctions,numFeatures):
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

    np.random.shuffle(errCode)
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
    index = randint(0,featureLen-1)
    vals = X[:,index]
    #print X[:,index]
    val = sample(vals,1)[0]
    #print Y
    err= getBestThresh(zip(vals,W,Y),val)
    #print 'Local Error:',err
    return (index,val,err)




# Update weights over all data points based on decison stump selected
def updateWeight(tpl,X,Y,W):
    X = np.array(X)
    #print tpl
    feature = tpl[0]
    thresh = tpl[1]
    error = tpl[2]
    if error == 0:
        return W,0,True
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
            W[i] *= temp
            #if prev > W[i]:
            #    print "Error"
        else:
            temp = np.exp(-1 * alpha)
            W[i] *= temp
            #if prev < W[i]:
            #    print "Error"

        i+=1


    #Normalize W
    total = np.sum(W)
    W /= float(total)
    return W,alpha,False




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
    #print codeList
    for row in errCode:
        i=0
        temp = 0
        for item in row:
            temp += abs(item - codeList[i])
            i+=1
        tempList.append(temp)
    #print tempList
    return tempList.index(min(tempList))



# Calculate total error in prediction
def getError(dataSet,labelList,allHList):
    i =0
    Acc=0
    for dPoint in dataSet:
        codeList = list()
        for row in allHList:
            code = getCode(dPoint,row,0)
            codeList.append(code)

        cls = getNearestClass(codeList)
        #print cls,labelList[i]
        if cls == labelList[i]:
            Acc+=1
        i+=1
        #if i==5:

    Acc = (Acc /float(len(dataSet))) * 100
    return Acc


xData,yLabel = loadData('training')
yLabel = [i[0] for i in yLabel]


rectList = generateRectangleList()

xTrain = generateFeatures(rectList,xData)



classes = sorted(set(yLabel))
classCount = len(classes)
numFunctions = 50
numFeatures = len(xTrain[0])

errCode = generateErrorCodes(classCount,numFunctions,numFeatures)
print errCode

#print classes
#exit()


# Run over all the functions
allHList = list()
for func in range(numFunctions):
    #print yLabel
    temp =  list(yLabel)
    print 'Function is:',func
    errColumn = errCode[:,func]
    yTrain = updateDataLabel(temp,errColumn)

    #Generate initial weights as equal over all sets
    weight = np.empty(len(xTrain))
    a = 1/float(len(xTrain))
    weight.fill(a)

    #Calculate best H(x) and update weights
    hList = list()
    itr=0
    while True:
        #print sum(weight)
        tpl = getDecisionStump(xTrain,yTrain,weight)

        weight,alpha,Flag = updateWeight(tpl,xTrain,yTrain,weight)

        if Flag:
            continue
        obj = DecisionStumps.DecisionStumps(tpl[0],tpl[1],alpha)
        hList.append(obj)
        #print itr
        if itr== 3000:
            break
        itr+=1

    allHList.append(hList)







xData,yTest = loadData('testing')
yTest = [i[0] for i in yTest]
xTest = generateFeatures(rectList,xData)



Acc = getError(xTest,yTest,allHList)

print 'Acc is:',Acc








