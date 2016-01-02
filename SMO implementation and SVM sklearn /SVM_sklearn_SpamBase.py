# Spambase dataset classification using sklearn's SVM library
__author__ = 'vaibhavtyagi'


import numpy as np
from sklearn import svm,preprocessing,metrics

inputFile = '../Dataset2/spambase.data'
dataSet = list()
k=2


# Load data into python list
def readFile(file):
    dataSet = np.loadtxt(file,dtype=float,delimiter=",")
    xList = list()
    yList = list()
    feature_data = preprocessing.scale(dataSet[:, range(dataSet.shape[1] - 1)])
    dataSet = np.c_[feature_data, dataSet[:, -1]]
    np.random.shuffle(dataSet)
    for data in dataSet:
        xList.append([j for j in data[0:-1]])
        yList.append(data[-1])

    return xList,yList


# Predict accuracy on dataset
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




dataSet,targetSet=readFile(inputFile)


total = len(targetSet)
reqd = int(total * 0.80)


xTrain = dataSet[0:reqd]
yTrain = targetSet[0:reqd]
xTest = dataSet[reqd:]
yTest = targetSet[reqd:]

Y=np.array(yTrain)
#Y=Y.T
X=np.array(xTrain)
#print Y
#exit()

# Create class object for polynomial, RBF and linear kernals
clf_poly = svm.SVC(C=2.0, coef0=1.0, kernel='poly', degree=2.0, verbose=False)
clf_rbf = svm.SVC(C=2.0, kernel='rbf', verbose=False)
clf_linear = svm.SVC(kernel=metrics.pairwise.linear_kernel, verbose=False)


# Output accuracy for all 3 kernels 
Acc = getAccuracy(clf_poly,xTrain,yTrain,xTest,yTest)
print 'Polynomial Kernal:',Acc

Acc = getAccuracy(clf_rbf,xTrain,yTrain,xTest,yTest)
print 'RBF Kernal:',Acc

Acc = getAccuracy(clf_linear,xTrain,yTrain,xTest,yTest)
print 'Linear Kernal:',Acc




















