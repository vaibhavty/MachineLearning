__author__ = 'vaibhavtyagi'


import numpy as np

inFile = "../perceptronData.txt"


lam=0.001


# read input file
def readFile(fileName):
    fp=open(fileName,'r')
    return fp


# load data to python lists
def loadDataSet(fp):
    dataSet = list()
    targetSet = list()
    i=1
    for line in fp.readlines():
        valList = line.split("\t")
        if len(valList) != 0:
            label = int(valList[-1])
            dataSet.append([float(j) for j in valList[:-1]])
            targetSet.append([label])
            i+=1
    return dataSet,targetSet



# run peceptron algorithm until it converges
# it will not converge unless data is not linearly seprable
# algorithm updates the weights for the misclassifies points in each iteration
def applyPerceptron(X,Y,W):
    global lam
    thresh=0
    X=np.c_[X,np.ones(len(Y))]
    itr = 0
    while True:
        i=0
        errorCount=0
        for dPoint in X:
            s = np.dot(dPoint,W)
            #print s[0]
            #print Y[i][0]
            if s[0] > thresh:
                n = 1
            else:
                n = -1
            e= Y[i][0] - n


            if n!= Y[i][0]:
                errorCount+=1


            d = lam * e
            diff = np.dot(dPoint,d)
            k=0
            for data in diff:
                W[k][0] += data
                k+=1
            i+=1
            #print W
            #exit()
        print 'In Iteration',itr,'Total Mistake',errorCount
        if errorCount==0:
            break
        itr+=1
    W= np.array(W)
    output= list()
    print W
    for data in W:
        output.append(data/float(-1*float(W[-1])))
    print output[:-1]


fp=readFile(inFile)
X,Y=loadDataSet(fp)
W=np.zeros((len(X[0])+1,1))

#X=[[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]]
#Y=[[1],[1],[1],[0]]
#W=np.zeros((len(X[0]),1))
#print X[0],Y[0]
#print W

applyPerceptron(X,Y,W)