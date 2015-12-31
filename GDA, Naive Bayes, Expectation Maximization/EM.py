__author__ = 'vaibhavtyagi'


import numpy as np
import EMClass


infile = '2gaussian.txt'

numModels =2
features=2



# Read input file
def readFile(infile):
    dataSet = np.loadtxt(infile,dtype=float,delimiter=" ")
    return dataSet

X=readFile(infile)
dataCounts = len(X)

"""
mean =  [3,3]
var = [[1,0],[0,3]]
n=2000
nTotal= 6000
"""
# Initialize parameter for model 1
mean =  [1,1]
var = [[1,0],[0,1]]
n=2500
nTotal= 6000


obj1 = EMClass.EMclass(dataCounts,mean,var,n,nTotal)
"""
mean =  [7,4]
var = [[1,0.5],[0.5,1]]
n=4000
nTotal= 6000
"""

# initialize parameters for model 2
mean =  [2,2]
var = [[1,0],[0,1]]
n=3500
nTotal= 6000
obj2 = EMClass.EMclass(dataCounts,mean,var,n,nTotal)


modelList = [obj1,obj2]


# normalize new updated Z values after expectation step
def normalize(modelList,dataCounts):
    i=0
    for i in range(dataCounts):
        total=0
        for model in modelList:
            total += model.Z[i]
        for model in modelList:
            model.Z[i] = model.Z[i]/float(total)



# Show output on the progress
def displayOutput(modelList):
    j=0
    for model in modelList:
        print "Model:",j+1
        print 'CoVar matrix:',model.abs
        print 'Mean matrix:',model.U
        print 'Prob Matrix:',model.P
        print 'Latent variable',model.Z
        print "*******"
        j+=1


# Run expectation maximization steps iteratively and see convergence of a datapoint to a model
j=0
while True:
    print 'Iteration is',j
    for obj in modelList:
        #print 'abc'
        obj.Estep(features,X)

    normalize(modelList,dataCounts)

    for obj in modelList:
        obj.Mstep(features,X)

    j+=1
    if j==100:
        displayOutput(modelList)
        exit()



