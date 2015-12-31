# This is a sample autoencoder with 8 inputs and 8 outputs and 1 size 3 hidden layers


#from __future__ import division
import numpy as np
import math


# Threshold for MSE at output layer
thresh = 0.001
lam = 0.4



# Neural network class for storing layers and connection information
class NeuralNetwork():
    W = list()
    netList = []
    outputList = []


    # Initialize weights for connection between input and hidden layer
    # & connection between hidden to output layer to random values
    # we have taken 1 additional feature as bias at input and hidden layer extra
    # we can create any number of layers depending upon requirement on simmilar fashion
    def __init__(self):
        self.W.append(np.random.normal(scale=0.1, size=(3, 9)))
        self.W.append(np.random.normal(scale=0.1, size=(8, 4)))


    # We are using sigmoid as activation function
    # Calculate output at input layer, then calculate input for hidden layer using weights
    # calculate output for hidden layer using activation function and return output list
    def forwardPass(self, X):
        self.netList = []
        self.outputList = []
        rows, cols = X.shape
        
        # Generate hidden layer data
        net = self.W[0].dot(np.c_[X, np.ones(rows)].T)
        out = self.calcSigmoid(net)
        self.netList.append(net)
        self.outputList.append(out)

        # Generate output data
        net = self.W[1].dot(np.c_[self.outputList[-1].T, np.ones(rows)].T)
        out = self.calcSigmoid(net)
        self.netList.append(net)
        self.outputList.append(out)

        # return output for final execution
        return self.outputList[-1].T


    # Based on the error on the output layer between predicted and actual labels
    # update weights for all the connections in the network using back propagation
    def backPass(self, X, Y):
        global lam
        currVal = []
        rows, cols = X.shape

        # calculate SE for the output
        SE = self.outputList[1] - Y.T
        # calculate SSE for the output
        SSE = np.sum(SE**2)
        #print SSE

        # Calculate common component
        part1 = SE * self.calcDifferential(self.netList[1])
        currVal.append(part1)
        part2 = self.W[1].T.dot(currVal[-1])
        currVal.append(part2[:-1, :] * self.calcDifferential(self.netList[0]))


        # Update weights at hidden
        part1 = np.c_[X, np.ones(rows)].T
        part2 = part1[None, :, :].transpose(2, 0, 1)
        part3 = currVal[1][None, :, :].transpose(2, 1, 0)
        eTotal = np.sum(part2 * part3,axis=0)
        self.W[0] -= lam * eTotal

        # update weights at input
        part1 = np.c_[self.outputList[0].T, np.ones(self.outputList[0].shape[1])].T
        part2 = part1[None, :, :].transpose(2, 0, 1)
        part3 = currVal[0][None, :, :].transpose(2, 1, 0)
        eTotal = np.sum(part2 * part3,axis=0)
        self.W[1] -= lam * eTotal


        return SSE


    # Returns the value of the activation function
    def calcSigmoid(self,val):
        val = 1 / (1 + np.exp(-val))
        return val

    def calcDifferential(self,val):
        val = self.calcSigmoid(val)
        val = val * (1 - val)
        return val




def convertToBinary(X):
    out = list()
    for row in X:
        rowList = list()
        for data in row:
            if data > 0.5:
               temp = 1
            else:
                temp = 0
            rowList.append(temp)
        out.append(rowList)
    return out


#Inputs
obj = NeuralNetwork()
X = np.eye(8)
Y = np.eye(8)

i=0

# Repeat until error at output is less than the threshold defined for expected error
while True:
    obj.forwardPass(X)
    SSE = obj.backPass(X, Y)
    if  i % 1000 == 0:
        print 'Iteration:',i,'SSE:',SSE
        print "***"
    if SSE < thresh:
        print 'Iteration:',i,'SSE:',SSE
        break
    if i== 50000:
        print "Max Number of Iteration reached"
        break
    i+=1


print "***"
print "***"
print "***"
print "Input provided is :"
print X

print "***"
print "***"
# Calculate final out when loop breaks
finalOut  = obj.forwardPass(X)
print "Final raw Output :"
print finalOut

finalOut_Binary = convertToBinary(finalOut)
print "Final Binary output :"
for data in finalOut_Binary:
    print data

print "***"
print "***"
# Get values on hidden layer
Whi = obj.W[1][:, :-1]
Whi = obj.calcSigmoid(Whi)
print "Final raw at Hidden"
print Whi
Whi_Binary  = convertToBinary(Whi)
print "Final Binary at Hidden"
for data in  Whi_Binary:
    print data

