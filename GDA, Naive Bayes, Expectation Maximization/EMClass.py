#This is class that manage expectation and maximization steps for a gaussian model
__author__ = 'vaibhavtyagi'

import numpy as np

class EMclass():

    # Initialize Pm, mean, Z, and co variance
    def __init__(self,dataCounts,mean,var,n,nTotal):
        self.P = n/float(nTotal)
        self.U = np.array(mean)
        self.Z = np.zeros(dataCounts,)
        self.abs = np.array(var)


    # Update z values for each gaussian model for each data point
    def Estep(self,features,X):
        det = np.linalg.det(self.abs)
        invAbs = np.linalg.inv(self.abs)
        i=0
        for dPoint in X:
            part1 = 1 / float( (2 * np.pi)** (features/float(2)) * (det)**0.5)
            part2 = 0.5 * np.sum((np.subtract(dPoint,self.U))  * invAbs * np.matrix(np.subtract(dPoint,self.U)).T)
            val  = part1 * np.exp(-1 * part2)
            self.Z[i] = val * self.P
            i+=1


    # Calculate new parameters mean, Pm , co variance matrix using Z matrix from expectation step
    def Mstep(self,features,X):
        i=0
        ABS = np.zeros((features,features))
        mean= np.zeros((features,))
        for dPoint in X:
            temp = self.Z[i] * np.matrix(np.subtract(dPoint , self.U)).T \
                  *  np.matrix(np.subtract(dPoint,self.U))

            ABS = np.add(ABS,temp)
            temp = self.Z[i] * dPoint
            mean = np.add(mean,temp)
            i+=1
        total = np.sum(self.Z)
        self.abs = ABS/float(total)
        self.U = mean/float(total)
        self.P = total/float(len(X))
        return self.P