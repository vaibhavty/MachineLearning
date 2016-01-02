
# This is KNN implementation
# Contains Euclidian distance, cosine distance,
# Linear ,Gaussian and polynomial simmilarity
# Implementation for k neighbours and window range base prediction


__author__ = 'vaibhavtyagi'

import numpy as np
from scipy import spatial
from sklearn import preprocessing

class knn():

    def __init__(self,k):
        self.k=k

    def Euclidian(self,pointSet,labelSet,point):
        i=0
        distList = list()
        pointSet = np.array(pointSet)
        point = np.array(point)
        for dPoint in pointSet:
            dist = np.linalg.norm(dPoint - point)
            #print dist
            distList.append((float(dist),labelSet[i]))
            i+=1
        #exit()
        return distList


    def Dot(self,pointSet,labelSet,point):
        i=0
        distList = list()
        point = np.matrix(point).T
        pointSet = np.matrix(pointSet)
        for dPoint in pointSet:
            dist = dPoint * point

            distList.append((float(dist),labelSet[i]))
            i+=1
        #print distList
        #exit()
        return distList

    def Cosine(self,pointSet,labelSet,point):
        i=0
        distList = list()
        pointSet = np.array(pointSet)
        point = np.array(point)
        for dPoint in pointSet:
            dist = spatial.distance.cosine(dPoint,point)
            #print dist
            distList.append((float(dist),labelSet[i]))
            i+=1
        #exit()
        return distList


    def Gaussian(self,pointSet,labelSet,point,gamma):
        i=0
        distList = list()
        pointSet = np.array(pointSet)
        point = np.array(point)
        for dPoint in pointSet:
            dist = np.linalg.norm(dPoint - point)
            dist = -1 * gamma * (dist**2)
            dist = np.exp(dist)
            distList.append((float(dist),labelSet[i]))
            i+=1
        #exit()
        return distList



    def Polynomial(self,pointSet,labelSet,point):
        i=0
        distList = list()
        pointSet = np.matrix(pointSet)
        point = np.matrix(point)
        for dPoint in pointSet:
            dist = (dPoint *point.T) * 0.002
            dist =  (30 + dist)
            distList.append((float(dist),labelSet[i]))
            i+=1
        #print distList
        #exit()
        #print distList
        #exit()
        return distList


    def returnTopK(self,distList,model):
        predList = list()
        for val in self.k:
            if (model=='Cosine') | (model =='Eucl'):
                temp = sorted(distList, key=lambda tup: tup[0])[0:val]
            else:
                #print 'here'
                temp = sorted(distList, key=lambda tup: tup[0],reverse=True)[0:val]
                #print temp
                #exit()
            labelList = [i[1] for i in temp]
            predList.append(max(set(labelList), key=labelList.count))
        #print predList
        return predList


    def returnWindow(self,dataList,R):
        predList = list()
        distList = [i[0] for i in  dataList]
        labelList = [i[1] for i in dataList]
        distList = preprocessing.scale(distList)
        i=0
        #print distList
        for dist in distList:
            if dist <= R:
                predList.append(labelList[i])
            i+=1
        if len(predList) >0:
            return max(set(predList), key=predList.count)
        else:
            return 'NoData'


    def getPZC_PC(self,dataList):
        dictPZC = dict()
        dictPC = dict()

        for data in dataList:
            #print data
            label = data[1]
            dist = data[0]

            try:
                dictPZC[label]+=dist
                dictPC[label]+=1
            except:
                dictPZC[label] = dist
                dictPC[label] = 1

        total=0
        #print dictPZC
        #print dictPC
        for key in dictPC:
            classCount = dictPC[key]
            dictPZC[key]/=float(classCount)
            total+=classCount
        #print dictPZC
        for key in dictPC:
            dictPC[key]/=float(total)


        return dictPZC,dictPC