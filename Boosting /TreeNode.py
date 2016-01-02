# Data structure for storing tree's node components 

__author__ = 'vaibhavtyagi'


class TreeNode:

    def addParam(self,feature,threshold):
        self.feature=feature
        self.threshold=threshold


    def addChild(self,leftChild,rightChild):
        self.leftChild = leftChild
        self.rightChild = rightChild

    def isLeafNode(self,boll,val):
        self.isLeaf = boll
        self.val = val


