__author__ = 'vaibhavtyagi'


class TreeNode:
    def __init__(self,instanceList,parent,feature,threshold,val):
        self.instanceList=instanceList
        self.parent=parent
        self.feature=feature
        self.threshold=threshold
        self.val = val

    def addChild(self,leftChild,rightChild):
        self.leftChild = leftChild
        self.rightChild = rightChild

    def isLeafNode(self,boll):
        self.isLeaf = boll


