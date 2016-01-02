
# Data structure for storing details for a weak classifier
__author__ = 'vaibhavtyagi'


class DecisionStumps():

    def __init__(self,feature,threshold,alpha):
        self.feature = feature
        self.threshold = threshold
        self.alpha = alpha

