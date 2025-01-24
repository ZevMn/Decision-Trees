#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import math as math

class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self. root = None
        self.depth = 0
        self.num_nodes = 0
        self.num_leaves = 0
        self.num_classes = 0

        # stopping criteria:
        self.stop_splitting_at = 1
        self.min_impurity_decrease = 0.0001
        self.max_depth = None
    

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(x), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    
        

        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True


    def entropy (self, set_of_labels):
        entropy = 0

        [_, count] = np.unique(set_of_labels, return_counts=True)
        for i in count:
            proportion = i / len(set_of_labels)
            entropy -= proportion * math.log2(proportion)

        return entropy


    def information_gain(self, y, y_left, y_right):

        h_before = self.entropy(y_left)
        h_after = self.entropy(y_right) + self.entropy(y_left)

        return h_before - h_after


    def best_split(self, x, y):
        best_feature, best_threshold = None, None
        best_gain = 0.0

        n_samples, n_features = x.shape

        if n_samples < self.stop_splitting_at:
            return None, None

        for feature in range(n_features):
            thresholds = np.unique(x[:, feature])

            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                entropy_gain = self.information_gain(y, y[left_mask], y[right_mask])

                if entropy_gain > best_gain and entropy_gain > self.min_impurity_decrease:
                    best_gain = entropy_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def build_tree(self, x, y, depth=0):

        # Stop if max depth is reached or labels are perfectly sorted
        n_samples = len(y)
        n_labels = len(np.unique(y))
        if (self.max_depth is not None) and (depth <= self.max_depth) or n_labels == 1 or n_samples <= self.stop_splitting_at:
            return np.argmax(np.bincount(y))








        
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
    
        # remember to change this if you rename the variable
        return predictions
        
