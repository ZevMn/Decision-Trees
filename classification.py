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



    def calculate_new_information_entropy(self, y_left, y_right):
        """ Calculates the entropy gain of the decision tree
        for a given splitting rule.

        Args:
            y (numpy.ndarray): Class labels, numpy array of shape (N, )
            y_left (numpy.ndarray): Class labels to the left of the split, numpy array of shape (N, )
            y_right (numpy.ndarray): Class labels to the right of the split, numpy array of shape (N, )

        Returns:
            Information entropy gain for a given splitting rule
        """

        information_entropy_left = 0
        information_entropy_right = 0

        [_, count] = np.unique(y_left, return_counts=True)
        for i in count:
            proportion = i/len(y_left)
            information_entropy_left -= proportion * math.log2(proportion)

        [_, count] = np.unique(y_right, return_counts=True)
        for i in count:
            proportion = i / len(y_right)
            information_entropy_right -= proportion * math.log2(proportion)




        information_entropy_left = - len(y_left)/len(y) * math.log2(len(y_left)/len(y))
        information_entropy_right = - len(y_right)/len(y) * math.log2(len(y_right)/len(y))

        return information_entropy_left + information_entropy_right






        
    
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
        
