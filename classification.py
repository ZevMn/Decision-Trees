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

from pyexpat import features


class Node:
    """ Represents a node in the decision tree """
    def __init__(self, label=None, feature=None, threshold=None, left=None, right=None):
        self.label = label            # Class label or None (if not terminal Node)
        self.feature = feature        # Feature for splitting or None
        self.threshold = threshold    # Threshold for splitting or None
        self.left = left              # Point to another node or None
        self.right = right            # Point to another node or None

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
        self.root = None

        self.depth = 0
        self.num_nodes = 0
        self.num_leaves = 0
        self.num_classes = 1 # Is this necessary? ______________________________

        # stopping criteria:
        self.min_elements_in_subset = 1
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
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        self.root = self.build_tree(x, y)

        # set a flag so that we know that the classifier has been trained
        if self.root is not None:
            self.num_nodes += 1
            self.is_trained = True


    def entropy (self, labels):
        """ Calculates the information entropy of a given dataset

        Args:
        labels (numpy.ndarray): Labels, numpy array of shape (N, )
                                N is the number of instances

        Returns:
        The information entropy of the given dataset.
        """
        entropy = 0

        [_, count] = np.unique(labels, return_counts=True)
        for i in count:
            proportion = i / len(labels)
            entropy -= proportion * math.log2(proportion)

        return entropy


    def information_gain(self, y, y_left, y_right):
        """ Calculates the information gain of splitting a dataset into two given subsets

        Args:
        y (numpy.ndarray): Set of class labels, numpy array of shape (N, )
        y_left (numpy.ndarray): Subset of class labels, numpy array of shape (N - i, )
                                i is an integer from 0 to N - 1
        y_right (numpy.ndarray): Subset of class labels, numpy array of shape (i, )

        Returns:
        The information gain of a given splitting.
        """

        len_before = len(y)
        len_left = len(y_left)
        len_right = len(y_right)

        h_before = self.entropy(y)
        h_after = ((len_left / len_before) * self.entropy(y_left)) + ((len_right / len_before) * self.entropy(y_right))

        return h_before - h_after


    def best_split(self, x, y):
        """ Determines the (binary) splitting for a given dataset which maximises information entropy gain

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
        y (numpy.ndarray): Class labels, numpy array of shape (N, )

        Returns:
        The feature and threshold which represent the splitting with the greatest information entropy gain.
        """

        best_feature, best_threshold = None, None
        best_gain = -1.0

        n_samples, n_features = x.shape

        # Stop if splitting produces a subset of size less than predetermined minimum
        if n_samples < self.min_elements_in_subset:
            return None, None

        for feature in range(n_features):
            thresholds = np.unique(x[:, feature])

            # Loop through all possible thresholds for each feature
            # NB: This is possible because features are discrete integers from 0 to 15
            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Calculate the information entropy gain for each possible splitting
                entropy_gain = self.information_gain(y, y[left_mask], y[right_mask])
                # print(f"Evaluating feature {feature}, threshold {threshold}, gain {entropy_gain}")

                # Keep track of the best splitting
                if entropy_gain > best_gain and entropy_gain > self.min_impurity_decrease:
                    best_gain = entropy_gain
                    best_feature = feature
                    best_threshold = threshold

        # print(f"Best gain found: {best_gain}")
        return best_feature, best_threshold


    def build_tree(self, x, y, depth=0):

        # Stop if max depth is reached or labels are perfectly sorted
        n_samples = len(y)
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or (n_labels == 1) or (n_samples <= self.min_elements_in_subset):
            # Return a node with the mode class as its label
            unique_labels, y_int = np.unique(y, return_inverse=True)
            most_common_label = unique_labels[np.argmax(np.bincount(y_int))]
            # print(f"Creating leaf node at depth {depth} with label {most_common_label}")
            return Node(label=most_common_label)

        # Determine next split
        feature, threshold = self.best_split(x, y)

        # Stop if no valid split is found
        if feature is None or threshold is None:
            unique_labels, y_int = np.unique(y, return_inverse=True)
            most_common_label = unique_labels[np.argmax(np.bincount(y_int))]
            return Node(label=most_common_label)
            #return Node(label=np.argmax(np.bincount(y)))

        # Split the current dataset in the left and right subsets
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        x_left, y_left = x[left_mask], y[left_mask]
        x_right, y_right = x[right_mask], y[right_mask]


        # Create the left and right subtrees recursively
        left_subtree = self.build_tree(x_left, y_left, depth + 1)
        right_subtree = self.build_tree(x_right, y_right, depth + 1)

        # Create a new node with this split
        node = Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

        self.depth = depth
        self.num_nodes += 2
        self.num_leaves += 2

        # Return a node with the current split
        # # NB: This line is called on all non-terminal nodes once a branch has terminated
        #return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
        return node
        
    
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

        for i in range(len(x)):
            predictions[i] = self.predict_sample(x[i], self.root)

        return predictions

    def predict_sample(self, x, node):
        # If a terminal node has been reached
        if node.label is not None:
            return node.label

        # Otherwise traverse to the left or right subtree based on the splitting condition
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

class Evaluation(object):

    def __init__(self):
        pass


    def confusion_matrix(self, y_gold, y_prediction, class_labels=None):


        # if no class_labels are given, we obtain the set of unique class labels from
        # the union of the ground truth annotation and the prediction
        if not class_labels:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

        confusion = np.zeros((len(class_labels), len(class_labels)))

        # for each correct class (row),
        # compute how many instances are predicted for each class (columns)
        for (i, label) in enumerate(class_labels):
            # get predictions where the ground truth is the current class label
            indices = (y_gold == label)
            gold = y_gold[indices]
            predictions = y_prediction[indices]

            # quick way to get the counts per label
            (unique_labels, counts) = np.unique(predictions, return_counts=True)

            # convert the counts to a dictionary
            frequency_dict = dict(zip(unique_labels, counts))

            # fill up the confusion matrix for the current row
            for (j, class_label) in enumerate(class_labels):
                confusion[i, j] = frequency_dict.get(class_label, 0)

        return confusion


    def accuracy(self, y_gold, y_prediction):

        assert len(y_gold) == len(y_prediction)

        try:
            return np.sum(y_gold == y_prediction) / len(y_gold)
        except ZeroDivisionError:
            return 0.


    def precision(self, y_gold, y_prediction):

        confusion = self.confusion_matrix(y_gold, y_prediction)
        p = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[:, c]) > 0:
                p[c] = confusion[c, c] / np.sum(confusion[:, c])

        # Compute the macro-averaged precision
        macro_p = 0.
        if len(p) > 0:
            macro_p = np.mean(p)

        return (p, macro_p)


    def recall(self, y_gold, y_prediction):

        confusion = self.confusion_matrix(y_gold, y_prediction)
        r = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[c, :]) > 0:
                r[c] = confusion[c, c] / np.sum(confusion[c, :])

        # Compute the macro-averaged recall
        macro_r = 0.
        if len(r) > 0:
            macro_r = np.mean(r)

        return (r, macro_r)


    def f1_score(self, y_gold, y_prediction):

        (precisions, macro_p) = self.precision(y_gold, y_prediction)
        (recalls, macro_r) = self.recall(y_gold, y_prediction)

        # just to make sure they are of the same length
        assert len(precisions) == len(recalls)

        f = np.zeros((len(precisions),))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)

        # Compute the macro-averaged F1
        macro_f = 0.
        if len(f) > 0:
            macro_f = np.mean(f)

        return (f, macro_f)

    def evaluate(self, y_gold, y_prediction):
        confusion = self.confusion_matrix(y_gold, y_prediction)
        accuracy = self.accuracy(y_gold, y_prediction)
        precision = self.precision(y_gold, y_prediction)
        recall = self.recall(y_gold, y_prediction)
        f1_score = self.f1_score(y_gold, y_prediction)

        return confusion, accuracy, precision, recall, f1_score
