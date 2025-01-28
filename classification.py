#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import math


class Node:
    """ Represents a node in the decision tree """

    def __init__(self, label=None, feature=None, threshold=None, left=None, right=None):
        self.label = label  # Class label (int for internal use)
        self.feature = feature  # Feature index (int) or None
        self.threshold = threshold  # Threshold (float) or None
        self.left = left  # Left child Node
        self.right = right  # Right child Node


class DecisionTreeClassifier:
    def __init__(self):
        self.is_trained = False
        self.root = None
        self.label_map = {}  # Maps string labels to integers
        self.inverse_label_map = {}  # Maps integers back to strings

    def fit(self, X, y):
        unique_labels = np.unique(y)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_map = {idx: label for label, idx in self.label_map.items()}
        y_encoded = np.array([self.label_map[label] for label in y])

        self.root = self._build_tree(X, y_encoded)
        self.is_trained = True

    def _entropy(self, labels):
        """ Calculate entropy for integer-encoded labels """
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Avoid log(0)

    def _information_gain(self, y, y_left, y_right):
        """ Calculate information gain """
        p = len(y_left) / len(y)
        return self._entropy(y) - p * self._entropy(y_left) - (1 - p) * self._entropy(y_right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    print(f"Skipping invalid split: Feature {feature}, Threshold {threshold}")
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                print(f"Feature: {feature}, Threshold: {threshold}, Gain: {gain}")
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        print(f"Best split: Feature {best_feature}, Threshold {best_threshold}, Gain {best_gain}")
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        print(f"Building tree at depth {depth}, number of samples: {len(y)}")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Unique labels: {unique_labels}, Counts: {counts}")

        # Stopping criteria: pure node or single sample
        if len(unique_labels) == 1 or len(y) == 1:
            most_common_label = np.argmax(np.bincount(y))
            print(f"Creating leaf node with label: {most_common_label}")
            return Node(label=most_common_label)

        # Find best split
        feature, threshold = self._best_split(X, y)
        if feature is None:  # No valid split
            most_common_label = np.argmax(np.bincount(y))
            print(f"No valid split found. Creating leaf node with label: {most_common_label}")
            return Node(label=most_common_label)

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        x_left, y_left = X[left_mask], y[left_mask]
        x_right, y_right = X[right_mask], y[right_mask]

        # Avoid empty splits
        if len(y_left) == 0 or len(y_right) == 0:
            most_common_label = np.argmax(np.bincount(y))
            print(f"Empty split detected. Creating leaf node with label: {most_common_label}")
            return Node(label=most_common_label)

        # Recursively build subtrees
        print(f"Splitting on feature {feature} <= {threshold}")
        left = self._build_tree(x_left, y_left, depth + 1)
        right = self._build_tree(x_right, y_right, depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        encoded_preds = np.array([self._predict_sample(x, self.root) for x in X])
        return np.array([self.inverse_label_map[pred] for pred in encoded_preds])

    def _predict_sample(self, x, node):
        if node.label is not None:
            print(f"Leaf node reached with label: {node.label}")
            return node.label
        print(f"Traversing: Feature {node.feature}, Threshold {node.threshold}, Value {x[node.feature]}")
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
