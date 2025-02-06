##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from kfold import kfold

class improvement:
    def __init__(self, x, y, n_folds=10):
        self.x = x
        self.y = y
        self.n_folds = n_folds
        return

    def train_and_predict(self, x_train, y_train, x_test, x_val, y_val):
        """ Interface to train and test the new/improved decision tree.

        This function is an interface for training and testing the new/improved
        decision tree classifier.

        x_train and y_train should be used to train your classifier, while
        x_test should be used to test your classifier.
        x_val and y_val may optionally be used as the validation dataset.
        You can just ignore x_val and y_val if you do not need a validation dataset.

        Args:
        x_train (numpy.ndarray): Training instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of attributes
        y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str
        x_test (numpy.ndarray): Test instances, numpy array of shape (M, K)
                                M is the number of test instances
                                K is the number of attributes
        x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K)
                           L is the number of validation instances
                           K is the number of attributes
        y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )

        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
        """

        (best_max_depth,
         best_min_sample_split,
         best_min_impurity_decrease,
         gridsearch_optimised_tree) = self.grid_search(x_train, y_train)

        return gridsearch_optimised_tree.predict(x_test)

    def grid_search(self, x, y, n_folds=10, random_generator=np.random.default_rng()):

        accuracies = np.zeros((n_folds,))

        best_max_depth = 0
        best_min_sample_split = 0
        best_min_impurity_decrease = 0

        # Iterate through each set of folds
        for i, (train_indices, val_indices, test_indices) in enumerate(
                self.train_val_test_k_fold(len(x), n_folds, random_generator)):
            # Set up the dataset for the current fold
            x_train = x[train_indices, :]
            y_train = y[train_indices]
            x_val = x[val_indices, :]
            y_val = y[val_indices]
            x_test = x[test_indices, :]
            y_test = y[test_indices]

            # Perform grid search, i.e.
            # evaluate DecisionTreeClassifier for max_depth, min_sample_split, min_impurity_decrease
            # and store the accuracy and classifier for each max_depth

            # Grid search for max_depth
            gridsearch_accuracies = []
            for max_depth in [None, 5, 10, 15, 20]:  # Avoid using None
                decision_tree_classifier = DecisionTreeClassifier(max_depth=max_depth)
                decision_tree_classifier.fit(x_train, y_train)
                predictions = decision_tree_classifier.predict(x_val)
                current_accuracy = np.mean(predictions == y_val)
                gridsearch_accuracies.append((current_accuracy, max_depth, decision_tree_classifier))

            # Select the classifier with the highest accuracy
            # NB: key=lambda x:x[0] sorts the list by the first tuple element (the accuracy)
            (best_accuracy, best_max_depth, best_classifier) = max(gridsearch_accuracies, key=lambda param: param[0])

            # Grid search for min_sample_split
            gridsearch_accuracies = []
            for min_sample_split in np.arange(1, 50, 2):
                decision_tree_classifier = DecisionTreeClassifier(max_depth=best_max_depth,
                                                                  min_sample_split=min_sample_split)
                decision_tree_classifier.fit(x_train, y_train)
                predictions = decision_tree_classifier.predict(x_val)
                current_accuracy = np.mean(predictions == y_val)
                gridsearch_accuracies.append((current_accuracy, min_sample_split, decision_tree_classifier))

            # Select the classifier with the highest accuracy
            # NB: key=lambda x:x[0] sorts the list by the first tuple element (the accuracy)
            (best_accuracy, best_min_sample_split, best_classifier) = max(gridsearch_accuracies, key=lambda param: param[0])

            # Grid search for min_impurity_decrease
            gridsearch_accuracies = []
            for min_impurity_decrease in reversed(np.arange(0.01, 0.5, 0.1)):  # Start from 0.01
                decision_tree_classifier = DecisionTreeClassifier(
                    max_depth=best_max_depth,
                    min_sample_split=best_min_sample_split,
                    min_impurity_decrease=min_impurity_decrease
                )
                decision_tree_classifier.fit(x_train, y_train)
                predictions = decision_tree_classifier.predict(x_val)
                current_accuracy = np.mean(predictions == y_val)
                gridsearch_accuracies.append((current_accuracy, min_impurity_decrease, decision_tree_classifier))

            # Select the classifier with the highest accuracy
            # NB: key=lambda x:x[0] sorts the list by the first tuple element (the accuracy)
            (best_accuracy, best_min_impurity_decrease, best_classifier) = max(gridsearch_accuracies,
                                                                          key=lambda param: param[0])

            print("\nBest accuracy for current fold: ", best_accuracy)
            print("Best max_depth: ", best_max_depth)
            print("Best min_sample_split: ", best_min_sample_split)
            print("Best min_impurity_decrease: ", best_min_impurity_decrease)

            # Finally, evaluate this classifier on x_test
            predictions = best_classifier.predict(x_test)
            final_accuracy = np.mean(predictions == y_test)
            accuracies[i] = final_accuracy

        print("Final accuracies: ", accuracies)
        print("Mean: ", np.mean(accuracies))
        print("Standard deviation: ", np.std(accuracies))

        return best_max_depth, best_min_sample_split, best_min_impurity_decrease, best_classifier

    def train_val_test_k_fold(self, n_instances, n_folds=10, random_generator=np.random.default_rng()):

        # Split the dataset into k splits of indices
        kfold_object = kfold()
        split_indices = kfold_object.k_fold_split(n_instances, n_folds, random_generator=random_generator)

        folds = []
        # Iterate through the folds each time selecting one as the test set and the rest for training
        for k in range(n_folds):
            # Select k as the test set, and k+1 as validation (or 0 if k is the final split)
            test_indices = split_indices[k]
            val_indices = split_indices[(k + 1) % n_folds]

            # Concatenate remaining folds for training
            train_indices = np.zeros((0,), dtype=int)
            for i in range(n_folds):
                if i not in [k, (k + 1) % n_folds]: # Concatenate to training set if not validation or test
                    train_indices = np.hstack([train_indices, split_indices[i]])

            folds.append([train_indices, val_indices, test_indices])

        return folds