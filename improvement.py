##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import itertools

from classification import DecisionTreeClassifier
from kfold import k_fold_split


def train_and_predict(x_train, y_train, x_test, x_val, y_val, n_folds=10):
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

    (best_accuracy,
     best_combination,
     gridsearch_optimised_tree) = grid_search(x_train, y_train, n_folds)

    improved_tree = DecisionTreeClassifier()
    improved_tree.fit(x_train,
                      y_train,
                      max_depth=best_combination[0],
                      min_sample_split=best_combination[1],
                      min_impurity_decrease=best_combination[2]
                      )

    return improved_tree.predict(x_test)

def grid_search(x, y, n_folds=10, random_generator=np.random.default_rng()):

    # Perform grid search, i.e.
    # evaluate DecisionTreeClassifier for many possible combinations of
    # max_depth, min_sample_split, min_impurity_decrease

    # Define the hyperparameter ranges to test
    max_depths = [None, 5, 10]
    min_sample_splits = range(1, 3)
    min_impurity_decreases = np.linspace(0.1, 0.05, num=2)
    param_combinations = cartesian_product_matrix(max_depths, min_sample_splits, min_impurity_decreases)

    # Grid search for all combinations of parameters
    gridsearch_accuracies = []

    # Iterate through all combinations of hyperparameters
    for combination in param_combinations:
        # Initialise the decision tree with current hyperparameters
        decision_tree_classifier = DecisionTreeClassifier(
            max_depth=combination[0],
            min_sample_split=combination[1],
            min_impurity_decrease=combination[2]
        )

        # Iterate through each set of folds- Perform k-fold cross validation
        for i, (train_indices, val_indices, test_indices) in enumerate(
                train_val_test_k_fold(len(x), n_folds, random_generator)):

            # Set up the dataset for the current fold
            x_train = x[train_indices, :]
            y_train = y[train_indices]
            x_val = x[val_indices, :]
            y_val = y[val_indices]
            x_test = x[test_indices, :]
            y_test = y[test_indices]

            # Train the decision tree on the training set
            decision_tree_classifier.fit(x_train, y_train)

            # Predict labels on the validation set
            predictions = decision_tree_classifier.predict(x_val)

            # Compute the accuracy for the current fold
            current_accuracy = np.mean(predictions == y_val)

            # Store the current accuracy along with the parameter combination
            gridsearch_accuracies.append((current_accuracy, combination, decision_tree_classifier))

    # Select the classifier with the highest accuracy
    # NB: key=lambda x:x[0] sorts the list by the first tuple element (the accuracy)
    (best_accuracy, best_combination, best_classifier) = max(gridsearch_accuracies,
                                                                       key=lambda param: param[0])

    print("\nBest accuracy: ", best_accuracy)
    print("Best max_depth: ", best_combination[0])
    print("Best min_sample_split: ", best_combination[1])
    print("Best min_impurity_decrease: ", best_combination[2])

    return best_accuracy, best_combination, best_classifier

def train_val_test_k_fold(n_instances, n_folds=10, random_generator=np.random.default_rng()):

    # Split the dataset into k splits of indices
    split_indices = k_fold_split(n_instances, n_folds, random_generator=random_generator)

    folds = []
    # Iterate through the folds each time selecting one as the test set and the rest for training
    for k in range(n_folds):
        # Select k as the test set, and k+1 as validation (or 0 if k is the final split)
        test_indices = split_indices[k]
        val_indices = split_indices[(k + 1) % n_folds]

        # Concatenate remaining folds for training
        train_indices = np.zeros((0,), dtype=int)
        for i in range(n_folds):
            # Concatenate to training set if not validation or test
            if i not in [k, (k + 1) % n_folds]:
                # Horizontally stack the arrays
                train_indices = np.hstack([train_indices, split_indices[i]])

        folds.append([train_indices, val_indices, test_indices])

    return folds


def cartesian_product_matrix(list1, list2, list3):
    """
    Returns a matrix where each row represents an element of the Cartesian product
    of the three input lists.

    Parameters:
    - list1, list2, list3: Input lists.

    Returns:
    - A NumPy array where each row is a combination from the Cartesian product.
    """
    product = list(itertools.product(list1, list2, list3))
    return np.array(product)