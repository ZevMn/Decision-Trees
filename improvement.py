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
from kfold import k_fold_split, majority_vote


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

    best_accuracy, best_combination, classifiers = grid_search(x_train, y_train, n_folds)

    print("Best accuracy:", best_accuracy)
    print("Best combination:", best_combination)

    # Model 1: Training a decision tree using best parameters
    improved_tree = DecisionTreeClassifier()
    improved_tree.fit(x_train,
                      y_train,
                      max_depth=best_combination[0],
                      min_sample_split=best_combination[1],
                      min_impurity_decrease=best_combination[2]
                      )
    single_tree_predictions = improved_tree.predict(x_test)


    # Model 2: Using 10 trees to do majority vote
    predictions = []
    for tree in classifiers:
        predictions.append(tree.predict(x_test))
    predictions = np.array(predictions)
    majority_predictions = majority_vote(predictions)


    return single_tree_predictions, majority_predictions


def grid_search(x, y, n_folds=10, random_generator=np.random.default_rng(42)):
    """
    Perform grid search to evaluate DecisionTreeClassifier for many possible combinations of
    max_depth, min_sample_split, and min_impurity_decrease, using k-fold cross-validation.

    Args:
        x (np.array): Feature matrix.
        y (np.array): Target vector.
        n_folds (int): Number of folds for k-fold cross-validation.
        random_generator: Random number generator for consistent splits.

    Returns:
        best_accuracy (float): Best average accuracy across folds.
        best_combination (tuple): Best hyperparameter combination.
        best_classifiers (list): List of 10 trained DecisionTreeClassifiers using the best parameters.
    """
    # Define the hyperparameter ranges to test
    max_depths = [None, 3, 6, 9, 12]
    min_sample_splits = range(1, 3)
    min_impurity_decreases = np.linspace(0.1, 0.5, num=5)
    param_combinations = cartesian_product_matrix(max_depths, min_sample_splits, min_impurity_decreases)

    # Grid search for all combinations of parameters
    gridsearch_results = []

    # Iterate through all combinations of hyperparameters
    for combination in param_combinations:
        fold_accuracies = []  # Store accuracy for each fold
        fold_classifiers = []  # Store trained models for each fold

        # Perform k-fold cross-validation
        for train_indices, val_indices, test_indices in train_val_test_k_fold(len(x), n_folds, random_generator):
            # Initialise the decision tree with current hyperparameters
            decision_tree_classifier = DecisionTreeClassifier(
                max_depth=combination[0],
                min_sample_split=combination[1],
                min_impurity_decrease=combination[2]
            )

            # Set up the dataset for the current fold
            x_train = x[train_indices, :]
            y_train = y[train_indices]
            x_val = x[val_indices, :]
            y_val = y[val_indices]

            # Train the decision tree on the training folds
            decision_tree_classifier.fit(x_train, y_train)

            # Predict labels on the validation fold
            predictions = decision_tree_classifier.predict(x_val)

            # Compute accuracy for the current fold
            current_accuracy = np.mean(predictions == y_val)
            fold_accuracies.append(current_accuracy)
            fold_classifiers.append(decision_tree_classifier)

        # Compute the average accuracy across all folds
        average_accuracy = np.mean(fold_accuracies)

        # Store the results for this parameter combination
        gridsearch_results.append((average_accuracy, combination, fold_classifiers))

    # Select the best parameter combination based on highest average accuracy
    best_accuracy, best_combination, best_classifiers = max(gridsearch_results, key=lambda param: param[0])

    print("\nBest average accuracy: ", best_accuracy)
    print("Best max_depth: ", best_combination[0])
    print("Best min_sample_split: ", best_combination[1])
    print("Best min_impurity_decrease: ", best_combination[2])

    return best_accuracy, best_combination, best_classifiers



def train_val_test_k_fold(n_instances, n_folds=10, random_generator=np.random.default_rng(42)):

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