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


def train_and_predict(x_train, y_train, x_test, x_val, y_val):
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

    # Perform grid search to find the best parameters
    best_params = optimise_parameters(x_train, y_train, x_val, y_val)

    # Train improved decision tree with best hyperparameters
    improved_tree = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_sample_split=best_params["min_samples_split"],
        min_impurity_decrease=best_params["min_impurity_decrease"]
    )
    improved_tree.fit(x_train, y_train)

    # Make predictions on x_test
    predictions = improved_tree.predict(x_test)
    return predictions

def train_val_test_k_fold(n_instances, n_folds=10, random_generator=np.random.default_rng()):

    # Split the dataset into k splits of indices
    split_indices = kfold.k_fold_split(n_instances, n_folds, random_generator)

    folds = []
    # Iterate through the folds each time selecting one as the test set and the rest for training
    for k in range(n_folds):
        # Select k as the test set, and k+1 as validation (or 0 if k is the final split)
        test_indices = split_indices[k]
        val_indices = split_indices[(k + 1) % n_folds]

        # Concatenate remaining folds for training
        train_indices = np.zeros((0,), dtype=np.int)
        for i in range(n_folds):
            if i not in [k, (k + 1) % n_folds]: # Concatenate to training set if not validation or test
                train_indices = np.hstack([train_indices, split_indices[i]])

        folds.append([train_indices, val_indices, test_indices])

    return folds

def grid_search(x, y, n_folds=10, random_generator=np.random.default_rng()):

    accuracies = np.zeros((n_folds,))

    # Iterate through each set of folds
    for i, (train_indices, val_indices, test_indices) in enumerate(train_val_test_k_fold(len(x), n_folds, random_generator)):
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
        gridsearch_accuracies = []


        for nn in range(1, 11):
            knn_classifier = KNNClassifier(k=nn)
            knn_classifier.fit(x_train, y_train)
            predictions = knn_classifier.predict(x_val)
            acc = accuracy(y_val, predictions)
            gridsearch_accuracies.append((acc, nn, knn_classifier))

        # Select the classifier with the highest accuracy
        # and evaluate this classifier on x_test
        # key=lambda x:x[0] sorts the list by the first tuple element (the accuracy)
        (best_acc, best_nn, best_classifier) = max(gridsearch_accuracies, key=lambda x: x[0])
        # print(gridsearch_accuracies)
        print((best_acc, best_nn))

        predictions = best_classifier.predict(x_test)
        acc = accuracy(y_test, predictions)
        accuracies[i] = acc
    return

def optimise_parameters(x_train, y_train, x_val, y_val):
    best_params = {"max_depth": None, "min_sample_split": 1, "min_samples_leaf": 1}
    best_accuracy = 0

    for max_depth in [None, 1, 5, 10]:
        acc = comp_accuracy(x_train, y_train, x_val, best_params)
        if acc > best_accuracy:
            best_accuracy = acc
            best_params["max_depth"] = max_depth

    for min_elements_in_subset in range(10):
        acc = comp_accuracy(x_train, y_train, x_val, best_params)
        if acc > best_accuracy:
            best_accuracy = acc
            best_params["min_sample_split"] = min_elements_in_subset

    for min_impurity_decrease in np.arange(0, 1, 0.1):
        acc = comp_accuracy(x_train, y_train, x_val, best_params)
        if acc > best_accuracy:
            best_accuracy = acc
            best_params["min_impurity_decrease"] = min_impurity_decrease

    return best_params

def comp_accuracy(x_train, y_train, x_val, y_val, params):
    # Create a decision treem model for the new values
    model = DecisionTreeClassifier(
        max_depth = params["max_depth"],
        min_sample_split = params["min_samples_split"],
        min_impurity_decrease = params["min_impurity_decrease"]
    )
    # Fit the model
    model.fit(x_train,y_train)
    val_predictions = model.predict(x_val)

    return np.mean(val_predictions == y_val)

def grid_search(x_train, y_train, x_test, x_val, y_val):
    # for i, (train_indices, val_indices, test_indices) in enumerate(train_val_test_k_fold(n_folds, len(x), rg)):
    #     # set up the dataset for this fold
    #     x_train = x[train_indices, :]
    #     y_train = y[train_indices]
    #     x_val = x[val_indices, :]
    #     y_val = y[val_indices]
    #     x_test = x[test_indices, :]
    #     y_test = y[test_indices]

    #

    # Iterate through depth
    # For each fold, call predict on training split
    # and evaluate accuracy using validation split
    # Append accuracy to an empty list
    # After evaluating all folds, return depth that gave the best accuracy
    # Now using this depth, repeat for min_sample_size
    # Now using that depth and min_sample_size, repeat for min_entropy_gain
    return


