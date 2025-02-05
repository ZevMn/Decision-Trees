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

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################
       

    # TODO: Train new classifier
    best_params = grid_search(x_train, y_train, x_test, x_val, y_val);
    # set up an empty (M, ) numpy array to store the predicted labels 
    # feel free to change this if needed
    improved_tree = DecisionTreeClassifier(max_depth = best_params["max_depth"], min_samples_split = best_params["min_samples_split"])
    improved_tree.fit(x_train, y_train)

    predictions = np.zeros((x_test.shape[0],), dtype=object)
        
    # TODO: Make predictions on x_test using new classifier        
        
    # remember to change this if you rename the variable
    return predictions

def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list of length n_folds. Each element in the list is a list (or tuple)
                with three elements:
                - a numpy array containing the train indices
                - a numpy array containing the val indices
                - a numpy array containing the test indices
        """
    shuffled_indices = random_generator.permutation(n_folds)
    fold_splits = np.array_split(shuffled_indices, n_folds)
    folds = []

    for k in range(n_folds):
        test_indices = fold_splits[k]
        remaining_folds = [fold_splits ]

    return

def optimise_parameters():
    for max_depth in [None, 1, 5, 10]:
        # Call grid search
        # Call comp_accuracy to find best max_depth
        pass
    for min_elements_in_subset in range(10):
        # Call grid search with max_depth set to the above
        # Call comp_accuracy to find best min_elements_in_subset
        pass
    for min_impurity_decrease in np.arange(0, 1, 0.1):
        # Call grid search with max_depth and min_elements_in_subset set to the above
        # Call comp_accuracy to find best min_impurity_decrease
        pass
    return

def comp_accuracy():
    return;

def grid_search(x_train, y_train, x_test, x_val, y_val):
    # for i, (train_indices, val_indices, test_indices) in enumerate(train_val_test_k_fold(n_folds, len(x), rg)):
    #     # set up the dataset for this fold
    #     x_train = x[train_indices, :]
    #     y_train = y[train_indices]
    #     x_val = x[val_indices, :]
    #     y_val = y[val_indices]
    #     x_test = x[test_indices, :]
    #     y_test = y[test_indices]

    # Iterate through depth
    # For each fold, call predict on training split
    # and evaluate accuracy using validation split
    # Append accuracy to an empty list
    # After evaluating all folds, return depth that gave the best accuracy
    # Now using this depth, repeat for min_sample_size
    # Now using that depth and min_sample_size, repeat for min_entropy_gain
    return


