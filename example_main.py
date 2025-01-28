##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from improvement import train_and_predict
from read_data import read_dataset, display_barcharts

if __name__ == "__main__":
    print("Printing the full dataset info:")
    x_full, y_full, classes_full = read_dataset("data/train_full.txt")
    x_sub, y_sub, classes_sub = read_dataset("data/train_sub.txt")
    x_noisy, y_noisy, classes_noisy = read_dataset("data/train_noisy.txt")

    x_test, y_test, classes_test = read_dataset("data/test.txt")
    x_val, y_val, classes_val = read_dataset("data/validation.txt")

    display_barcharts(x_full, y_full, x_sub, y_sub, classes_full)

    # print("Loading the training dataset...");
    # x = np.array([
    #         [5,7,1],
    #         [4,6,2],
    #         [4,6,3],
    #         [1,3,1],
    #         [2,1,2],
    #         [5,2,6]
    #     ])
    #
    # y = np.array(["A", "A", "A", "C", "C", "C"])

    print("Training the decision tree...")
    classifier_full = DecisionTreeClassifier()
    classifier_full.fit(x_full, y_full)

    classifier_sub = DecisionTreeClassifier()
    classifier_sub.fit(x_sub, y_sub)

    classifier_noisy = DecisionTreeClassifier()
    classifier_noisy.fit(x_noisy, y_noisy)

    # print("Loading the test set...")

    # x_test = np.array([
    #             [1,6,3],
    #             [0,5,5],
    #             [1,5,0],
    #             [2,4,2]
    #         ])
    #
    # y_test = np.array(["A", "A", "C", "C"])

    print("Making predictions on the test set for train_full...")
    predictions_full = classifier_full.predict(x_test)
    print("Predictions: {}".format(predictions_full))

    print("Making predictions on the test set for train_sub...")
    predictions_sub = classifier_sub.predict(x_test)
    print("Predictions: {}".format(predictions_sub))

    print("Making predictions on the test set for train_noisy...")
    predictions_noisy = classifier_noisy.predict(x_test)
    print("Predictions: {}".format(predictions_noisy))


    # x_val = np.array([
    #             [6,7,2],
    #             [3,1,3]
    #         ])
    # y_val = np.array(["A", "C"])

    # print("Training the improved decision tree, and making predictions on the test set...")
    # predictions = train_and_predict(x_full, y_full, x_test, x_val, y_val)
    # print("Predictions: {}".format(predictions))
    
