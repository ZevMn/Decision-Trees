##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################
from itertools import accumulate

import numpy as np
from classification import DecisionTreeClassifier
from evaluation import Evaluation
#from improvement import train_and_predict
from read_data import read_dataset, display_barcharts, different_labels
from kfold import kfold

if __name__ == "__main__":

    """
        *******************************************
                        PART 1
        *******************************************
    """

    print("\n------------------------------------------")
    print("PART 1: LOADING AND EXAMINING THE DATASETS")
    print("------------------------------------------")

    print("\nPrinting the full dataset info:")
    x_full, y_full, classes_full = read_dataset("data/train_full.txt")

    print("\nPrinting the subset dataset info:")
    x_sub, y_sub, classes_sub = read_dataset("data/train_sub.txt")

    print("\nPrinting the noisy dataset info:")
    x_noisy, y_noisy, classes_noisy = read_dataset("data/train_noisy.txt")

    print("\nPrinting the test dataset info:")
    x_test, y_test, classes_test = read_dataset("data/test.txt")

    print("\nPrinting the validation dataset info:")
    x_val, y_val, classes_val = read_dataset("data/validation.txt")

    print("\nGenerating class distribution bar charts...")
    print("\nCalculating statistics:")
    display_barcharts(y_full, y_sub, classes_full, "train_full.txt", "train_sub.txt")
    different_labels(x_full, y_full, x_noisy, y_noisy, classes_full)

    """
        *******************************************
                        PART 2
        *******************************************
    """

    print("\n-------------------------------------------------")
    print("PART 2: IMPLEMENTING THE DECISION TREE CLASSIFIER")
    print("-------------------------------------------------")

    print("Training the decision tree on the full dataset...")
    classifier_full = DecisionTreeClassifier()
    classifier_full.fit(x_full, y_full)

    print("Training the decision tree on the subset dataset...")
    classifier_sub = DecisionTreeClassifier()
    classifier_sub.fit(x_sub, y_sub)

    print("Training the decision tree on the noisy dataset...")
    classifier_noisy = DecisionTreeClassifier()
    classifier_noisy.fit(x_noisy, y_noisy)

    print("Using the three trained models to make predictions on the test dataset...")
    predictions_full = classifier_full.predict(x_test)
    predictions_sub = classifier_sub.predict(x_test)
    predictions_noisy = classifier_noisy.predict(x_test)

    """
        *******************************************
                        PART 3.1
        *******************************************
    """

    print("\n---------------------------------------------")
    print("PART 3.1: EVALUATION OF THE MODEL PERFORMANCE")
    print("---------------------------------------------")

    evaluation = Evaluation()

    print("\nEvaluation of the model trained on the full dataset: ")
    evaluation.evaluate(y_test, predictions_full, "train_full.txt")

    print("\nEvaluation of the model trained on the subset dataset: ")
    evaluation.evaluate(y_test, predictions_sub, "train_sub.txt")

    print("\nEvaluation of the model trained on the noisy dataset: ")
    evaluation.evaluate(y_test, predictions_noisy, "train_noisy.txt")

    """
        *******************************************
                        PART 3.2
        *******************************************
    """
    print("\n-----------------------------------------------------------------")
    print("PART 3.2: PERFORMING 10-FOLD CROSS VALIDATION ON THE FULL DATASET")
    print("-----------------------------------------------------------------")

    # Performing K-fold cross-validation
    kfold_validator = kfold()

    print("\nPerforming k-fold cross-validation on full dataset:")
    avg_acc_full, std_dev_full = kfold_validator.k_fold_train_and_evaluation(x_full, y_full, n_folds=10)
    print(f"Avg Accuracy (Full): {avg_acc_full:.4f}, Std Dev: {std_dev_full:.4f}")

    print("\nPerforming k-fold cross-validation on subset dataset:")
    avg_acc_sub, std_dev_sub = kfold_validator.k_fold_train_and_evaluation(x_sub, y_sub, n_folds=10)
    print(f"Avg Accuracy (Subset): {avg_acc_sub:.4f}, Std Dev: {std_dev_sub:.4f}")

    print("\nPerforming k-fold cross-validation on noisy dataset:")
    avg_acc_noisy, std_dev_noisy = kfold_validator.k_fold_train_and_evaluation(x_noisy, y_noisy, n_folds=10)
    print(f"Avg Accuracy (Noisy): {avg_acc_noisy:.4f}, Std Dev: {std_dev_noisy:.4f}")

    print("Performing majority voting on test set...")
    fold_predictions = []

    # for train_indices, val_indices, test_indices in kfold_validator.train_val_test_k_fold(10, len(x_full)):
    #     classifier = DecisionTreeClassifier()  # Create new instance for each fold
    #     classifier.fit(x_full[train_indices], y_full[train_indices])
    #     fold_predictions.append(classifier.predict(x_test))
    #
    # fold_predictions = np.array(fold_predictions)

    """
        *******************************************
                        PART 3.3
        *******************************************
    """

    # # Compute majority vote
    # final_predictions = kfold_validator.majority_vote(fold_predictions)
    #
    # print("Evaluating ensemble model...")
    # evaluation.evaluate(y_test, final_predictions, "K-fold")

    """
        *******************************************
                        PART 4
        *******************************************
    """
