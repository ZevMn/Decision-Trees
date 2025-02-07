##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from evaluation import evaluate
from improvement import train_and_predict
from kfold import majority_vote, k_fold_train_and_evaluation
from read_data import read_dataset, display_barcharts, different_labels

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

    print("\nEvaluation of the model trained on the full dataset: ")
    evaluate(y_test, predictions_full, "train_full.txt")

    print("\nEvaluation of the model trained on the subset dataset: ")
    evaluate(y_test, predictions_sub, "train_sub.txt")

    print("\nEvaluation of the model trained on the noisy dataset: ")
    evaluate(y_test, predictions_noisy, "train_noisy.txt")

    """
        *******************************************
                        PART 3.2
        *******************************************
    """
    print("\n-----------------------------------------------------------------")
    print("PART 3.2: PERFORMING 10-FOLD CROSS VALIDATION ON THE FULL DATASET")
    print("-----------------------------------------------------------------")

    # Performing K-fold cross-validation

    print("\nPerforming k-fold cross-validation on full dataset:")
    avg_acc_full, std_dev_full, full_trees = k_fold_train_and_evaluation(x_full, y_full, n_folds=10)
    print(f"Avg Accuracy (Full): {avg_acc_full:.4f}, Std Dev: {std_dev_full:.4f}")

    '''Performing k-fold cross-validation on the subset and noisy datasets:'''
    # print("\nPerforming k-fold cross-validation on subset dataset:")
    # avg_acc_sub, std_dev_sub, sub_trees = k_fold_train_and_evaluation(x_sub, y_sub, n_folds=10)
    # print(f"Avg Accuracy (Subset): {avg_acc_sub:.4f}, Std Dev: {std_dev_sub:.4f}")
    #
    # print("\nPerforming k-fold cross-validation on noisy dataset:")
    # avg_acc_noisy, std_dev_noisy, noisy_trees = k_fold_train_and_evaluation(x_noisy, y_noisy, n_folds=10)
    # print(f"Avg Accuracy (Noisy): {avg_acc_noisy:.4f}, Std Dev: {std_dev_noisy:.4f}")


    """
        *******************************************
                        PART 3.3
        *******************************************
    """

    print("\n--------------------------------------------------------")
    print("PART 3.3: COMBINING THE PREDICTIONS USING MAJORITY VOTING")
    print("---------------------------------------------------------")

    print("\nPerforming majority voting on test set...")

    # Make predictions on the test set
    test_predictions = []
    for tree in full_trees:
        test_predictions.append(tree.predict(x_test))
    test_predictions = np.array(test_predictions) # Convert to numpy array

    # Use majority voting to combine predictions
    majority_predictions = majority_vote(test_predictions)

    # Calculate and print the final accuracy
    ensemble_accuracy = np.mean(majority_predictions == y_test)
    print(f"\nEnsemble Model Accuracy on Test Set: {ensemble_accuracy:.4f}")

    print("\nEvaluating ensemble model...")
    evaluate(y_test, majority_predictions, "K-fold")

    """
        *******************************************
                        PART 4
        *******************************************
    """

    print("\n-----------------------------------")
    print("PART 4: IMPROVING OUR DECISION TREE")
    print("-----------------------------------")


    print("Improving decision tree model (train_full)")
    single_tree_preds, majority_vote_preds = train_and_predict(x_full, y_full, x_test, x_val, y_val, n_folds=10)
    print("Evaluating 'improved' single tree model (train_full)")
    evaluate(y_test, single_tree_preds, "Single Tree")
    print("Evaluating 'improved' majority voting model (train_full")
    evaluate(y_test, majority_vote_preds, "Majority Voting")


    print("Improving decision tree model (train_noisy)")
    single_tree_preds_noisy, majority_voting_preds_noisy = train_and_predict(x_noisy, y_noisy, x_test, x_val, y_val, n_folds=10)
    print("Evaluating 'improved' single tree model (train_noisy)")
    evaluate(y_test, single_tree_preds_noisy, "Single Tree")
    print("Evaluating 'improved' majority voting model (train_noisy")
    evaluate(y_test, majority_voting_preds_noisy, "Majority Voting")


