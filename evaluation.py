import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng

from classification import DecisionTreeClassifier

def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """
    Compute the confusion matrix.

    Args:
        y_gold (np.array): Ground truth labels.
        y_prediction (np.array): Predicted labels.
        class_labels (np.array, optional): List of class labels. If None, inferred from y_gold and y_prediction.

    Returns:
        np.array: Confusion matrix.
    """

    # If no class_labels are given, obtain the set of unique class labels
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    # Create a base confusion matrix filled with zeros
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # For each correct class (row) compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # Get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        predictions = y_prediction[indices]

        # Quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # Convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # Fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion

def accuracy(y_gold, y_prediction):
    """
    Compute the accuracy of the predictions.

    Args:
        y_gold (np.array): Ground truth labels.
        y_prediction (np.array): Predicted labels.

    Returns:
        float: Accuracy.
    """
    # Ensure both arrays have the same length
    assert len(y_gold) == len(y_prediction)

    try:
        # Calculate the proportion of correct predictions
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        # Handle the case where the input is empty
        return 0.0

# To check accuracy is correct:
def accuracy_from_confusion(confusion):
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """
    # Ensure the confusion matrix is not empty
    if np.sum(confusion) > 0:
        # Calculate accuracy as sum of diagonal elements divided by total sum
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        # Handle the case where the confusion matrix is empty
        return 0.

def precision(confusion):
    """
    Compute precision for each class and the macro-averaged precision.

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        tuple: (precision per class, macro-averaged precision).
    """
    # Initialise an array to store precision for each class
    precisions = np.zeros((len(confusion),))

    # Loop through each class to compute its precision
    for c in range(confusion.shape[0]):
        # Check if the total predicted positives for the class is non-zero
        if np.sum(confusion[:, c]) > 0:
            # Compute precision as true positives / total prediction
            precisions[c] = confusion[c, c] / np.sum(confusion[:, c])

    # Compute the macro-averaged precision
    macro_precision = np.mean(precisions) if len(precisions) > 0 else 0.0

    return precisions, macro_precision

def recall(confusion):
    """
    Compute recall for each class and the macro-averaged recall.

    Args:
        confusuion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        tuple: (recall per class, macro-averaged recall).
    """
    # Initialise an array to store recall for each class
    recalls = np.zeros((len(confusion),))

    for c in range(confusion.shape[0]):
        # Compute recall for class c if there are any ground truth sample for class c
        if np.sum(confusion[c, :]) > 0:
            recalls[c] = confusion[c, c] / np.sum(confusion[c, :])

    # Compute the macro-averaged recall
    macro_recall = np.mean(recalls) if len(recalls) > 0 else 0.
    return recalls, macro_recall

def f1_score(confusion):
    """
    Compute F1 score for each class and the macro-averaged F1 score.

    Args:
        confusuion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        tuple: (F1 score per class, macro-averaged F1 score).
    """
    # Compute precision and recall for all classes
    (precisions, macro_precision) = precision(confusion)
    (recalls, macro_recall) = recall(confusion)

    # Ensure the number of classes in precision and recall match
    assert len(precisions) == len(recalls) # Sanity check

    # Initialise an array to store F1 scores for each class
    f1_scores = np.zeros((len(precisions),))

    for c, (prec, rec) in enumerate(zip(precisions, recalls)):
        # Compute F1 score only if precision + recall > 0 to avoid division by zero
        if prec + rec > 0:
            f1_scores[c] = 2 * prec * rec / (prec + rec)

    # Compute the macro-averaged F1 score
    macro_f1_score = np.mean(f1_scores) if len(f1_scores) > 0 else 0.
    return (f1_scores, macro_f1_score)


def plot_confusion_matrix(confusion, class_labels, title="Confusion Matrix"):
    """
    Plot the confusion matrix as a heatmap.

    Args:
        confusion (np.array): Confusion matrix.
        class_labels (np.array): List of class labels.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".2f",  
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title(f"Confusion Matrix {title}", fontsize=25, pad=20)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"Confusion_{title}.png")
    plt.show()

def plot_metrics(y_gold, y_prediction, confusion, class_labels=None, title="Precision, Recall, and F1 Score per Class"):
    """
    Plot precision, recall, and F1 score for each class.

    Args:
        y_gold (np.array): Ground truth labels.
        y_prediction (np.array): Predicted labels.
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions
        class_labels (np.array, optional): List of class labels. If None, inferred from y_gold and y_prediction.
        title (str, optional): Title of the plot.
    """
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    # Compute metrics
    (precisions, macro_precision) = precision(confusion)
    (recalls, macro_recall) = recall(confusion)
    (f1_scores, macro_f1_score) = f1_score(confusion)

    # Plot metrics
    metrics = {
        "Precision": precisions,
        "Recall": recalls,
        "F1 Score": f1_scores
    }

    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(class_labels, values, marker='o', label=metric_name)

    plt.title(f"Evaluation metrics for {title}", fontsize=25, pad=20)
    plt.xlabel("Class Labels")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.savefig(f"Eval_{title}.png")

def evaluate(y_gold, y_prediction, title, class_labels=None):
    """
    Evaluate the model and print/plot metrics.

    Args:
        y_gold (np.array): Ground truth labels.
        y_prediction (np.array): Predicted labels.
        title: Title of the plot.
        class_labels (np.array, optional): List of class labels. If None, inferred from y_gold and y_prediction.

    Returns:
        tuple: (confusion matrix, accuracy, precision, recall, F1 score).
    """
    # Infer class label if not provided
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    # Compute metrics
    confusion = confusion_matrix(y_gold, y_prediction, class_labels)
    acc = accuracy(y_gold, y_prediction)
    precisions, macro_precision = precision(confusion)
    recalls, macro_recall = recall(confusion)
    f1_scores, macro_f1_score = f1_score(confusion)

    second_accuracy_calc = accuracy_from_confusion(confusion)
    assert acc == accuracy_from_confusion(confusion) # Sanity check

    # Display evaluation metrics
    print("Confusion Matrix: ", confusion)
    print("Accuracy: ", acc)
    print("Accuracy (confusion calculation): ", second_accuracy_calc)
    print("Precisions: ", precisions)
    print("Macro-averaged precision: ", macro_precision)
    print("Recalls: ", recalls)
    print("Macro-averaged recall: ", macro_recall)
    print("F1 Score: ", f1_scores)
    print("Macro-averaged F1 Score: ", macro_f1_score)


    # Plot confusion matrix and metrics
    plot_confusion_matrix(confusion, class_labels, title)
    plot_metrics(y_gold, y_prediction, confusion, class_labels, title)