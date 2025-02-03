import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng

from classification import DecisionTreeClassifier


class Evaluation(object):

    def __init__(self):
        pass

    def confusion_matrix(self, y_gold, y_prediction, class_labels=None):
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

        confusion = np.zeros((len(class_labels), len(class_labels)))

        # For each correct class (row),
        # compute how many instances are predicted for each class (columns)
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

    def accuracy(self, y_gold, y_prediction):
        """
        Compute the accuracy of the predictions.

        Args:
            y_gold (np.array): Ground truth labels.
            y_prediction (np.array): Predicted labels.

        Returns:
            float: Accuracy.
        """
        assert len(y_gold) == len(y_prediction)
        try:
            return np.sum(y_gold == y_prediction) / len(y_gold)
        except ZeroDivisionError:
            return 0.

    def precision(self, y_gold, y_prediction):
        """
        Compute precision for each class and the macro-averaged precision.

        Args:
            y_gold (np.array): Ground truth labels.
            y_prediction (np.array): Predicted labels.

        Returns:
            tuple: (precision per class, macro-averaged precision).
        """
        confusion = self.confusion_matrix(y_gold, y_prediction)
        p = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[:, c]) > 0:
                p[c] = confusion[c, c] / np.sum(confusion[:, c])

        # Compute the macro-averaged precision
        macro_p = np.mean(p) if len(p) > 0 else 0.
        return (p, macro_p)

    def recall(self, y_gold, y_prediction):
        """
        Compute recall for each class and the macro-averaged recall.

        Args:
            y_gold (np.array): Ground truth labels.
            y_prediction (np.array): Predicted labels.

        Returns:
            tuple: (recall per class, macro-averaged recall).
        """
        confusion = self.confusion_matrix(y_gold, y_prediction)
        r = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[c, :]) > 0:
                r[c] = confusion[c, c] / np.sum(confusion[c, :])

        # Compute the macro-averaged recall
        macro_r = np.mean(r) if len(r) > 0 else 0.
        return (r, macro_r)

    def f1_score(self, y_gold, y_prediction):
        """
        Compute F1 score for each class and the macro-averaged F1 score.

        Args:
            y_gold (np.array): Ground truth labels.
            y_prediction (np.array): Predicted labels.

        Returns:
            tuple: (F1 score per class, macro-averaged F1 score).
        """
        (precisions, macro_p) = self.precision(y_gold, y_prediction)
        (recalls, macro_r) = self.recall(y_gold, y_prediction)

        assert len(precisions) == len(recalls)

        f = np.zeros((len(precisions),))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)

        # Compute the macro-averaged F1 score
        macro_f = np.mean(f) if len(f) > 0 else 0.
        return (f, macro_f)

    def plot_confusion_matrix(self, confusion, class_labels, title="Confusion Matrix"):
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
            fmt=".2f",  # Use floating-point format with 2 decimal places
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def plot_metrics(self, y_gold, y_prediction, class_labels=None):
        """
        Plot precision, recall, and F1 score for each class.

        Args:
            y_gold (np.array): Ground truth labels.
            y_prediction (np.array): Predicted labels.
            class_labels (np.array, optional): List of class labels. If None, inferred from y_gold and y_prediction.
        """
        if class_labels is None:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

        # Compute metrics
        (precisions, macro_p) = self.precision(y_gold, y_prediction)
        (recalls, macro_r) = self.recall(y_gold, y_prediction)
        (f1_scores, macro_f) = self.f1_score(y_gold, y_prediction)

        # Plot metrics
        metrics = {
            "Precision": precisions,
            "Recall": recalls,
            "F1 Score": f1_scores
        }

        plt.figure(figsize=(10, 6))
        for metric_name, values in metrics.items():
            plt.plot(class_labels, values, marker='o', label=metric_name)

        plt.title("Precision, Recall, and F1 Score per Class")
        plt.xlabel("Class Labels")
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate(self, y_gold, y_prediction, class_labels=None):
        """
        Evaluate the model and print/plot metrics.

        Args:
            y_gold (np.array): Ground truth labels.
            y_prediction (np.array): Predicted labels.
            class_labels (np.array, optional): List of class labels. If None, inferred from y_gold and y_prediction.

        Returns:
            tuple: (confusion matrix, accuracy, precision, recall, F1 score).
        """
        if class_labels is None:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

        confusion = self.confusion_matrix(y_gold, y_prediction, class_labels)
        accuracy = self.accuracy(y_gold, y_prediction)
        precision = self.precision(y_gold, y_prediction)
        recall = self.recall(y_gold, y_prediction)
        f1_score = self.f1_score(y_gold, y_prediction)

        print("Confusion Matrix:")
        print(confusion)
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1_score)

        # Plot confusion matrix and metrics
        self.plot_confusion_matrix(confusion, class_labels)
        self.plot_metrics(y_gold, y_prediction, class_labels)

        return confusion, accuracy, precision, recall, f1_score

