import numpy as np

class Evaluation(object):

    def __init__(self):
        pass


    def confusion_matrix(self, y_gold, y_prediction, class_labels=None):


        # if no class_labels are given, we obtain the set of unique class labels from
        # the union of the ground truth annotation and the prediction
        if not class_labels:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

        confusion = np.zeros((len(class_labels), len(class_labels)))

        # for each correct class (row),
        # compute how many instances are predicted for each class (columns)
        for (i, label) in enumerate(class_labels):
            # get predictions where the ground truth is the current class label
            indices = (y_gold == label)
            gold = y_gold[indices]
            predictions = y_prediction[indices]

            # quick way to get the counts per label
            (unique_labels, counts) = np.unique(predictions, return_counts=True)

            # convert the counts to a dictionary
            frequency_dict = dict(zip(unique_labels, counts))

            # fill up the confusion matrix for the current row
            for (j, class_label) in enumerate(class_labels):
                confusion[i, j] = frequency_dict.get(class_label, 0)

        return confusion


    def accuracy(self, y_gold, y_prediction):

        assert len(y_gold) == len(y_prediction)

        try:
            return np.sum(y_gold == y_prediction) / len(y_gold)
        except ZeroDivisionError:
            return 0.


    def precision(self, y_gold, y_prediction):

        confusion = self.confusion_matrix(y_gold, y_prediction)
        p = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[:, c]) > 0:
                p[c] = confusion[c, c] / np.sum(confusion[:, c])

        # Compute the macro-averaged precision
        macro_p = 0.
        if len(p) > 0:
            macro_p = np.mean(p)

        return (p, macro_p)


    def recall(self, y_gold, y_prediction):

        confusion = self.confusion_matrix(y_gold, y_prediction)
        r = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[c, :]) > 0:
                r[c] = confusion[c, c] / np.sum(confusion[c, :])

        # Compute the macro-averaged recall
        macro_r = 0.
        if len(r) > 0:
            macro_r = np.mean(r)

        return (r, macro_r)


    def f1_score(self, y_gold, y_prediction):

        (precisions, macro_p) = self.precision(y_gold, y_prediction)
        (recalls, macro_r) = self.recall(y_gold, y_prediction)

        # just to make sure they are of the same length
        assert len(precisions) == len(recalls)

        f = np.zeros((len(precisions),))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)

        # Compute the macro-averaged F1
        macro_f = 0.
        if len(f) > 0:
            macro_f = np.mean(f)

        return (f, macro_f)


    def evaluate(self, y_gold, y_prediction):
        confusion = self.confusion_matrix(y_gold, y_prediction)
        accuracy = self.accuracy(y_gold, y_prediction)
        precision = self.precision(y_gold, y_prediction)
        recall = self.recall(y_gold, y_prediction)
        f1_score = self.f1_score(y_gold, y_prediction)
        print("Confusion: ", confusion)
        print("Accuracy: " , accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1_score)


        return confusion, accuracy, precision, recall, f1_score
