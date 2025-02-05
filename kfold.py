import numpy as np

class kfold(object):

    def __init__(self):
        pass


    def k_fold_split(self,n_instances, n_splits=10, random_generator=np.random.default_rng()):
        """
        Splits dataset indices into k folds using random shuffling.

        Args:
            n_instances (int): Number of samples in the dataset.
            n_splits (int): Number of folds (default: 10).
            random_generator: Random generator instance for reproducibility.

        Returns:
            list: Each element is a numpy array containing the indices of the instances in that fold.
        """
        shuffled_indices = random_generator.permutation(n_instances)  # Generate a random permutation of indices
        split_indices = np.array_split(shuffled_indices, n_splits)  # Split into k folds
        return split_indices

    def train_val_test_k_fold(self, n_folds, n_instances, random_generator=np.random.default_rng()):
        """ Generate train, validation, and test indices for k-fold.

        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list of length n_folds. Each element in the list is a tuple
                with three elements: train indices, validation indices, and test indices.
        """
        split_indices = self.k_fold_split(n_instances, n_folds, random_generator)

        folds = []
        for k in range(n_folds):
            test_indices = split_indices[k]

            # Use first remaining split as validation set
            remaining_splits = [split_indices[j] for j in range(n_folds) if j != k]
            val_indices = remaining_splits[0]  # Select first split as validation set

            # Remaining data as training set
            train_indices = np.hstack(remaining_splits[1:])  # Exclude val and test indices

            folds.append((train_indices, val_indices, test_indices))

        return folds

    def k_fold_evaluation(self, model, X, Y, n_folds=10):
        accuracies = np.zeros((n_folds,))
        rg = np.random.default_rng()
        folds = self.train_val_test_k_fold(n_folds, len(X), rg)

        for i, (train_indices, val_indices, test_indices) in enumerate(folds):
            x_train, y_train = X[train_indices], Y[train_indices]
            x_val, y_val = X[val_indices], Y[val_indices]
            x_test, y_test = X[test_indices], Y[test_indices]

            # Train the model
            model.fit(x_train, y_train)

            # Validate model (Optional: You can perform hyperparameter tuning here)
            y_val_pred = model.predict(x_val)
            val_accuracy = np.mean(y_val_pred == y_val)

            # Test the best model on test set
            y_test_pred = model.predict(x_test)
            test_accuracy = np.mean(y_test_pred == y_test)

            accuracies[i] = test_accuracy  # Store test accuracy

        print("Accuracies per fold:", accuracies)
        print("Average Accuracy:", accuracies.mean())
        print("Standard Deviation:", accuracies.std())

        return accuracies.mean(), accuracies.std()

    def majority_vote(self, predictions):
        n_samples = predictions.shape[1]
        final_predictions = np.empty(n_samples, dtype=predictions.dtype)
        for i in range(n_samples):
            unique_labels, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions[i] = unique_labels[np.argmax(counts)]
        return final_predictions
