import numpy as np

from classification import DecisionTreeClassifier

def k_fold_split(n_instances, n_folds=10, random_generator=np.random.default_rng(42)):
    """
    Splits dataset indices into k folds using random shuffling.

    Args:
        n_instances (int): Number of samples in the dataset.
        n_folds (int): Number of folds (default: 10).
        random_generator: Random generator instance for reproducibility.

    Returns:
        list: Each element is a numpy array containing the indices of the instances in that fold.
    """
    # Generate a random permutation of indices
    shuffled_indices = random_generator.permutation(n_instances)
    # Split shuffled indices into k folds
    split_indices = np.array_split(shuffled_indices, n_folds)

    return split_indices

def train_test_k_fold(n_instances, n_folds=10, random_generator=np.random.default_rng(42)):
    """ Generate train, validation, and test indices for k-fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a tuple
            with two elements: a numpy array containing the train indices
            and a numpy array containing the test indices.
    """

    # Split the dataset into k splits of indices
    split_indices = k_fold_split(n_instances, n_folds, random_generator)

    folds = []
    # Iterate through the folds each time selecting one as the test set and the rest for training
    for k in range(n_folds):
        # Select the current fold as the test set
        test_indices = split_indices[k]

        # Combine all other folds for training
        # NB: np.hstack() horizontally stacks (concatenates) arrays
        train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

        folds.append([train_indices, test_indices])

    return folds

def k_fold_train_and_evaluation(x, y, n_folds=10):

    # Store accuracy for each fold
    accuracies = np.zeros((n_folds,))
    # Store trained decision trees
    trees = []

    # Random generator for shuffling
    rg = np.random.default_rng()
    # Get train/test splits
    folds = train_test_k_fold(len(x), n_folds, rg)

    # Iterate through each split of the folds and compute the accuracy
    for i, (train_indices, test_indices) in enumerate(folds):

        # Get the dataset from the split indices
        x_train, y_train = x[train_indices, :], y[train_indices]
        x_test, y_test = x[test_indices, :], y[test_indices]

        # Train the decision tree
        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        trees.append(tree)

        # Test it
        predictions = tree.predict(x_test)
        accuracy = np.mean(predictions == y_test)
        accuracies[i] = accuracy  # Store test accuracy

    print("Accuracies per fold:", accuracies)
    print("Average Accuracy:", accuracies.mean())
    print("Standard Deviation:", accuracies.std())

    return accuracies.mean(), accuracies.std(), trees

def majority_vote(predictions):
    """
    Combines predictions from multiple models using majority voting.

    Args:
        predictions (numpy.ndarray): Array of shape (n_models, n_samples) containing
                                   predictions from multiple models

    Returns:
        numpy.ndarray: Array of length n_samples containing the majority vote predictions
    """
    n_samples = predictions.shape[1]
    final_predictions = np.empty(n_samples, dtype=predictions.dtype)

    # For each sample
    for i in range(n_samples):
        # Get predictions from all models for this sample
        sample_predictions = predictions[:, i]
        # Count occurrences of each unique label
        unique_labels, counts = np.unique(sample_predictions, return_counts=True)
        # Select the label with the highest count
        final_predictions[i] = unique_labels[np.argmax(counts)]

    return final_predictions
