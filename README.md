## Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Data

The ``data/`` directory contains the datasets for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets to help with implementation or debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``.


### Code

- ``classification.py``

	* Contains the code for the ``DecisionTreeClassifier`` class. The ``train()`` and ``predict()`` methods were implemented as part of this coursework.

- ``evaluation.py``
	* Contains functions and classes for evaluating model performance. This includes calculation of accuracy, precision, recall, F1-score and confusion matrices.

- ``example_main.py``
	* Provides a sample script demonstrating how to use the decision tree classifier end-to-end. It shows how to import data, train a model, make predictions, and evaluate results.

- ``improvement.py``
	* Contains the code for the ``train_and_predict()`` function. This function serves as an interface to the new/improved decision tree classifier, showcasing any optimisations or enhancements over the original classifier.

- ``kfold.py``
	* Implements K-Fold cross-validation logic. It handles partitioning the dataset into multiple folds, training on each fold, and aggregating performance metrics across folds.
