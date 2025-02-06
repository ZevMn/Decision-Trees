import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and should be integers from 0 to C-1
                   where C is the number of classes
               - classes : a numpy array with shape (C, ), which contains the
                   unique class labels corresponding to the integers in y
    """

    x_attribute = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "": # handle empty rows in file
            row = line.strip().split(",")
            x_attribute.append(list(map(int, row[:-1])))
            y_labels.append(row[-1])

    [classes,_] = np.unique(y_labels, return_inverse=True)

    x_attribute = np.array(x_attribute)
    y = np.array(y_labels)

    ratio_list = []
    for i in classes:
        count = y_labels.count(i)
        ratio = count/len(y_labels)
        print(f'{i} : {ratio*100:.2f}')
        ratio_list.append(ratio)

    return x_attribute, y, classes


def display_barcharts(y1, y2, classes, legend1, legend2):

    ratio_list1 = []
    ratio_list2 = []
    counts1 = []
    counts2 = []

    for i in classes:
        count1 = np.count_nonzero(y1 == i)
        counts1.append(count1)
        ratio1 = count1 / len(y1)
        count2 = np.count_nonzero(y2 == i)
        counts2.append(count2)
        ratio2 = count2 / len(y2)
        ratio_list1.append(ratio1)
        ratio_list2.append(ratio2)

    chart_labels = list(classes)
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(chart_labels))
    width = 0.35

    plt.bar(bar_positions + width / 2, ratio_list1, width, label=chart_labels, color='blue', alpha=0.7, )
    plt.bar(bar_positions - width / 2, ratio_list2, width, label=chart_labels, color='green', alpha=0.7)

    print("Standard deviation: ", np.std(ratio_list1))
    print("Range: ", np.max(ratio_list1) - np.min(ratio_list1))
    print("Standard deviation: ", np.std(ratio_list2))
    print("Range: ", np.max(ratio_list2) - np.min(ratio_list2))

    for bar, count in zip(bar_positions, counts1):
        plt.text(bar + (width / 2), count/sum(counts1) - 0.01, str(count), ha='center', va='bottom', fontsize=10, color='white')
    for bar, count in zip(bar_positions, counts2):
        plt.text(bar - (width / 2), count/sum(counts2) - 0.01, str(count), ha='center', va='bottom', fontsize=10, color='white')


    plt.xlabel('Characters represented in the datasets')
    plt.ylabel('Proportion of class instances in a given dataset')
    plt.title("Frequency distribution of class instances in datasets 'train_full.txt' and 'train_sub.txt'")
    plt.xticks(bar_positions, chart_labels)
    plt.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', label=legend1, markerfacecolor='blue', markersize=10),
                 plt.Line2D([0], [0], marker='o', color='w', label=legend2, markerfacecolor='green', markersize=10)], loc='upper left', bbox_to_anchor=(1,1))

    plt.tight_layout()
    plt.show()


def different_labels(x_full, y_full, x_noisy, y_noisy, classes):

    sorted_indices_full = np.lexsort(x_full.T)
    sorted_indices_noisy = np.lexsort(x_noisy.T)

    y_full_sorted = y_full[sorted_indices_full]
    y_noisy_sorted = y_noisy[sorted_indices_noisy]

    num_changed = np.sum(y_full_sorted != y_noisy_sorted)
    total_labels = len(y_full_sorted)
    proportion_changed = num_changed / total_labels

    print(f"\nProportion of changed labels: {proportion_changed:.4f} ({num_changed} out of {total_labels})")

    display_barcharts(y_full_sorted, y_noisy_sorted, classes, "train_full.txt", "train_noisy.txt")
