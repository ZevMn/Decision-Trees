import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

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

    [classes,y_mask] = np.unique(y_labels, return_inverse=True)

    x_attribute = np.array(x_attribute)
    y = np.array(y_labels)

    ratio_list = []
    for i in classes:
        count = y_labels.count(i)
        ratio = count/len(y_labels)
        #print(f'{i} : {count}')
        print(f'{i} : {ratio*100:.2f}')
        ratio_list.append(ratio)

    return x_attribute, y_mask, classes

def display_barcharts(x1, y1, x2, y2, classes):
    ratio_list1 = []
    ratio_list2 = []

    for i in classes:
        ratio1 = np.count_nonzero(y1 == i) / len(y1)
        ratio2 = np.count_nonzero(y2 == i) / len(y2)
        ratio_list1.append(ratio1)
        ratio_list2.append(ratio2)

    chart_labels = list(classes)
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(chart_labels))
    width = 0.35

    plt.bar(bar_positions + width / 2, ratio_list1, width, label=chart_labels, color='blue', alpha=0.7, )
    plt.bar(bar_positions - width / 2, ratio_list2, width, label=chart_labels, color='green', alpha=0.7)

    plt.xlabel('Characters')
    plt.ylabel('Ratios')
    plt.title('Distribution of character appearances in datasets')
    plt.xticks(bar_positions, chart_labels)
    plt.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', label='Full dataset', markerfacecolor='blue', markersize=10),
                 plt.Line2D([0], [0], marker='o', color='w', label='Subset dataset', markerfacecolor='green', markersize=10)])

    plt.tight_layout()
    plt.show()

    return



# # Add value labels on top of each bar
# for i, (full_val, subset_val) in enumerate(zip(full_values, subset_values)):
#     plt.text(x[i] - width/2, full_val, f'{full_val:.2f}', ha='center', va='bottom')
#     plt.text(x[i] + width/2, subset_val, f'{subset_val:.2f}', ha='center', va='bottom')