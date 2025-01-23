import numpy as np
import matplotlib as plt
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
    
    for i in classes:
        count = y_labels.count(i)
        ratio = count/len(y_labels)
        #print(f'{i} : {count}')
        print(f'{i} : {ratio*100:.2f}')

    return x_attribute, y, classes