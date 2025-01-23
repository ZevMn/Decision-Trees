## Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.


### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``improvement.py``

	* Contains the skeleton code for the ``train_and_predict()`` function (Task 4.1).
Complete this function as an interface to your new/improved decision tree classifier.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods/functions defined in ``classification.py`` and ``improvement.py``.


## Instructions for using git

### Initial set-up

Steps to run a local version of the repo

#### Clone the repository
```
git clone https://gitlab.doc.ic.ac.uk/lab2425_spring/intro2ml_cw1_12.git
```

#### Create and activate a venv
```
python -m venv venv
source venv/bin/activate
```

#### Install packages
```
pip install -r requirements.txt
```

### Using branches

#### Create a branch
```
git checkout -b [branch_name]
```

#### List all branches
```
git branch
```

#### Switch bewteen branches
```
git checkout [branch_name]
git checkout main
```

#### When pushing to remote, set an upstream branch like the below
```
git push -u origin [branch_name]
```

### Workflow

#### Always update your local main branch
```
git checkout main
git pull origin main
```

#### Switch to your feature branch and merge latest main
```
git checkout [branch_name]
git merge main
```

#### Stage and commit your changes
```
git add .
git commit -m "Descriptive commit message"
```

#### Push your branch to remote repository
```
git push -u origin [branch_name]
```





