# One-Class SVM Model for Hospitalization Risk 

A One-Class SVM (OCSVM) is an anomaly detection model based on Support Vector Machines (SVMs). This can be applied to our dataset, which features an extremely low positivity rate for hospitalization. The model implemented here uses an LibSVM's OCSVM model created by Zhongyi Que and Chih-Jen Lin. The most recent version, v3.30, has been updated to allow probability prediction for supervised OCSVMs.

Probabilities are desired to determine which patients have a higher risk of hospitalization.

## Dependencies
* LibSVM - Python (install using `pip3 install libsvm-official`)
* SciKit-Learn v1.1.3

## How to Run
While in the directory this README is housed in, run `python3 allFile.py`, which runs all three of the following files in order 
1. `data_prep.py` which reads in a comma separated values (csv) file and creates LibSVM formatted training and testing data for the OCSVM to use
2. `run_svm.py` which trains the OCSVM model and predicts the values
3. `metrics.py` which determines performance metrics like precision, recall, and F1 score

There is also a `hyperparams.ipynb` Jupyter Notebook, which uses grid search to find the best hyperparameters among a list of commonly used SVM hyperparameters.

## Structure
src

This folder contains the scripts used for this project

output

Contains any files created through executing the scripts. Contains training and testing data in LibSVM format as well as the predicted outputs and probabilities.

## Bibliography

Z. Que and C.-J. Lin. "One-class SVM probabilistic outputs." Technical report, 2022.

Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.

Link to LibSVM Github: https://github.com/cjlin1/libsvm

## Credits

Special thanks to Dr. Feng Chen at UTD for technical assistance throughout the development of this project as well as Chris Simonds for guidance and project management.
