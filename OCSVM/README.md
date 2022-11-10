# One-Class SVM Model for Hospitalization Risk 

A One-Class SVM (OCSVM) is an anomaly detection model based on Support Vector Machines (SVMs). This can be applied to our dataset, which features an extremely low positivity rate for hospitalization. The model implemented here uses an LibSVM's OCSVM model created by Zhongyi Que and Chih-Jen Lin. The most recent version, v3.30, has been updated to allow probability prediction for supervised OCSVMs.

Probabilities are desired to determine which patients have a higher risk of hospitalization.

## Dependencies
* LibSVM - Python (install using `pip3 install libsvm-official`)
* SciKitLearn
* Pandas

## How to Run
While in the directory this README is housed in:
1. Run data_prep.py using `python3 ./src/data_prep .py`
2. Run run_svm.py to create the OCSVM model and predict the values
3. Run metrics.py to determine performance metrics like precision, recall, and F1 score
5. Run python script to visualize data (optional?)

TO BE IMPLEMENTED:

Run the commands as listed in the jupyter notebook. It should compile the C source code, and then have the ability to run these scripts sequentially

## Structure
src

This folder contains the scripts used for this project

output

Contains any files created through running the scripts. Contains training and testing data as well as their outputs and probabilities.

## Bibliography

Z. Que and C.-J. Lin. "One-class SVM probabilistic outputs." Technical report, 2022.

Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.

Link to LibSVM Github: https://github.com/cjlin1/libsvm
