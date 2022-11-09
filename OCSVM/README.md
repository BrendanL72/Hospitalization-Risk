# One-Class SVM Model for Hospitalization Risk 

A One-Class SVM (OCSVM) is an anomaly detection model based on Support Vector Machines (SVMs). This can be applied to our dataset, which features an extremely low positivity rate for hospitalization. The model implemented here uses an experimental version of LibSVM's OCSVM model created by Zhongyi Que and Chih-Jen Lin modified to output probabilities.

Probabilities are desired to determine which patients have a higher risk of hospitalization.

This currently only runs on Linux devices.

## Dependencies
* LibSVM - Python
* SciKitLearn
* Pandas

## How to Run
While in the directory this README is housed in:
1. Run data_prep.py using `python3 ./src/data_prep .py`
2. Run run_svm.sh to create the OCSVM model
3. Run a python script to get predicted probabilities
4. Run another python script to get obtain performance metrics
5. Run python script to visualize data (optional?)

TO BE IMPLEMENTED:

Run the commands as listed in the jupyter notebook. It should compile the C source code, and then have the ability to run these scripts sequentially

## Structure

script

This folder contains scripts to predict probability, perform metrics, and visualizations.

src

This folder contains the source code created by Zhongyi Que and Chih-Jen Lin used to create the model and predictions for the data.

output

Contains any files created through running the script. Contains training and testing data as well as their outputs and probabilities.

## Bibliography

Z. Que and C.-J. Lin. "One-class SVM probabilistic outputs." Technical report, 2022.

Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.