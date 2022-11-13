'''
Author: Dhruva Mambapoor

This model will take a CSV file as input and output a dump of a Univariate Gaussian Model.
Note on CSV: 
1. Must be named `frequencies.csv`.
2. First row and column are ignored; treated as header row and index column respectively;

Note on output:
1. Output is a file named `adl_gaussian_model_high_contamination_assume_centered.joblib`

To run the model, simply run `gaussian_test.py`.
'''

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.covariance import EllipticEnvelope

#SEE README.md for terminology brief.

#Reading and formatting data
data = pd.read_csv('frequencies.csv')

#Calculate 'contamination' = the percentage of outliers in the data-set
# Use 0.065 for "high_contamination" -> better results.
# Use commented "len(da..." for true contamination rate.
contam = 0.065 #len(data[data['hasHospitilization']  == 1])/len(data)

#Formatting data to include only training data. i.e. without clientID and hospitalization
data_train = np.array(data.iloc[:, 1:-1].values)
#print(data_train.shape)

#Create and run the model with previously defined contamination...
#   and assume center (assume the mean of the data-set is 0)
cov = EllipticEnvelope(contamination=contam, assume_centered=True).fit(data_train)
print("MODEL TRAINING COMPLETE!!!!!!")

#Outputs model to a joblib file. Joblib is basically just the raw object binary data stored in a file.
dump(cov, 'adl_gaussian_model_high_contamination_assume_centered.joblib')

print("MODEL SAVED!!!!!!!!!!!!")
