import pandas as pd
import numpy as np
from joblib import dump
from sklearn.covariance import EllipticEnvelope

#Reading and formatting data
data = pd.read_csv('frequencies.csv')

#Calculate 'contamination' = the percentage of outliers in the data-set
contam = len(data[data['hasHospitilization']  == 1])/len(data)

#Formatting data to include only training data. i.e. without clientID and hospitalization
data_train = np.array(data.iloc[:, 1:-1].values)
#print(data_train.shape)

#Create and run the model.
cov = EllipticEnvelope(contamination=contam, assume_centered=True).fit(data_train)
print("MODEL TRAINING COMPLETE!!!!!!")

dump(cov, 'adl_gaussian_model_assume_centered.joblib')

print("MODEL SAVED!!!!!!!!!!!!")
