import pandas as pd
import numpy as np
from joblib import load
from sklearn.covariance import EllipticEnvelope



#Reading and formatting data
data = pd.read_csv('frequencies.csv')
#Formatting hospitalization data to fit standard sci-kit output. i.e. (-1 = outlier, 1 = normal)
def hosp_formatter(x):
    return -2*x + 1
data['hasHospitilization'] = data['hasHospitilization'].apply(hosp_formatter)
contam = len(data[data['hasHospitilization']  == -1])/len(data)

#Formatting data to include only training data. i.e. without clientID and hospitalization
data_train = np.array(data.iloc[:, 1:-1].values)

#Loading previously trained model
cov = load('adl_gaussian_model_assume_centered.joblib')

#Classify data into outliers and normals.
y_pred = cov.predict(data_train)
y_true = data.iloc[:, -1].values


true_p = 0
false_p = 0
false_n = 0
for i in range(0, len(y_true)):
    if y_pred[i] == -1:
        if y_true[i] == -1:
            #If model predicts hospitalization, and ground-truth is hospitalization, then add to the true positive
            true_p += 1
        else:
            #If model predicts hospitalization, and ground-truth is NOT hospitalization, then add to the false positive
            false_p += 1
    else:
        if y_pred[i] == -1:
            false_n += 1

print("True Positives: %i" %(true_p))
print("False Positives: %i" %(false_p))
print("False Negatives: %i" %(false_n))

#Calculate 'precision' : (true positives) / (true positives + false positives)
precision = 0 if (true_p + false_p) == 0 else float(true_p)/float(true_p + false_p)
print("Precision: %f" % (precision))
#Calculate 'recall' : (true positives) / (true positives + false negatives)
recall = 0 if (true_p + false_n) == 0 else float(true_p)/float(true_p + false_n)
print("Recall: %f" % (recall))
#Calculate 'f1-score' : 2*precision*recall / (precision + recall)
f1_score =  0 if (precision + recall) == 0 else (2*precision*recall) / (precision + recall)
print("F1-Score: %f" %(f1_score))
