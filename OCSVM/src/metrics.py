#  This script provides anomaly detection metrics (precision, recall, F1 score) to determine how good the OCSVM performed
#  Inputs:
#     File containing Test Predictions
#     File containing True Test Positives
#     
#  Outputs:
#     Console:
#         Predicted and True Positives (Count and Percentage)
#        Anomaly Detection Metrics (Precision, Recall, F1 Score)

# TODO:
#  Rename variables to be generalized (e.g. preds instead of gamma_preds)

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, precision_score


#file names and paths
output_path = "./output"

gamma_preds_file_name = "ocsvm_preds"
gamma_preds_path = Path(output_path, gamma_preds_file_name)

test_data_file_name = "ocsvm_true_test"
test_output_path = Path(output_path , test_data_file_name)


#read files and load into arrays
pred_outcomes = np.loadtxt(gamma_preds_path) < 0
actual_outcomes = np.loadtxt(test_output_path) < 0

print(pred_outcomes)
print(actual_outcomes)

#numpy wizardy lets you find number of bools based on sum of bools
num_gauss_preds = pred_outcomes.sum()
real_positives = actual_outcomes.sum()

gauss_pred_pos_percent = num_gauss_preds/pred_outcomes.shape[0]
real_pos_percentage = real_positives/actual_outcomes.shape[0]


#find positivity rate for prediction and actual outcomes
print("Number of Predicted Positives:", num_gauss_preds)
print("Actual Positives in Dataset:", real_positives)

print("Percentage of Predicted Positives:",gauss_pred_pos_percent)
print("Percentage of Actual Positives in Test Dataset", real_pos_percentage)

#compute anomaly detection metrics: precision, recall, and f1 score
prec_score = precision_score(actual_outcomes, pred_outcomes, average = "binary")
rec_score = recall_score(actual_outcomes, pred_outcomes, average = "binary")
f_score = f1_score(actual_outcomes, pred_outcomes, average = "binary")

print("Precision Score: " + str(prec_score))
print("Recall Score: " + str(rec_score))
print("F1 Score: " + str(f_score))