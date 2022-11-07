#This script provides anomaly detection metrics (precision, recall, F1 score) to determine how good the OCSVM performed

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, precision_score

output_path = "./output"

gamma_preds_file_name = "gamma_preds"
gamma_preds_path = Path(output_path, gamma_preds_file_name)

test_data_file_name = "true_test_values"
test_output_path = Path(output_path , test_data_file_name)

#read files and load into arrays
pred_outcomes = np.loadtxt(gamma_preds_path).astype(bool)

actual_outcomes = np.loadtxt(test_output_path).astype(bool)

print(pred_outcomes)
print(actual_outcomes)

#find positivity rate for prediction and actual outcomes

#compute anomaly detection metrics: precision, recall, and f1 score
prec_score = precision_score(actual_outcomes, pred_outcomes, average = "binary")
rec_score = recall_score(actual_outcomes, pred_outcomes, average = "binary")
f_score = f1_score(actual_outcomes, pred_outcomes, average = "binary")

print("Precision Score: " + str(prec_score))
print("Recall Score: " + str(rec_score))
print("F1 Score: " + str(f_score))