# This script trains the One Class SVM and predicts labels and probabilities on testing data using LibSVM

#  Inputs:
#     Training Data prepared in LibSVM format (see data_prep.py)
#     Testing Data prepared in LibSVM format
#  Outputs:
#     ocsvm_preds:      Contains the OCSVM predictions for testing dataset, -1 means anomaly
#     ocsvm_probs:      Contains predicted outcomes for testing dataset, first number is normal chance, second is anomaly chance
#     ocsvm_true_test:  Contains the actual outcomes for the testing dataset, to be used to deteermine performance metrics

from libsvm.svmutil import *
from libsvm.svm import *
from pathlib import Path
import numpy as np

output_folder = "./output"

#input file names
training_data = "freq_train"
testing_data = "freq_test"

#output file names
predictions_file_name = "ocsvm_preds"
probs_file_name = "ocsvm_probs"
true_test_file_name = "ocsvm_true_test"

#read in testing and training data into LibSVM format
train_path = Path(output_folder, training_data)
test_path = Path(output_folder, testing_data)

y_train, x_train = svm_read_problem(train_path)
y_test, x_test = svm_read_problem(test_path)

train_prob = svm_problem(y_train,x_train)

#params:
#  -s:   Choose OCSVM model. 2 means One Class SVM
#  -b:   Choose to predict probability. 1 means predict probability
#  -d:   Degree of polynomial
#  -t:   Chooses kernel type (linear, poly, radial, sigmoid, and precomputed kernel)
#  -n:   nu hyperparameter, upper limit of incorrect labels, lower means less tolerance
#  -g:   gamma hyperparameter, determines similarity required to be in same class, higher means more curvature
#  -h:   Use shrinking heuristic or not
#  -wN:  Adds penalty weighting to the Nth classification class, starting at 0
#  -q:   quiet mode
params = svm_parameter('-s 2 -b 1 -d 5 -t 1 -h 0 -n 0.01 -g 0.025 -w1 200 -q')

#train and save model
model = svm_train(train_prob, params)

#Predict on testing data
#  p_labels holds the predictions for the test dataset
#  p_acc holds the accuracy percentage for the predictions
#  p_vals holds the probability predictions
p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model, '-b 1')

# for test_i in x_test:
#    test, max_idx = gen_svm_nodearray(test_i)
#    print(libsvm.svm_predict(model, test,"-b 1"))

#probability predicts using binning method, probabilities will be in intervals of 0.1
#write predictions and probabilities to file
preds_path = Path(output_folder, predictions_file_name)
np.savetxt(preds_path, p_labels, "%u")

probs_path = Path(output_folder, probs_file_name)
np.savetxt(probs_path, p_vals)

true_test_path = Path(output_folder, true_test_file_name)
np.savetxt(true_test_path, y_test, "%u")

