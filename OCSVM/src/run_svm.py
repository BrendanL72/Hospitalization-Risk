from libsvm.svmutil import *
from libsvm.svm import *
from pathlib import Path
import numpy as np
import scipy

output_folder = "./output"

#input file names
training_data = "freq_train"
testing_data = "freq_test"

#output file names
predictions_file_name = "ocsvm_preds"
probs_file_name = "ocsvm_probs"
true_test_file_name = "ocsvm_true_test"

#read in testing and training data
train_path = Path(output_folder, training_data)
test_path = Path(output_folder, testing_data)

y_train, x_train = svm_read_problem(train_path)
y_test, x_test = svm_read_problem(test_path)

print(type(y_train), type(x_train))

train_prob = svm_problem(y_train,x_train)
test_prob = svm_problem(y_test,x_test)

#params:
#  -s 2: Choose OCSVM model
#  -b 1: Choose to predict probability
#  -n:   nu hyperparameter, upper limit of incorrect labels, lower means less tolerance
#  -g:   gamma hyperparameter, determines similarity required to be in same class, higher means more curvature
#  -h:   Use shrinking heuristic or not
params = svm_parameter('-s 2 -b 1 -n 0.01 -g 0.0001 -h 0')

#train and save model
model = svm_train(train_prob, params)

p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model, '-b 1')

# for test_i in x_test:
#    test, max_idx = gen_svm_nodearray(test_i)
#    print(libsvm.svm_predict(model, test,"-b 1"))

#probability predicts using binning method, probabilities
#write predictions and probabilities to file
preds_path = Path(output_folder, predictions_file_name)
np.savetxt(preds_path, p_labels, "%u")

probs_path = Path(output_folder, probs_file_name)
np.savetxt(probs_path, p_vals)

true_test_path = Path(output_folder, true_test_file_name)
np.savetxt(true_test_path, y_test, "%u")

