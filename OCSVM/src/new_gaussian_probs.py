#This script converts the values output by running the svm scripts into probabilities using the new gaussian method
#  Inputs:
#     Output file generated from training the SVM
#     Output file generated from testing the SVM
#  Outputs:
#     Gamma probabilities for chance of hospitalization
#     Predictions based on which Gamma probabilities are significantly above the mean


from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma

# new_gamma scaling takes series of decision values and calculates the New Gamma Scaling according
#credit to Chih-Chung Chang and Chih-Jen Lin for making this function
def new_gamma_scaling(dec_values, gamma_params):
   mean, var, max_dec = gamma_params
   dec = [max_dec - dec_value if dec_value<max_dec else 0 for dec_value in dec_values]
   k = mean*mean/var
   theta = var/mean
   cdf = gamma.cdf(dec,a=k,scale=theta)
   probs = 1-cdf
   prob_dec0 = 1-gamma.cdf(max_dec,a=k,scale=theta)
   probs = np.where(probs>=prob_dec0, 0.5+0.5*(probs-prob_dec0)/(1-prob_dec0),0.5*probs/prob_dec0)
   return probs

#get_decval retrieves the decision values from the testing run of the SVM model
#credit to Chih-Chung Chang and Chih-Jen Lin for making this function
def get_decval(filename):
   data = []
   with open(filename) as f:
      for line in f:
         data.append([str(n) for n in line.strip().split('\t',5)])
   data_df = pd.DataFrame(data,columns=["label_platt","platt","dec_value","uniform","density","label"])
   dec=[float(i) for i in data_df.dec_value.drop(0).tolist()]
   return dec

#file paths
output_path = "./output"
train_output_file_name = "training_output"
train_output_path = Path(output_path, train_output_file_name)

test_output_file_name = "test_output"
test_output_path = Path(output_path, test_output_file_name)

gamma_file_name = "gamma_probs"
gamma_output_path = Path(output_path, gamma_file_name)

gamma_pred_file_name = "gamma_preds"
gamma_pred_path = Path(output_path, gamma_pred_file_name)

#get decision values
decval_df = get_decval(test_output_path)

#get the 3 gamma parameters from training output: mean, variance, and max
train_fhandle = open(train_output_path)
gamma_params = [0,0,0]
for row in train_fhandle:
   tokens = row.split()
   if tokens[0] == "dec_value":
      if "mean" in tokens[1]:
         gamma_params[0] = float(tokens[2])
      elif "variance" in tokens[1]:
         gamma_params[1] = float(tokens[2])
      elif "max" in tokens[1]:
         gamma_params[2] = float(tokens[2])

#determine probabilities using new gamma scaling and write to file
gamma_probs = new_gamma_scaling(decval_df, gamma_params)
# with open(gamma_output_path, 'w') as gamma_file:
#    gamma_file.write(gamma_probs)
np.savetxt(gamma_output_path, gamma_probs, fmt = "%.18f")

#predict outcome using gamma distribution and standard deviation
gamma_std = np.std(gamma_probs)
gamma_mean = np.mean(gamma_probs)
print("Mean Probability Prediction: " + str(gamma_mean))
print("Probability Prediction STDev: " + str(gamma_std))

#numpy wizardry filters a numpy array based on whether the probability is X*stdev above mean
#True/1 means that it predicts a hospitalization
#False/0 means no hospitalization
gamma_pred = gamma_probs > (gamma_mean + 1*gamma_std)
np.savetxt(gamma_pred_path, gamma_pred, "%u")
