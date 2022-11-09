#This data preparation script reads a csv and does the following to prep the data for training:
#  1. Writes the dataset in a LibSVM-friendly format
#  2. Splits the dataset into training and testing datasets randomly
#  3. Scales the data to make computation faster and to avoid having one factor dominate the others

# LibSVM/Pandas format is:
#  <label> 1:<value1> 2:<value2> ... n:<valueN>
#  <label> 1:<value1> ...
#  ...
#  <label> ...
#
# Where the hospitalization outcome <labels> are:
#     1 - Normal
#    -1 - Anomaly
#
# Hospitalization Outcome symbols:
#     0 - Not hospitalized
#     1 - Hospitalized

import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#write_libsvm_file takes input and outcome data and writes it into Pandas to_string format
#SHOULD REWRITE TO USE PANDAS TO_STRING FUNCTION INSTEAD
def write_libsvm_file(filename, x, y):
   output_file = open(filename, 'w')
   for index, row in x.iterrows():
      if y[index] == 0:
         row_string = "1"
      else:
         #anomaly
         row_string = "-1"
      for count, value in enumerate(row[:-1]):
         row_string += f" {count}:{value}"
      row_string += "\n"
      output_file.write(row_string)

#file name parameters
data_path = "../data"
output_path = "./output"

csv_name = 'frequencies.csv'
csv_path = Path(data_path, csv_name)

train_file_name = Path(output_path, "freq_train") 
test_file_name = Path(output_path, "freq_test") 


df = pd.read_csv(csv_path)

#remove index and get outcome from csv
df = df.drop("Unnamed: 0", axis = 1)
y = df['hasHospitilization']
x = df.drop(['hasHospitilization'], axis=1)

#split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#make dataset smaller
scale_factor = 1
x_train = x_train.iloc[0:math.floor(x_train.shape[0]*scale_factor),:]
x_test = x_test.iloc[0:math.floor(x_test.shape[0]*scale_factor),:]
y_train = y_train.iloc[0:math.floor(y_train.shape[0]*scale_factor)]
y_test = y_test.iloc[0:math.floor(y_test.shape[0]*scale_factor)]

#scale data from 0 to 1 to avoid domination of one factor
scaler = MinMaxScaler(copy = False)
scaler.fit_transform(x_train)
scaler.transform(x_test)

#create output folder if it doesn't already exist
try:
   os.mkdir(output_path)
except OSError:
   print("Output folder already exists.")

#write data into LibSVM formatted files
write_libsvm_file(train_file_name, x_train, y_train)
write_libsvm_file(test_file_name, x_test, y_test)

