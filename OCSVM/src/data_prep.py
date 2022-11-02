#This data preparation script reads a csv and does the following to prep the data for training:
#  1. Writes the dataset into LibSVM format
#  2. Splits the dataset into training and testing datasets randomly
#  3. Scales the data to make computation faster and to avoid having one factor dominate the others

#LibSVM format is:
#  <label> 1:<value1> 2:<value2> ... n:<valueN>
#  <label> 1:<value1> ...
#  ...
#  <label> ...

import os
from pathlib import Path
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def write_libsvm_file(filename, x, y, slice=1):
   output_file = open(filename, 'w')
   for index, row in x.iterrows():
      if y[index] == 0:
         row_string = "1"
      else:
         row_string = "-1"
      for count, value in enumerate(row[:-1]):
         row_string += f" {count}:{value}"
      row_string += "\n"
      in_slice = random.random()
      if (in_slice < testset_slice):
         output_file.write(row_string)

#data prep parameters
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

#scale data
Scaler = MinMaxScaler(copy = False)
Scaler.fit_transform(x_train)
Scaler.transform(x_test)

#create output folder if it doesn't already exist
try:
   os.mkdir(output_path)
except OSError:
   print("Output folder already exists.")

#write training data into LibSVM formatted files
testset_slice = 0.1
write_libsvm_file(train_file_name, x_train, y_train, testset_slice)
write_libsvm_file(test_file_name, x_test, y_test, testset_slice)
