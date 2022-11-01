#This script changes a csv formatted file into the style desired by LibSVM

import csv
from pathlib import Path
import random

data_path = "./data"
csv_name = 'frequencies.csv'
csv_path = Path(data_path, csv_name)

train_file_name = Path(csv_path.parent, "freq_train") 
test_file_name = Path(csv_path.parent, "freq_test") 

test_train_split = 0.8
testset_slice = 0.1

with open(csv_path) as csv_file:
   train_file = open(train_file_name, 'w')
   test_file = open(test_file_name, 'w')
   row_number = 0
   csv_handle = csv.reader(csv_file)
   next(csv_handle)
   for row in csv_handle:
      if row[-1] == 0:
            row_string = "1"
      else:
         row_string = "-1"
      for count, value in enumerate(row[:-1]):
         row_string += f" {count}:{value}"
      row_string += "\n"
      test_or_train = random.random()
      in_slice = random.random()
      if (in_slice < testset_slice):
         if (test_or_train < test_train_split):
            train_file.write(row_string)
         else:
            test_file.write(row_string)