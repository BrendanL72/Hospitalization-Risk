#!/bin/bash
output_dir=./output

mkdir -p ${output_dir}

echo "Training..." 
./src/OCSVM/svm-train -s 2 -b 1 -n 0.25 -g 0.0001 -h 0 ${output_dir}/freq_train > ${output_dir}/training_output

echo "Predicting..." 
./src/OCSVM/svm-predict -b 1 ${output_dir}/freq_test ./freq_train.model ${output_dir}/test_output