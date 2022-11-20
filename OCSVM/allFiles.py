## This file will run all the code in order

exec(open("./src/data_prep.py").read()) ## runs the first file
exec(open("./src/run_svm.py").read()) ## runs the next file 
exec(open("./src/metrics.py").read()) ## runs the last file