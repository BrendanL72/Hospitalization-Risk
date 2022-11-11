## This file will run all the code in order

exec(open("./data_prep.py").read()) ## runs the first file
exec(open("./metrics.py.py").read()) ## runs the next file 
exec(open("./new_gaussian_robs.py").read()) ## runs the last file