---
    title : "Gaussian Model Introduction"
    author: "Dhruva Mambapoor"
---

## How do I run the model?
1. **Add frequency data in a file called `frequencies.csv`.** 
    * *Note: First row and column are ignored; treated as header row and index column respectively;*
2. **Run `gaussian_train.py`** || This will train the model on the data-set above.
    * *Note: Make sure you have the following dependencies installed*
        * numpy
        * pandas
        * joblib
        * sklearn
        * matplotlib
3. **Run `gaussian_test.py`** || This will predict hospitalization with the data-set above and compare with the ground-truth.


## How does the model work?
See [`Gaussian Model Introduction`](Gaussian-Model-Introduction.md).

## What do these words mean?
See [`Gaussian Terminology Cheat Sheet`](Gaussian-Terminology-Cheat-Sheet.md)