# k-Nearest-Neighbors 

k-Nearest Neighbors (kNN) is a supervised classification machine learning model which classifies a new data point by looking at a number of classified data points nearest to the pooint and determining the majority class among those points.
This can be used for our use case as the outcome is known for all patients in our dataset. 
Intuitively, this model works by looking at similar instances of hospitalization agnostic of reason for hospitalization.

Among the models tested in this project, kNN consistenty scored the best with an F1 score of around 0.9 when optimized.

kNNs are able to predict hospitalization risk with a probability attached to it. However this requires a higher number of neighbors which the model makes the model perform worse.

## Dependencies
SciKit-Learn v1.1.3's `neighbors` module

## How to Run
Simply run the `KNN.ipynb` jupyter notebook script cell by cell to train and test the kNN model.

Optionally, there is the `KNN_hyperparams.ipynb` script which tests for the best hyperparameters using Grid Search and record the test results into a CSV file for reading. Run cell by cell to run the Grid Search.

## Bibliography
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=k+nearest+neighbors

## Credits
Wang Haoliang for suggesting the testing of this model.
