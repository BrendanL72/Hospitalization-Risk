# Hospitalization-Risk
TrenData Analytics Team Project

This project is aimed at using several anomaly detection methods to determine the likelihood of an elderly client becoming hospitalized. We used Client Review Forms (CRFs) and Activities of Daily Living Forms (ADLs) to comprise our dataset, which are commonly used for keeping track of and ensuring elderly care.

Created as part of the UTD Senior Capstone Project elective.

## How to Use

For each model type, navigate to the respective folders and run the code inside according to the README. 

NOTE: Some notebooks/scripts require the use of `FrequenciesExtra.csv`. However the file cannot be stored on Github as it is >100MB. As such, follow the link below to download it and insert it into the ./data folder.

https://drive.google.com/file/d/1V58RWwR3rDc8RZGOe5pRNs77AKNd1L3o/view

## Project Structure

* Gaussian Model: Contains all scripts to run a Univariate Gaussian model
* OCSVM: Contains scripts to run a One-Class SVM and find optimal hyperparameters
* KNN: Contains scripts to run a k-Nearest Neighbors and find optimal hyperparameters
* Data: Contains data used for the anomaly detection models, including de-indentified client appointments and the frequency of each ADL.
* CreateAppointments.py: This script generates `Appointments.csv` from the ADLs and CRFs, as well as provides visualizations like correlation matrices
* CreateFrequencies.py: This script generates `Frequencies.csv` from the ADLs and CRFs, as well as provides visualizations like correlation matrices

## Authors
Dhruva Mambapoor

Chris Zeller

Brendan Lim

Arjun Balasubramanian

Pramod Vivek

## Credits

Dr. Chen and Wang Haoliang for technical advice, direction, and oversight

TrenData and Chris Simonds for project management

Outreach Health for providing the data necessary for this project and providing the background information necessary to develop this project

