# Hospitalization-Risk
TrenData Analytics Team Project

This project is aimed at using several anomaly detection methods to determine the likelihood of an elderly client becoming hospitalized. We used Client Review Forms (CRFs) and Activities of Daily Living Forms (ADLs) to comprise our dataset, which are commonly used for keeping track of and ensuring elderly care.

Created as part of the UTD Senior Capstone Project elective.

## How to Use

For each model type, navigate to the respective folders and run the code inside according to the README. 

## Project Structure

* Gaussian Model: Contains all scripts to run a Univariate Gaussian model
* OCSVM: Contains scripts to run a One-Class SVM and find optimal hyperparameters
* KNN: Contains scripts to run a k-Nearest Neighbors and find optimal hyperparameters
* Data: Contains data used for the anomaly detection models, including de-indentified client appointments and the frequency of each ADL.
* CreateAppointments.py: This script generates `Appointments.csv` from the ADLs and CRFs, as well as provides visualizations like correlation matrices
* CreateFrequencies.py: This script generates `Frequencies.csv` from the ADLs and CRFs, as well as provides visualizations like correlation matrices

## Datasets

* OutreachAdls.csv: Combined csv of all Activities of Daily Living (ADLs)
* OutreachCrfs.csv: Combined csv of all Client Review Forms (CRFs)
* Appointments.csv: Data after being transformed by `CreateAppointments.iypnb`. Requires `OutreachAdls.csv` and `OutreachCrfs.csv`
* Frequencies.csv: Date after being transformed by `CreateFrequencies.ipynb`. Requires `Appointments.csv` and `OutreachCrfs.csv`
* FrequenciesExtra.csv: Appended version of `Frequencies.csv` with header info including Client, Caregiver, and Date

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

