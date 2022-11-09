import pandas as pd
import numpy as np
#import seaborn as sns
import math
import os
import datetime
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Predetermined locations for files
rawAdlsDir = './data/outreachAdls.csv'
rawCrfsDir = './data/outreachCrfs.csv'

aptCsvLoc = './data/outreachAppointmentAdls.csv'
freqCsvLoc = './data/frequencies.csv'

### CELL 4
# Load CRFs and ADLs into memory
print('Loading sheets...')
adls = pd.read_csv(rawAdlsDir)
crfs = pd.read_csv(rawCrfsDir)

print(f'Load complete: {len(adls)} ADLs, {len(crfs)} CRFs')
###

### CELL 6
#CSP-4: Convert time data to object types

#Convert in CRFs
#Cols to DR: Time (y), Date of Review (y)
#Cols to Bool: Any Hospitalizations (n),
#Cols to 1 Hot: ? Which Review is this (n), ? Insurance (n), ?
print('Converting sheets to objects...\nDatetime...')
crfs['Date of Review'] = crfs['Date of Review'].transform(lambda x: datetime.datetime.strptime(x, "%d-%b-%y"))
crfs = crfs.drop(columns=['Time'])

#Convert in ADLs
#Cols: VisitDate, ActualTimeIn, ActualTimeOut, VisitDuration, SignatureDate
visitDurationRe = re.compile('^(?:(?P<hour>[0-9]+) h ?)?(?:(?P<minute>[0-9]+)m)?$')
def standardizeVisitDuration(durationStr):
  m = visitDurationRe.match(durationStr)
  return f"{m.group('hour') if m.group('hour') else 0} h {m.group('minute') if m.group('minute') else 0}m"

def parseTimedelta(string, template):
  time = datetime.datetime.strptime(string, template)
  return datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)

adls['VisitDate'] = adls['VisitDate'].transform(lambda x: datetime.datetime.strptime(x, '%m/%d/%y'))
adls['ActualTimeIn'] = adls['ActualTimeIn'].transform(lambda x: datetime.datetime.strptime(x, '%H:%M'))
adls['ActualTimeOut'] = adls['ActualTimeOut'].transform(lambda x: datetime.datetime.strptime(x, '%H:%M'))
#Format VisitDuration into '%H h %Mm' from '%H h' or '%H h %Mm' or '%Mm'
adls['VisitDuration'] = adls['VisitDuration'].transform(lambda x: parseTimedelta(standardizeVisitDuration(x), '%H h %Mm'))
#txtBreakTimesValue is all nan
adls['SignatureDate'] = adls['SignatureDate'].transform(lambda x: datetime.datetime.strptime(x, 'Signature Date: %m/%d/%Y %H:%M:%S %p') if len(x) != 16 else datetime.datetime(year=1, month=1, day=1))
adls = adls.drop(columns=['VisitDate2'])
###

### CELL 8
#CSP-4: Consolidate string data

# CRF Insurance
# Needs to be updated regularly so that insurance can be properly combined into a single value
# Update insurance string to group similarly written entries
print('Insurance...')
insuranceCorrections = {'^Other': 'Other:', ',': 'And', '^And': '', 'And$': '', 'Healthcare': 'Health Care', 'Ins': 'Insurance', 'Ddd': 'DDD', 'Uhc': 'UHC', 'Va': 'VA', 'United Health Care': 'UHC', 'VA Program': 'VA', 'Marai': 'Marpai', '^Other: UHC$': 'UHC'}

# Apply corrections specified above from string
def fromLooseInsuranceStr(form):
  # Capitalize beginning of each word
  ret = ' '.join([i[0].upper() + i[1:] for i in form.split(' ')])
  # Apply corrections from across insuranceCorrections list
  for k in insuranceCorrections:
    ret = re.sub(k, insuranceCorrections[k], ret)
  return ret

# Remove extranneous characters from string
def toLooseInsuranceStr(unform):
  # Remove non-alphanum and unacceptable punctuation
  unform = re.sub('[^A-Za-z0-9 _,]', '', unform)
  # Convert acceptible punctuation into spaces
  unform = re.sub('[ _,]+', ' ', unform)
  # Make lower case for compatibility
  unform = unform.lower().strip()
  return unform

def formatInsurance(ins):
  return fromLooseInsuranceStr(toLooseInsuranceStr(ins))

#Consolidates insurances given into similar
crfs['Insurance'] = crfs['Insurance'].transform(lambda x: formatInsurance(x))
#Consolidates singular insurance providers into an 'Other' category
crfs['Insurance'] = crfs['Insurance'].transform(lambda x: x if crfs['Insurance'].value_counts()[x] > 1 else 'Other')
#Combines DDD and DDD Ahcccs
crfs['Insurance'] = crfs['Insurance'].transform(lambda x: 'Other: DDD' if x == 'Other: DDD Ahcccs' else x)
###

### CELL 9
#CSP-4: Convert string data to objects
# Convert CRFs into machine readable format
print('Machine readable...')
def isYes(val):
  return 1 if val == 'Yes' else 0

crfs['Any Hospitalizations in the last 30 days'] = crfs['Any Hospitalizations in the last 30 days'].transform(isYes)
crfs['Matrix (Do you worry about falling?)'] = crfs['Matrix (Do you worry about falling?)'].transform(isYes)
crfs['Matrix (Do you have a Living Will?)'] = crfs['Matrix (Do you have a Living Will?)'].transform(isYes)
crfs['Matrix (Do you have a  DNR?)'] = crfs['Matrix (Do you have a  DNR?)'].transform(isYes)
crfs['Do you have a POA'] = crfs['Do you have a POA'].transform(isYes)
crfs['Do you need help with DNR, Living Will or POA'] = crfs['Do you need help with DNR, Living Will or POA'].transform(isYes)
crfs['Are there any changes to your home environment since the last Intake/Review?'] = crfs['Are there any changes to your home environment since the last Intake/Review?'].transform(isYes)
crfs['Has anyone moved in our out in the last 30 days?'] = crfs['Has anyone moved in our out in the last 30 days?'].transform(isYes)
crfs['Are you receiving Home Health Services'] = crfs['Are you receiving Home Health Services'].transform(isYes)
crfs['Are you expecting a Nurse or Therapist to visit you at home?'] = crfs['Are you expecting a Nurse or Therapist to visit you at home?'].transform(isYes)
crfs['Any changes to Transportation needs?'] = crfs['Any changes to Transportation needs?'].transform(isYes)
crfs['Have you had any new DME equipment in the last 90 days?'] = crfs['Have you had any new DME equipment in the last 90 days?'].transform(isYes)
crfs['Do you need any DME Equipment?'] = crfs['Do you need any DME Equipment?'].transform(isYes)
crfs['Any new medical devises in the past 90 days'] = crfs['Any new medical devises in the past 90 days'].transform(isYes)
crfs['Any changes to Mental Status'] = crfs['Any changes to Mental Status'].transform(isYes)
crfs['Any new medical conditions in the last 90 days'] = crfs['Any new medical conditions in the last 90 days'].transform(isYes)
crfs['Have you had your flu shot this year?'] = crfs['Have you had your flu shot this year?'].transform(isYes)
crfs['Does your health plan meet your needs?'] = crfs['Does your health plan meet your needs?'].transform(isYes)
crfs['Has the Care Plan been reviewed with the Client'] = crfs['Has the Care Plan been reviewed with the Client'].transform(isYes)
crfs['Any changes to current care plan'] = crfs['Any changes to current care plan'].transform(isYes)
crfs['Does your Caregiver carry out the care plan satisfactory'] = crfs['Does your Caregiver carry out the care plan satisfactory'].transform(isYes)
crfs['Are the cargivers able to perform all the tasks on your care plan'] = crfs['Are the cargivers able to perform all the tasks on your care plan'].transform(isYes)
crfs['Are there any upcoming schedule changes?'] = crfs['Are there any upcoming schedule changes?'].transform(isYes)
crfs['Are you happy with the services being provided?'] = crfs['Are you happy with the services being provided?'].transform(isYes)
###

### CELL 10
#Convert CRFs to one-hot and remove free response data
toDrop = ['Note on Hospitalization (date and reason)', 'How can we help prevent a return to hospital?', 'Note on help needed?',
          'If so from which Agency', 'What will the Nurse/Therapist do?', 'Notes on transportation needs',
          'Notes on new DME equipment', 'Notes on DME Equipment needed', 'Notes on new medical devises',
          'Notes on changes to mental status', 'Notes on changes to medical conditions', 'Where and when did you have your flu shot?',
          'Date and Place of last HbA1C Check', 'Date and place of last Mammogram (if female)',
          'Date and place of last Colorectal Exam', 'Date and place of last time you had your cholesterol checked',
          'Date and place of last Bone Density Scan', 'Notes on changes to Care Plan', 'Details of timekeeping issues',
          'Detail any issues', 'Detail any issues.1', 'Details of Schedule changes', 'If No make notes here',
          'Any Questions or Concerns?', 'Case Manager Notes']
to1Hot = ['COVID -19', 'Which Review is this', 'Insurance', 'Have you fallen in:', 'Did your fall result in',
          'Condition of the Home', 'How is your vision?', 'How is your hearing', 'Do you take your medications',
          'Do you fill all your prescriptions?', 'How would you describe your diet',
          'How much water do you drink in a day?', 'How do you sleep',
          'What do you do about your medical appointments', 'Do your caregivers arrive on time?']
crfs = crfs.drop(columns=toDrop)
crfs = pd.get_dummies(crfs, columns=to1Hot)
###

### CELL 11
# Convert 'HasBeenDone' and 'Reason' into a column 'Performed' based on whether
# an action has been performed a unique time since the last visit
doneReasons = ['Care Provided By Family or Friend or Self']
adls.insert(14, 'Performed', ((adls['HasBeenDone'] == 'Y') | (adls['Reason'].transform(lambda x: x in doneReasons))))
###

### CELL 12
# Convert adls into machine readable format

#Remove adls with no tasks
adls = adls.loc[adls['TaskCodeGroupName'].transform(lambda x: isinstance(x, str))]
#Replace nan id with 0
adls['ProviderID'] = adls['ProviderID'].transform(lambda x: 0 if np.isnan(x) else x)

adls['HasBeenDone'] = adls['HasBeenDone'].transform(lambda x: 1 if x == 'Y' else 0)
adls['Performed'] = adls['Performed'].transform(lambda x: 1 if x else 0)
adlDropCols = ['PayerGroup', 'VisitDuration', 'txtBreakTimesValue', 'ScheduledTimeIn',
               'ScheduledTimeOut', 'Comments', 'NoSignature', 'SignatureRefusalReason',
               'RectangleVoiceCaptured', 'SignatureDate']
adlTo1Hot = ['ProviderID', 'ServiceTypeName', 'TaskCodeGroupName', 'Reason']
adls = adls.drop(columns=adlDropCols)
adls = pd.get_dummies(adls, columns=adlTo1Hot)
print('Complete')
###


### CELL 13
# Functions to remove identifying information that is incompatible with the model
def deIdCrf(crfs):
  return crfs.drop(columns=['Masked Client ID', 'Date of Review'])

def deIdAdl(adls):
  return adls.drop(columns=['DeIdentify ID', 'CaregiverID', 'VisitDate', 'ActualTimeIn', 'ActualTimeOut'])
###

### CELL 15
# Helper functions for performing operations on the datasets

# Make sure to apply the proper subset to data or else there will be an excess of entries and unintended behavior
def applyDiffToEntries(data, ref, compFunc, colName):
  mtData = data.copy()
  conds = mtData.apply(lambda r: compFunc(ref, r), axis=1)
  data.insert(0, colName, conds)
  return data

# Iterates the entries of a list from oldest to recent
def iterateEntriesSorted(compDf, sortCol, **kwargs):
  compDf = compDf.sort_values(by=sortCol)
  #Find least positive
  nexPoss = compDf[sortCol]
  if(len(nexPoss) > 0):
    r = None
    c = None
    for i in nexPoss.index:
      v = nexPoss[i]
      if r is None or v != c:
        if r is not None:
          if 'drop' in kwargs and kwargs['drop']:
            yield compDf.loc[r].drop(columns=sortCol)
          else:
            yield compDf.loc[r]
        r = [i]
        c = v
      else:
        r.append(i)
    if 'drop' in kwargs and kwargs['drop']:
      yield compDf.loc[r].drop(columns=sortCol)
    else:
      yield compDf.loc[r]
  else:
    return None

# Iterate all entries by compFunc
def findAllEntries(data, compFunc):
  sortData = applyDiffToEntries(data, data.iloc[0], compFunc, 'CompareValue')
  return iterateEntriesSorted(sortData, 'CompareValue', drop=True)

# Iterate entries from the specified entry by compFunc
def findNextEntries(entry, data, compFunc):
  sortData = applyDiffToEntries(data, entry, compFunc, 'CompareValue')
  return iterateEntriesSorted(sortData.loc[sortData['CompareValue'] > (0, 0)], 'CompareValue', drop=True)

# Creates a tuple of the difference of dates
# (inDateDiff, outTimeDiff)
def adlDateDiff(adl1, adl2):
  #return ((adl2['VisitDate'] - adl1['VisitDate']).days * 24 * 60 * 60 + (adl2['ActualTimeIn'] - adl1['ActualTimeIn']).seconds) * 1000 + ((adl2['ActualTimeOut'] - adl1['ActualTimeOut']).seconds/60)
  return ((adl2['VisitDate'] - adl1['VisitDate']).days * 24 * 60 * 60 + (adl2['ActualTimeIn'] - adl1['ActualTimeIn']).seconds, (adl2['ActualTimeOut'] - adl1['ActualTimeOut']).seconds/60)

# Wrapper func to iterate adlSub by date diff
def iterateAdls(adlSub):
  return findAllEntries(adlSub, adlDateDiff)

# Wrapper func to iterate adlSub from adlRow to older by date diff
def previousAdls(adlRow, adlSub):
  return findNextEntries(adlRow, adlSub, lambda a, b: adlDateDiff(b, a))

# Wrapper func to iterate adlSub from adlRow to newer by date diff
def nextAdls(adlRow, adlSub):
  return findNextEntries(adlRow, adlSub, adlDateDiff)

# Return ADLs in the same appointment (same: client, caregiver, visitdate, timein, timeout)
def sameAptAdls(adlRow, adlSub):
  # Subdivided into 2 locs to improve performance
  sub = adlSub.loc[(adlSub['DeIdentify ID'] == adlRow['DeIdentify ID']) &
                    (adlSub['CaregiverID'] == adlRow['CaregiverID'])]
  return sub.loc[(sub['VisitDate'] == adlRow['VisitDate']) &
                    (sub['ActualTimeIn'] == adlRow['ActualTimeIn']) &
                    (sub['ActualTimeOut'] == adlRow['ActualTimeOut'])]
###

### CELL 16
#CSP-3: Get ADLs for all CRFs

#Section above needs to be integrated into the current section

#print(targetCrf['Masked Client ID'])
#print(targetCrf)

# Helper function to get the previous CRF
def localMostRecent(crfRow, crfsSub):
  # Select all CRFs older than crfRow
  sub = crfsSub.loc[crfsSub['Date of Review'] < crfRow['Date of Review']]
  # Choose newest of the older
  return sub['Date of Review'].max() if len(sub) > 0 else None

# Gets previous adls that pertain to a given CRF
def getAdlsFromCrf(crfRow, adlSub, **kwargs):
  prevCrf = localMostRecent(crfRow, crfs.loc[crfs['Masked Client ID'] == crfRow['Masked Client ID']])
  if('print' in kwargs and kwargs['print']):
    if(prevCrf is not None):
      print(f'CRF on {crfRow["Date of Review"]} and previous CRF on {prevCrf["Date of Review"]}')
    else:
      print(f'CRF on {crfRow["Date of Review"]}')
  # Select ADLs that are the same client 30 days after the crfRow
  # Separate lines to speed up computation
  filterAdls = adlSub.loc[adlSub['DeIdentify ID'] == crfRow['Masked Client ID']]
  filterAdls = filterAdls.loc[(crfRow['Date of Review'] - filterAdls['VisitDate']).dt.days > 30]
  # If previous CRF exists, select adls that are after the previous CRF
  if prevCrf is not None:
    filterAdls = filterAdls.loc[filterAdls['VisitDate'] >= prevCrf['Date of Review']]
  return filterAdls

# Function to get all adls in a series for df crfSub
def adlsFromCrfs(crfSub, adlSub):
  crfAssocAdls = []
  for i, r in crfSub.iterrows():
    crfAssocAdls.append(getAdlsFromCrf(r, adlSub[adlSub['DeIdentify ID'] == r['Masked Client ID']]))
  crfAssocAdls = pd.Series(index=crfSub.index, data=crfAssocAdls)

  return crfAssocAdls.loc[crfAssocAdls.transform(lambda x: not x.empty)]
###

### CELL 17
# Compresses multiple adl rows into a row pertaining to a unique visit
def compressAdlApt(apt):
  # No need to copy apt.iloc[0] since comb is overwritten
  comb = apt.iloc[0]
  cInd = pd.Series(data=comb.index)
  # Drop reason columns, use performed column created above instead
  comb = comb.drop(cInd.loc[cInd.transform(lambda x: x.startswith('Reason'))])
  # Regenerate index after dropping
  cInd = pd.Series(data=comb.index)
  # Select only ADLs that are performed
  pSub = apt.loc[apt['Performed'] == 1]
  # Condense the one-hot functions (denoted with a '_' in them) using any
  for i in cInd.loc[cInd.transform(lambda x: '_' in x)].values:
    # Condense using any, convert to 1/0 for T/F
    comb[i] = 1 if pSub[i].any() else 0
  # Drop columns that don't make sense in an appointment format
  return comb.drop(['HasBeenDone', 'Performed'])
###

### CELL 18
#Turn adls into appointments

print('Compress ADLs into appointments...')
apts = []
for c in tqdm(sorted(adls['DeIdentify ID'].unique())):
  sub = adls.loc[adls['DeIdentify ID'] == c]
  #print(f'{c}: {len(sub)}')
  # Iterate ADLs for each client by appointment and compress into single row appointment
  for i in iterateAdls(sub):
    apts.append(compressAdlApt(i))
# Convert apts to df
# Must have at least 1 appointment
  # 0 appointments > 0 ADLs for client > client not in unique client list
compAdls = pd.DataFrame(columns=apts[0].index, data=apts)
print(f'Appointments: {len(compAdls)}')
###

### CELL 19
#Find invalid CRFs and ADLs and remove
"""
allAdls = adlsFromCrfs(crfs, compAdls)
validCrfs = crfs.loc[allAdls.index]
acc = pd.DataFrame(columns=adls.columns)
for df in allAdls:
  acc = acc.merge(df, how='outer')
validCompAdls = acc
print(f'Valid Comp ADLs: {len(validCompAdls)}, Valid CRFs: {len(validCrfs)}')
"""
###

### CELL 20
#Mask ADLs to if has valid CRF
"""
print(f'Pre: {len(adls)}')
adls = adls.loc[adls['DeIdentify ID'].transform(lambda x: x in crfClis)]
print(f'Post: {len(adls)}')
"""
###

### CELL 28
# Write appointments to csv
compAdls.reset_index(inplace = True)
print(f'Write appointments to csv at {aptCsvLoc}')
compAdls.to_csv(aptCsvLoc, index=False, header=True)
###

### BEGIN CREATE FREQUENCIES NOTEBOOK

### CELL 2
#adls = pd.read_csv("outreachAppointmentAdls.csv")
#crfs = pd.read_csv("ClientReviewFormRemovingData.csv")
print('Add columns for frequencies...')
freqAdls = compAdls.copy()
freqAdls['Date'] = pd.to_datetime(freqAdls['VisitDate']).dt.date
crfs['Date'] = pd.to_datetime(crfs['Date of Review']).dt.date
###

### CELL 3
def breakDown(x,y,monthsBefore, repeatData, noCrfsIncluded, cushion):
    dayLen = monthsBefore * 30
    lastDate = max(x['Date'])
    xs = []
    ys = []
    for i, row in x.iterrows():
        endFrame = row['Date'] + datetime.timedelta(days= dayLen)
        if endFrame > lastDate:
            break
        currDf = x[(x['Date'] >= row['Date']) & (x['Date'] <= endFrame)]
        hasHospitilizations = len(y[(y['Date'] <= endFrame + datetime.timedelta(days = 30 + cushion)) & (y['Date'] >= endFrame + datetime.timedelta(days = 30 - cushion)) & (y['Any Hospitalizations in the last 30 days'] == 'Yes')]) > 0
        xs.append(currDf)
        ys.append(int(hasHospitilizations))
    return xs,ys
###

### CELL 4
## Code to create datasets
## options
print('Get subsets to convert to frequencies...')
monthsBefore = 2 ## number of months used for prediction
repeatData = True ## if patient has a positive in march we use data from feb and march to correspond to
noCrfsIncluded = False ## if the patient has no adls do we treat it as no hospitilizations or omit them
cushion = 7 ## number of days for the end of the frame to result in hospitilization


crfsDf = crfs
adlsDf = freqAdls
crfsID = 'Masked Client ID'
adlsID = 'DeIdentify ID'
crfsDate = 'Date'
adlsDate = 'Date'


IDDfs = {}
xDfs = []
yVals = []
numpys ={}
count = 0
for clientId in tqdm(adlsDf[adlsID].unique()):
    tempx = adlsDf[adlsDf[adlsID] == clientId].sort_values(by = adlsDate)
    tempy = crfsDf[crfsDf[crfsID] == clientId].sort_values(by = crfsDate)
    IDDfs[clientId] = (tempx,tempy)
    x,y = breakDown(tempx,tempy,monthsBefore, repeatData, noCrfsIncluded, cushion)
    xDfs += x
    yVals += y
    count += 1
###

### REHASH CELL 13 FROM APPOINTMENTS NOTEBOOK
def deIdCrf(crfs):
  return crfs.drop(columns=['Masked Client ID', 'Date of Review', 'Date'])

def deIdAdl(adls):
  return adls.drop(columns=['DeIdentify ID', 'CaregiverID', 'VisitDate', 'ActualTimeIn', 'ActualTimeOut', 'Date'])
###

### CELL 6
print('Convert to frequencies...')
freqDf = pd.DataFrame()
for i in tqdm(range(len(xDfs))):
    row = deIdAdl(xDfs[i]).sum(axis =  0)

    ## get the quarters -- you can comment out if you don't want
    startDate = min(xDfs[i]['Date'])
    endDate = max(xDfs[i]['Date'])
    q1end = startDate + datetime.timedelta(days = 15)
    q2end = q1end + datetime.timedelta(days = 15)
    q3end = q2end + datetime.timedelta(days = 15)

    row = row.append(deIdAdl(xDfs[i][(xDfs[i]['Date'] >= startDate) &(xDfs[i]['Date'] <= q1end)]).sum(axis =  0), ignore_index=True)
    row = row.append(deIdAdl(xDfs[i][(xDfs[i]['Date'] >= q1end) &(xDfs[i]['Date'] <= q2end)]).sum(axis =  0), ignore_index=True)
    row = row.append(deIdAdl(xDfs[i][(xDfs[i]['Date'] >= q2end) &(xDfs[i]['Date'] <= q3end)]).sum(axis =  0), ignore_index=True)
    row = row.append(deIdAdl(xDfs[i][(xDfs[i]['Date'] >= q3end) &(xDfs[i]['Date'] <= endDate)]).sum(axis =  0), ignore_index=True)

     ## get the standard deviations --- you cant comment out if you don't want
    stds =[]
    for col in deIdAdl(xDfs[i]):
        stds.append(deIdAdl(xDfs[i])[col].std())
    row.append(pd.Series(stds))

    row = row.append(pd.Series(yVals[i], index = ['hasHospitilization']))

    freqDf = freqDf.append(row, ignore_index = True)
###

### CELL 7
# Write frequencies to CSV
print(f'Write appointments to csv at {freqCsvLoc}')
freqDf.to_csv(freqCsvLoc, index = True)
###
