#!/usr/bin/env python3
# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

# load loan data
df = pd.read_csv('LoanData.csv', low_memory=False)
df_clean = df.copy(deep=True)

### CLEAN DATA ##########################################

def addTargetColumn(): 
    print('...Adding TARGET column (default: 1, in time: 0)')
    TARGET = []
    for row in df['Status']:
        if row == 'Late' : TARGET.append(1)
        else: TARGET.append(0)
    
    df['TARGET'] = TARGET
    
def removeOngoingLoans():
    print('...removing Loans that are still active')
    global df_clean
    df_clean = df[df['Status'] != 'Current']
    
def removeIncompleteLines():
    print('...Removing incomplete lines')
    global df_clean
    susColumns = [
        'Education',
        'MaritalStatus',
        'DebtToIncome',
        'VerificationType',
        'FreeCash', 
        'Gender',
        'PreviousEarlyRepaymentsCountBeforeLoan',
        'AmountOfPreviousLoansBeforeLoan',
        'NoOfPreviousLoansBeforeLoan'
    ]
    
    # get the numbers of rows that have empty cells in the sus columns
    incompleteRows = []
    for col in susColumns:
        incompleteRows.extend(df_clean[df_clean[col].isna()].index)
    
    # remove duplicates
    incompleteRows = list(dict.fromkeys(incompleteRows))
    
    # remove the incomplete lines from the dataset
    df_clean = df_clean.drop(incompleteRows)
    
def getReducedDataset():
    print('...Reducing columns to relevant inputs')
    global df_clean 
    inputCols = [
        # APPLICATION
        'ApplicationSignedHour', 'ApplicationSignedWeekday', 'VerificationType', 
        # LOAN
        'AppliedAmount',
        # DEMOGRAPHIC
        'LanguageCode','Age','Gender','Country','LoanDuration', 'MonthlyPayment',  'UseOfLoan', 'Education', 
        'MaritalStatus', 'NrOfDependants', 'EmploymentStatus', 'EmploymentDurationCurrentEmployer', 'WorkExperience', 
        'OccupationArea', 'HomeOwnershipType', 
        # INCOME 
        'IncomeFromPrincipalEmployer','IncomeFromPension', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare', 
        'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther', 'IncomeTotal', 
        # LIABILITIES
        'ExistingLiabilities', 'LiabilitiesTotal', 'RefinanceLiabilities', 
        'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay', 
        # PREV LOANS
        'NewCreditCustomer', 'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan', 
        'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsBefoleLoan', 'PreviousEarlyRepaymentsCountBeforeLoan',

        'TARGET'
    ]
    df_reduced = df_clean.copy(deep=True)
    return df_reduced[inputCols]


### PLOT & PRINT STUFF ##################################

def plotTargetDistributionForCol(df, col): 
    print(pd.DataFrame({
        "Late": df[df['TARGET']==1][col].value_counts(),
        "On Time": df[df['TARGET']==0][col].value_counts()
    }))

def plotAgeDistribution(df):
    print('...Plotting age distribution')

    plt.figure(figsize=(10,8))
    plt.title('Age Distribution')
    plt.xlabel('Age')
    sns.kdeplot(df[df['TARGET']==1]['Age'], label='Target=1')
    sns.kdeplot(df[df['TARGET']==0]['Age'], label='Target=0')
    plt.grid()
    plt.show()
    
def printDtypes(df):
    print(df.dtypes)

def printValueDistribution(df, col):
    print()
    print(df[col].fillna('empty').value_counts())
    
def plotStuff(df): 
    printValueDistribution(df, 'Country')
    printValueDistribution(df, 'NrOfDependants')
    printValueDistribution(df, 'EmploymentDurationCurrentEmployer')


### FEATURE ENGINEERING #################################

def prepareNrOfDependants(df): 
    df['NrOfDependants'] = df['NrOfDependants'].fillna('empty')
    df['NrOfDependants'] = df['NrOfDependants'].replace('10Plus', '11')
    printValueDistribution(df, 'NrOfDependants')

def labelEncodeCountry(df): 
    print('...labelencoding Country column')
    labelencoder = LabelEncoder()
    df['Country'] = labelencoder.fit_transform(df['Country'])

def oheCountry(df_in): 
    print('...OneHotEncoding Country column')
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df_in[['Country']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    print('DF IN:    ', df_in.shape)
    print('FEATURES: ', features.shape)
    df_encoded = pd.concat([df_in.reset_index(drop=True), features.reset_index(drop=True)], axis=1, ignore_index=True)
    print('CONCATED: ', df_encoded.shape)

    df_encoded.head(5000).to_csv('LoanData_reduced.csv')
    return df_encoded


    



### EXECUTE CODE ########################################

print(df_clean.shape)

addTargetColumn()
removeOngoingLoans()
removeIncompleteLines()

df_reduced = getReducedDataset()
# prepareNrOfDependants(df_reduced)
# labelEncodeCountry(df_reduced)
df_reduced = oheCountry(df_reduced)
# plotTargetDistributionForCol(df_reduced, 'Country')

