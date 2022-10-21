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
    print('...Adding NrDependantsGiven column (default: 1, in time: 0)')
    df['NrOfDependants'] = df['NrOfDependants'].fillna('empty')
    df['NrOfDependants'] = df['NrOfDependants'].replace('10Plus', '11')
    NrDependantsGiven = []
    for row in df['NrOfDependants']:
        if row == 'empty' : NrDependantsGiven.append(1)
        else: NrDependantsGiven.append(0)
    df['NrDependantsGiven'] = NrDependantsGiven
    df['NrOfDependants'] = df['NrOfDependants'].replace('empty', '0')
    df['NrOfDependants'] = df['NrOfDependants'].astype(int)
    return df

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
    df_encoded = pd.concat([df_in.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop(['Country'], axis=1)
    return df_encoded

def oheLanguageCode(df):
    print('...preparing LanguageCode')

    # remove loans with language codes that are not valid
    df = df[df["LanguageCode"] != 22]
    df = df[df["LanguageCode"] != 15]
    df = df[df["LanguageCode"] != 21]
    df = df[df["LanguageCode"] != 10]
    df = df[df["LanguageCode"] != 13]
    df = df[df["LanguageCode"] != 7]
    map_dict = {
        1: "Estonian",
        2: "English",
        3: "Russian",
        4: "Finnish",
        5: "German",
        6: "Spanish",
        9: "Slovakian",
    }
    df["LanguageCode"] = df["LanguageCode"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['LanguageCode']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('LanguageCode', axis=1)
    return df_encoded

def oheVerificationType(df): 
    print('...OneHotEncoding VerificationType column')
    map_dict = {
        0: "NotSet", 
        1: "IncomeUnverified",
        2: "CrossRefferencedByPhone",
        3: "IncomeVerified",
        4: "IncomeExpensesVerified"
    }
    df["VerificationType"] = df["VerificationType"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['VerificationType']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('VerificationType', axis=1)
    return df_encoded

def oheUseOfLoan(df): 
    # remove loans for businesses which are no longer supported
    df = df[df["UseOfLoan"] != 102]
    df = df[df["UseOfLoan"] != 110]
    df = df[df["UseOfLoan"] != 108]

    map_dict = {
        -1: 'NoUseOfLoanGiven',
        0:'Loan consolidation',
        1:'Real estate',
        2:'Home improvement',
        3:'Business',
        4:'Education',
        5:'Travel',
        6:'Vehicle',
        7:'Other',
        8:'Health'
    }
    df["UseOfLoan"] = df["UseOfLoan"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['UseOfLoan']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('UseOfLoan', axis=1)
    return df_encoded
    
def prepareEmploymentDurationCurrentEmployer(df):
    print('...preparing EmploymentDurationCurrentEmployer')
    df['EmploymentDurationCurrentEmployer'] = df['EmploymentDurationCurrentEmployer'].fillna('empty')
    map_dict = {
        'MoreThan5Years': '5.5', 
        'UpTo5Years': "4.5",
        "TrialPeriod":"0.5",
        'UpTo1Year': "0.5",
        "UpTo2Years":"1.5",
        "UpTo3Years":"2.5",
        "UpTo4Years":"3.5",
        'Retiree': "5.5",
        'Other': "0.0",
        "empty":"0.0",
    }
    df["EmploymentDurationCurrentEmployer"] = df["EmploymentDurationCurrentEmployer"].map(map_dict)
    df[["EmploymentDurationCurrentEmployer"]] = df[["EmploymentDurationCurrentEmployer"]].apply(pd.to_numeric)
    df.reset_index(drop=True)
    return df

def prepareWorkExperience(df):
    print('...preparing WorkExperience')
    df['WorkExperience'] = df['WorkExperience'].fillna('empty')
    map_dict = {
        '15To25Years': '10.0', 
        '5To10Years': '7.5',
        "10To15Years":"12.5",
        'MoreThan25Years': "25.0",
        "2To5Years":"3.5",
        "LessThan2Years":"1.0",
        "empty": "0.0"
    }
    df["WorkExperience"] = df["WorkExperience"].map(map_dict)
    df[["WorkExperience"]] = df[["WorkExperience"]].apply(pd.to_numeric)
    df.reset_index(drop=True)
    return df


### EXECUTE CODE ########################################

def exportAsCSV(df):
    print('...exporting csv')
    df.head(2000).to_csv('Prepared_list.csv')


### EXECUTE CODE ########################################


addTargetColumn()
removeOngoingLoans()
removeIncompleteLines()

df_reduced = getReducedDataset()

df_reduced = prepareNrOfDependants(df_reduced)

df_reduced = oheCountry(df_reduced)
df_reduced = oheVerificationType(df_reduced)
df_reduced = oheLanguageCode(df_reduced)
df_reduced = oheUseOfLoan(df_reduced)

df_reduced = prepareEmploymentDurationCurrentEmployer(df_reduced)
df_reduced = prepareWorkExperience(df_reduced)


exportAsCSV(df_reduced)

# plotTargetDistributionForCol(df_reduced, 'Country')

