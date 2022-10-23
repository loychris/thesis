#!/usr/bin/env python3
# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import math
from sklearn.feature_selection import f_regression
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

def getDataForCorrelation(df): 
    print("...Reducing DataSet to numeric columns")
    inputCols = [
        # APPLICATION
        "ApplicationSignedHour", 'ApplicationSignedWeekday', 

        # LOAN
        'AppliedAmount',

        # DEMOGRAPHIC
        'Age','LoanDuration', 'MonthlyPayment',
        'NrOfDependants', 'EmploymentDurationCurrentEmployer', 'WorkExperience', 

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
    printValueDistribution(df, 'WorkExperience')
    df_copy = df.copy(deep=True)
    return df_copy[inputCols]

def fillMonthlyPayments(df):
    print('...filling monthlyPaymnet')
    df.loc[:,'MonthlyPayment'] = df['MonthlyPayment'].fillna(df['MonthlyPayment'].mean())

def fillPreviousRepaymentsBeforeLoan(df):
    print('...filling PreviousRepaymentsBeforeLoan')
    df.loc[:,'PreviousRepaymentsBeforeLoan'] = df['PreviousRepaymentsBeforeLoan'].fillna(df['PreviousRepaymentsBeforeLoan'].mean())

def fillPreviousEarlyRepaymentsBefoleLoan(df):
    print('...filling PreviousEarlyRepaymentsBefoleLoan')
    df.loc[:,'PreviousEarlyRepaymentsBefoleLoan'] = df['PreviousEarlyRepaymentsBefoleLoan'].fillna(df['PreviousEarlyRepaymentsBefoleLoan'].mean())



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
    print(df[col].fillna('empty').value_counts())
    
def plotStuff(df): 
    printValueDistribution(df, 'Country')
    printValueDistribution(df, 'NrOfDependants')
    printValueDistribution(df, 'EmploymentDurationCurrentEmployer')

def PlotHeatMap(df):
    plt.figure(figsize=(100,15))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu', fmt='.2f')
    plt.show()

def printNumberOfUniqueValues(df):
    for column in df.columns:
        print("{}\t: {}".format(column, len(np.unique(df[column]))))

def printIsNullCounts(df):
    print('...filling monthly payment')
    print(df.isnull().sum())


### ONE_HOT_ENCODEING CATEGORIAL FEATURES ###############

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
    print('...OneHotEncoding LanguageCode column')

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
    print('...OneHotEncoding useOfLoan column')

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
        4:'EducationLoan',
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

def oheEducation(df):
    print('...OneHotEncoding Education column')
    map_dict = {
        -1: 'NoEducationGiven',
        0:'Loan consolidation',
        1:'Real estate',
        2:'Home improvement',
        3:'Business',
        4:'EducationLoan',
        5:'Travel',
    }
    df["Education"] = df["Education"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['Education']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('Education', axis=1)
    return df_encoded

def oheMaritalStatus(df):
    print('...OneHotEncoding MaritalStatus column')
    map_dict = {
        -1: 'NoMaritalStatusGiven',
        0:'Loan consolidation',
        1:'Married',
        2:'Cohabitant',
        3:'Single',
        4:'Divorced',
        5:'Widow',
    }
    df["MaritalStatus"] = df["MaritalStatus"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['MaritalStatus']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('MaritalStatus', axis=1)
    return df_encoded

def oheEmploymentStatus(df):
    print('...OneHotEncoding EmploymentStatus column')
    df['EmploymentStatus'] = df['EmploymentStatus'].fillna('empty')
    df = df[df["EmploymentStatus"] != 0]
    df = df[df["EmploymentStatus"] != 'empty']
    map_dict = {
        -1: 'NoEmploymentStatusGiven',
        1:'Unemployed',
        2:'Partially employed',
        3:'Fully employed',
        4:'Self-employed',
        5:'Entrepreneur',
        6:'Retiree'
    }
    df["EmploymentStatus"] = df["EmploymentStatus"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['EmploymentStatus']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('EmploymentStatus', axis=1)
    return df_encoded

def oheOccupationArea(df):
    print('...OneHotEncoding OccupationArea column')
    df['OccupationArea'] = df['OccupationArea'].fillna(-1)
    map_dict = {
        -1:'NoOccupationAreaGiven',
        1:'Other',
        2:'Mining',
        3:'Processing',
        4:'Energy',
        5:'Utilities',
        6:'Construction',
        7:'Retail and wholesale',
        8:'Transport and warehousing',
        9:'Hospitality and catering',
        10:'Info and telecom',
        11:'Finance and insurance',
        12:'Real-estate',
        13:'Research',
        14:'Administrative',
        15:'CivilService',
        16:'Education',
        17:'Healthcare',
        18:'ArtEntertainment',
        19:'Agriculture'
    }
    df["OccupationArea"] = df["OccupationArea"].map(map_dict)
    df['OccupationArea'] = df['OccupationArea'].fillna('empty')
    df = df[df["OccupationArea"] != 'empty']
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['OccupationArea']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('OccupationArea', axis=1)
    return df_encoded

def oheHomeOwnershipType(df): 
    print('...OneHotEncoding HomeOwnershipType column')
    map_dict = {
        -1:'NoHomeOwnershipTypeGiven',
        0:'Homeless',
        1:'Owner',
        2:'Living with parents',
        3:'Tenant, pre-furnished property',
        4:'Tenant, unfurnished property',
        5:'Council house',
        6:'Joint tenant',
        7:'Joint ownership',
        8:'Mortgage',
        9:'Owner with encumbrance',
        10:'Other'
    }
    df["HomeOwnershipType"] = df["HomeOwnershipType"].map(map_dict)
    df['HomeOwnershipType'] = df['HomeOwnershipType'].fillna('NoHomeOwnershipTypeGiven')

    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['HomeOwnershipType']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('HomeOwnershipType', axis=1)
    return df_encoded

def oheGender(df):
    print('...OneHotEncoding Gender column')
    df['Gender'] = df['Gender'].fillna('NoGenderGiven')
    map_dict = {
        0:'Male',
        1:'Female',
        2:'NoGenderGivens',
    }
    df["Gender"] = df["Gender"].map(map_dict)
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['Gender']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('Gender', axis=1)
    return df_encoded

### PREPARING NUMERIC FEATURES ##########################

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

def prepareNewCreditCustomer(df): 
    df["NewCreditCustomer"] = df["NewCreditCustomer"].astype(int)
    return df

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

def exportAsCSV(df, name):
    print('...exporting csv')
    df.head(2000).to_csv(name + '.csv')


### EXECUTE CODE ########################################


addTargetColumn()
removeOngoingLoans()
removeIncompleteLines()

# remove unusable data 
df_reduced = getReducedDataset()
df_reduced = prepareNrOfDependants(df_reduced)
df_reduced = prepareNewCreditCustomer(df_reduced)

# One Hot Encode Stuff 
df_reduced = oheCountry(df_reduced)
df_reduced = oheVerificationType(df_reduced)
df_reduced = oheLanguageCode(df_reduced)
df_reduced = oheUseOfLoan(df_reduced)
df_reduced = oheEducation(df_reduced)
df_reduced = oheMaritalStatus(df_reduced)
df_reduced = oheEmploymentStatus(df_reduced)
df_reduced = oheOccupationArea(df_reduced)
df_reduced = oheHomeOwnershipType(df_reduced)
df_reduced = oheGender(df_reduced)

df_reduced = prepareEmploymentDurationCurrentEmployer(df_reduced)
df_reduced = prepareWorkExperience(df_reduced)

printIsNullCounts(df_reduced)

fillMonthlyPayments(df_reduced)
fillPreviousRepaymentsBeforeLoan(df_reduced)
fillPreviousEarlyRepaymentsBefoleLoan(df_reduced)

# print stuff 
# printNumberOfUniqueValues(df_reduced)
printIsNullCounts(df_reduced)

# Plot stuff
df_numeric = getDataForCorrelation(df_reduced)
# PlotHeatMap(df_numeric)


# Export stuff 
exportAsCSV(df_reduced, 'reduced')
exportAsCSV(df_numeric, 'numeric')

# plotTargetDistributionForCol(df_reduced, 'Country')

