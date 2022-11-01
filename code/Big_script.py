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

# der gesamt 

TRAIN_TEST_RATIO = 0.3
NUMBER_OF_TREES = 100
RANDOM_STATE = 22

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
    
    # remove bad lines found above
    df_clean = df_clean.drop(incompleteRows)


    print(df_clean.shape)
    # remove loans with language codes that are not valid
    df_clean = df_clean[df_clean["LanguageCode"] != 22]
    df_clean = df_clean[df_clean["LanguageCode"] != 15]
    df_clean = df_clean[df_clean["LanguageCode"] != 21]
    df_clean = df_clean[df_clean["LanguageCode"] != 10]
    df_clean = df_clean[df_clean["LanguageCode"] != 13]
    df_clean = df_clean[df_clean["LanguageCode"] != 7]
    print(df_clean.shape)

    
    # remove loans for businesses which are no longer supported
    df_clean = df_clean[df_clean["UseOfLoan"] != 102]
    df_clean = df_clean[df_clean["UseOfLoan"] != 110]
    df_clean = df_clean[df_clean["UseOfLoan"] != 108]
    print(df_clean.shape)

    # remove incomplete empty EmploymentStatus area lines
    df_clean['EmploymentStatus'] = df_clean['EmploymentStatus'].fillna('empty')
    df_clean = df_clean[df_clean["EmploymentStatus"] != 0]
    df_clean = df_clean[df_clean["EmploymentStatus"] != 'empty']
    print(df_clean.shape)

    # remove incomplete empty Occupation area lines
    df_clean['OccupationArea'] = df_clean['OccupationArea'].fillna('empty')
    df_clean = df_clean[df_clean["OccupationArea"] != 'empty']
    print(df_clean.shape)
    
    # remove the incomplete lines from the dataset
    
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
        0: 'NoOccupationAreaGiven',
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
        19:'Agriculture',
    }
    df["OccupationArea"] = df["OccupationArea"].map(map_dict)
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
        -1:'No Home Ownership Type Provided',
        0:'Homeless',
        1:'Owner',
        2:'Living with Parents',
        3:'Tenant, Pre-furnished Property',
        4:'Tenant, unfurnished Property',
        5:'Council House',
        6:'Joint Tenant',
        7:'Joint Twnership',
        8:'Mortgage',
        9:'Owner with Encumbrance',
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

### PLOTS ###############################################

### PRINTS  #############################################

def printDtypes(df):
    print(df.dtypes)

def printValueDistribution(df, col):
    print(df[col].fillna('empty').value_counts())
    

### PLOT NOMINAL DISTRIBUTIONS #########################

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
    sns.kdeplot(df[df['TARGET']==1]['Age'], label='Repayment Late/Default', color='red')
    sns.kdeplot(df[df['TARGET']==0]['Age'], label='Repayment on Time', color='green')
    plt.axvline(x=df[df['TARGET']==1]['Age'].mean(),color='red', ls='--')
    plt.axvline(x=df[df['TARGET']==0]['Age'].mean(),color='green', ls='--')
    plt.grid()
    plt.legend()
    plt.show()


def PlotHeatMap(df):
    print('...plotting Heatmap')
    plt.figure(figsize=(100,15))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu', fmt='.2f')
    plt.show()

def printNumberOfUniqueValues(df):
    print('...printing number of unique valies per column')
    for column in df.columns:
        print("{}\t: {}".format(column, len(np.unique(df[column]))))

def printIsNullCounts(df):
    print('...filling monthly payment')
    print(df.isnull().sum())

### PLOT NUMERICAL VALUES ##############################

def plotAgeDistrinution(df): 
    print('...plotting age distribution')
    printValueDistribution(df, 'age')
    guys = df.query('Month == 7')
    guys.insert(0,'Yr',range(0,len(guys)))

### EXPORTS ############################################

def exportAsCSV(df, name):
    print('...exporting csv')
    df.head(2000).to_csv(name + '.csv')

### MODEL ##############################################

def getTrainRestSplit(df): 
    print('...splitting data into train/test sets')
    X = df.drop('TARGET', axis=1).values
    y = df['TARGET'].values
    return train_test_split(X, y, test_size=TRAIN_TEST_RATIO, random_state=22)

def normalizeData(X_train, X_test):
    print('...normalizing data')
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled

def trainModel(X_train_scaled, X_test_scaled, y_train, y_test):
    print('...training model')
    model = lgb.LGBMClassifier(n_estimators=NUMBER_OF_TREES, class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train, eval_metric='auc', 
            eval_set=[(X_train_scaled, y_train),(X_test_scaled, y_test)])
    return model

def testModel(model, X_train_scaled, X_test_scaled, y_train, y_test): 
    print('...testing model')

    # Predict the probability score
    prob_train = model.predict_proba(X_train_scaled)
    prob_test = model.predict_proba(X_test_scaled)

    # Create train and test curve
    fpr_train, tpr_train, thresh_train = roc_curve(y_train, prob_train[:,1])
    fpr_test, tpr_test, thresh_test = roc_curve(y_test, prob_test[:,1])

    # Create the straight line (how the graph looks like if the model does random guess instead)
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs)

    # Plot the model
    print('...plotting result')
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve')
    plt.plot(fpr_train, tpr_train, label='Train')
    plt.plot(fpr_test, tpr_test, label='Test')
    plt.plot(p_fpr, p_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


### EXECUTE CODE ########################################

# PREPARATION & CLEANUP
addTargetColumn()
removeOngoingLoans()
removeIncompleteLines()

# REMOVING UNUSABLE DATA
df_reduced = getReducedDataset()
df_reduced = prepareNrOfDependants(df_reduced)
df_reduced = prepareNewCreditCustomer(df_reduced)

# PLOT DISTRIBUTIONS 
plotAgeDistribution(df_reduced)


# ONE HOT ENCODING STUFF
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

# FILL INCOMPLETE CELLS WITH MEAN OF COLUMNS
fillMonthlyPayments(df_reduced)
fillPreviousRepaymentsBeforeLoan(df_reduced)
fillPreviousEarlyRepaymentsBefoleLoan(df_reduced)


# PRINTS
# printNumberOfUniqueValues(df_reduced)
# printIsNullCounts(df_reduced)

# PLOTS
# df_numeric = getDataForCorrelation(df_reduced)
# PlotHeatMap(df_numeric)
# plotTargetDistributionForCol(df_reduced, 'Country')

# EXPORTS
# exportAsCSV(df_reduced, 'reduced')
# exportAsCSV(df_numeric, 'numeric')


# Der nachforlgende Code basiert auf dem Medium Post 
# 'LightGBM on Home Credit Default Risk Prediction' von Muhammad Ardi (2020)
# https://medium.com/becoming-human/lightgbm-on-home-credit-default-risk-prediction-5b17e68a6e9

# TEST TRAIN SPLIT 
# X_train, X_test, y_train, y_test = getTrainRestSplit(df_reduced) 

# NORMALISATION
# X_train_scaled, X_test_scaled = normalizeData(X_train, X_test)

# TRAIN MODEL
# model = trainModel(X_train_scaled, X_test_scaled, y_train, y_test)

# TEST MODEL
# testModel(model, X_train_scaled, X_test_scaled, y_train, y_test)
