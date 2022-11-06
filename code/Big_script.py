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
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report


# der gesamt 

TRAIN_TEST_RATIO = 0.1 # 0.05 0.1 0.2 0.3 
NUMBER_OF_TREES = 100 # 100 500 1000 10000
RANDOM_STATE = 22
NUMBER_OF_LEAVES = 31 # 10 20 30 40 50 
MAX_TREE_DEPTH = 5 # 3 5 7 

# preventin machine to crash 
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

# load loan data
df = pd.read_csv('LoanData.csv', low_memory=False)
df_clean = df.copy(deep=True)


### CLEAN DATA ##########################################

def addDefaultColumn(df): 
    print('...Adding DEFAULT column (default: 1, in time: 0)')
    DEFAULT = []
    for row in df['Status']:
        if row == 'Late' : DEFAULT.append(1)
        else: DEFAULT.append(0)
    
    df['DEFAULT'] = DEFAULT
    return df 

def addFreeCashToAmountColumn(df): 
    print('...Adding FreeCashToAmountColumn column')
    FreeCashToAmountColumn = []
    df = df.reset_index()  # make sure indexes pair with number of rows
    count = (df['FreeCash'] == 0).sum()
    print('ZEROOOOOOOOOOO', count)
    for index, row in df.iterrows():
        FreeCashToAmountColumn.append(row['FreeCash'] / row['AppliedAmount'])
    df['FreeCashToAmountColumn'] = FreeCashToAmountColumn
    return df 

def addFreeCashToMonthlyPaymentColumn(df): 
    print('...Adding FreeCashToMonthlyPaymentColumn column')
    FreeCashToMonthlyPaymentColumn = []
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        FreeCashToMonthlyPaymentColumn.append(row['FreeCash'] / row['MonthlyPayment'])
    df['FreeCashToMonthlyPaymentColumn'] = FreeCashToMonthlyPaymentColumn
    return df 


def removeOngoingLoans(df):
    print('...removing Loans that are still active')
    df = df[df['Status'] != 'Current']
    return df

def getReducedDataset(df):
    print('...Reducing columns to relevant inputs')
    inputCols = [
        # APPLICATION
        'VerificationType', 
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
        'DebtToIncome', 'FreeCash', 
        # PREV LOANS
        'NewCreditCustomer', 'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan', 
        'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsBefoleLoan', 'PreviousEarlyRepaymentsCountBeforeLoan',

        'DEFAULT'
    ]
    return df[inputCols]


def removeIncompleteLines(df):
    print('...Removing incomplete lines')

    # columns that showed the same small number of empty cells in explorative analysis
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
    
    # get the indices of rows that have empty cells in the sus columns
    incompleteRows = []
    for col in susColumns:
        incompleteRows.extend(df[df[col].isna()].index)
    
    # remove duplicates
    incompleteRows = list(dict.fromkeys(incompleteRows))
    
    # remove bad lines found above
    df = df.drop(incompleteRows)

    # remove loans with language codes that are not valid
    df = df[df["LanguageCode"] != 22]
    df = df[df["LanguageCode"] != 15]
    df = df[df["LanguageCode"] != 21]
    df = df[df["LanguageCode"] != 10]
    df = df[df["LanguageCode"] != 13]
    df = df[df["LanguageCode"] != 7]

    # removing Loans with Monthly payments equal to 0
    df = df[df["MonthlyPayment"] != 0]

    # remove incomplete empty EmploymentStatus area lines
    df['EmploymentStatus'] = df['EmploymentStatus'].fillna('empty')
    df = df[df["EmploymentStatus"] != 0]
    df = df[df["EmploymentStatus"] != 'empty']

    # remove incomplete empty Occupation area lines
    df['OccupationArea'] = df['OccupationArea'].fillna('empty')
    df = df[df["OccupationArea"] != 'empty']
    return df

def removeBusinessLoans(df):
    # remove loans for businesses which are no longer supported since 2012
    print('... removing business loans')
    df = df[df["UseOfLoan"] != 102]
    df = df[df["UseOfLoan"] != 110]
    df = df[df["UseOfLoan"] != 108]
    return df
        
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
        'DebtToIncome', 'FreeCash',  

        # PREV LOANS
        'NewCreditCustomer', 'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan', 
        'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsBefoleLoan', 'PreviousEarlyRepaymentsCountBeforeLoan',

        'DEFAULT'
    ]
    printValueDistribution(df, 'WorkExperience')
    df_copy = df.copy(deep=True)
    return df_copy[inputCols]




### TRANSLATE VALUES ####################################

def translateLanguageCode(df):
    print('...translating LanguageCode')
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
    return df

def translateVerificationTyle(df): 
    print('...translating VerificationType')
    map_dict = {
        0: "NotSet", 
        1: "IncomeUnverified",
        2: "CrossRefferencedByPhone",
        3: "IncomeVerified",
        4: "IncomeExpensesVerified"
    }
    df["VerificationType"] = df["VerificationType"].map(map_dict)
    return df

def translateUseOfLoan(df):
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
    return df 

def translateEducation(df): 
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
    return df

def translateMaterialStatus(df): 
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
    return df 

def translateEmploymentStatus(df): 
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
    return df

def translateOccupationArea(df): 
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
    return df

def translateHomeOwnershipType(df): 
    map_dict = {
        -1:'No Home Ownership Type Provided',
        0: 'Homeless',
        1: 'Owner',
        2: 'Living with Parents',
        3: 'Tenant, Pre-furnished Property',
        4: 'Tenant, unfurnished Property',
        5: 'Council House',
        6: 'Joint Tenant',
        7: 'Joint Ownership',
        8: 'Mortgage',
        9: 'Owner with Encumbrance',
        10:'Other'
    }
    df["HomeOwnershipType"] = df["HomeOwnershipType"].map(map_dict)
    df['HomeOwnershipType'] = df['HomeOwnershipType'].fillna('NoHomeOwnershipTypeGiven')
    return df

def translateGender(df): 
    df['Gender'] = df['Gender'].fillna('NoGenderGiven')
    map_dict = {
        0:'Male',
        1:'Female',
        2:'NoGenderGivens',
    }
    df["Gender"] = df["Gender"].map(map_dict)
    return df



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




### PRINTS  #############################################

def printDtypes(df):
    print(df.dtypes)

def printValueDistribution(df, col):
    print(df[col].fillna('empty').value_counts())

def printDistribution(df, col): 
    print('... printing DEFAULT distribution for', col)
    print(pd.DataFrame({
        "Late": df[df['DEFAULT']==1][col].value_counts(),
        "On Time": df[df['DEFAULT']==0][col].value_counts()
    }))

def printNumberOfUniqueValues(df):
    print('...printing number of unique valies per column')
    for column in df.columns:
        print("{}\t: {}".format(column, len(np.unique(df[column]))))




### PLOTS % EXPORTS #####################################

def PlotHeatMap(df):
    print('...plotting Heatmap')
    plt.figure(figsize=(100,15))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu', fmt='.2f')
    plt.show()

def plotDefaultDistribution(df):
    print('...plotting DEFAULT distribution')
    class_dist = df['DEFAULT'].value_counts()
    plt.figure(figsize=(12,3))
    plt.title('Class Distribution')
    plt.barh(class_dist.index, class_dist.values)
    plt.yticks([0, 1])

    for i, value in enumerate(class_dist.values):
        plt.text(value-2000, i, str(value), fontsize=12, color='white',
                horizontalalignment='right', verticalalignment='center')
    plt.legend()
    plt.savefig('DEFAULT_Distribution.png')

def plotAgeDistribution(df):
    print('...Plotting Age Distribution')
    plt.figure(figsize=(10,8))
    plt.title('Age Distribution')
    plt.xlabel('Age')
    sns.kdeplot(df[df['DEFAULT']==1]['Age'], label='Repayment Late/Default', color='red')
    sns.kdeplot(df[df['DEFAULT']==0]['Age'], label='Repayment on Time', color='green')
    plt.axvline(x=df[df['DEFAULT']==1]['Age'].mean(),color='red', ls='--')
    plt.axvline(x=df[df['DEFAULT']==0]['Age'].mean(),color='green', ls='--')
    plt.grid()
    plt.legend()
    plt.savefig('Age_Distribution.png')

def exportAsCSV(df, name):
    print('...exporting csv')
    df.head(2000).to_csv(name + '.csv')





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
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[['Gender']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_array, columns=feature_labels)
    df_encoded = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df_encoded = df_encoded.drop('Gender', axis=1)
    return df_encoded



### FILL SPARCE COLUMNS ##################################

def fillMonthlyPayments(df):
    print('...filling monthlyPaymnet')
    df.loc[:,'MonthlyPayment'] = df['MonthlyPayment'].fillna(df['MonthlyPayment'].mean())

def fillNrOfDependants(df):
    print('...filling NrOfDependants')
    df.loc[:,'NrOfDependants'] = df['NrOfDependants'].fillna(df['NrOfDependants'].mean())

def fillPreviousRepaymentsBeforeLoan(df):
    print('...filling PreviousRepaymentsBeforeLoan')
    df.loc[:,'PreviousRepaymentsBeforeLoan'] = df['PreviousRepaymentsBeforeLoan'].fillna(df['PreviousRepaymentsBeforeLoan'].mean())

def fillPreviousEarlyRepaymentsBefoleLoan(df):
    print('...filling PreviousEarlyRepaymentsBefoleLoan')
    df.loc[:,'PreviousEarlyRepaymentsBefoleLoan'] = df['PreviousEarlyRepaymentsBefoleLoan'].fillna(df['PreviousEarlyRepaymentsBefoleLoan'].mean())




### MODEL ##############################################

def getTrainRestSplit(df): 
    print('...splitting data into train/test sets')
    X = df.drop('DEFAULT', axis=1).values
    y = df['DEFAULT'].values
    return train_test_split(X, y, test_size=TRAIN_TEST_RATIO, random_state=22)

def normalizeData(X_train, X_test):
    print('...normalizing data')
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled

def trainModel(X_train_scaled, X_test_scaled, y_train, y_test):
    print('...training model')

    # initiating the model 
    model = lgb.LGBMClassifier(
        n_estimators=NUMBER_OF_TREES, 
        class_weight='balanced', 
        random_state=RANDOM_STATE,
        num_leaves=NUMBER_OF_LEAVES,
        max_depth=MAX_TREE_DEPTH, 

    )

    # fitting the model 
    model.fit(
        X_train_scaled, 
        y_train, 
        eval_metric='auc', 
        eval_set=[(X_train_scaled, y_train),(X_test_scaled, y_test)]
    
    )
    return model

def testModel(model, X_train_scaled, X_test_scaled, y_train, y_test): 
    print('...testing model')

    prob_train = model.predict_proba(X_train_scaled)
    prob_test = model.predict_proba(X_test_scaled)


    # Predict the probability score
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)


    print("ACCURACY TRAIN: ", accuracy_score(y_train, pred_train))
    print("ACCURACY TEST: ", accuracy_score(y_test, pred_test))

    cm = confusion_matrix(y_test, )
    # print('Confusion matrix\n\n', cm)
    # print('\nTrue Positives(TP) = ', cm[0,0])
    # print('\nTrue Negatives(TN) = ', cm[1,1])
    # print('\nFalse Positives(FP) = ', cm[0,1])
    # print('\nFalse Negatives(FN) = ', cm[1,0])

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
 
print('   ', df.shape)
df_reduced = addDefaultColumn(df)
print('   ', df_reduced.shape)
df_reduced = removeOngoingLoans(df_reduced)
print('   ', df_reduced.shape)
df_reduced = getReducedDataset(df_reduced)
print('   ', df_reduced.shape)


# REMOVE UNUSABLE DATA
df_reduced = removeBusinessLoans(df_reduced)
print('   ', df_reduced.shape)

df_reduced = removeIncompleteLines(df_reduced)
print('   ', df_reduced.shape)

df_reduced = addFreeCashToAmountColumn(df_reduced)
df_reduced = addFreeCashToMonthlyPaymentColumn(df_reduced)


df_reduced = prepareNrOfDependants(df_reduced)
df_reduced = prepareNewCreditCustomer(df_reduced)

# TRANSLATE COLUMNS
df_reduced = translateLanguageCode(df_reduced)
df_reduced = translateVerificationTyle(df_reduced)
df_reduced = translateUseOfLoan(df_reduced)
df_reduced = translateEducation(df_reduced)
df_reduced = translateMaterialStatus(df_reduced)
df_reduced = translateEmploymentStatus(df_reduced)
df_reduced = translateOccupationArea(df_reduced)
df_reduced = translateHomeOwnershipType(df_reduced)
df_reduced = translateGender(df_reduced)

exportAsCSV(df_reduced.head(1000), 'LoanData_after_cleanup')



# PLOT DISTRIBUTIONS 
plotAgeDistribution(df_reduced)
plotDefaultDistribution(df_reduced)
# printDistribution(df, 'LanguageCode')
# printValueDistribution(df, 'LanguageCode')

# COLUMNS THAT NEED SPECIAL PREPARATION
df_reduced = prepareEmploymentDurationCurrentEmployer(df_reduced)
df_reduced = prepareWorkExperience(df_reduced)

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

print('   ', df_reduced.shape)
exportAsCSV(df_reduced.head(1000), 'LoanData_after_encoding')


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
# plotDefaultDistributionForCol(df_reduced, 'Country')

# EXPORTS
# exportAsCSV(df_reduced, 'reduced')
# exportAsCSV(df_numeric, 'numeric')


# Der nachforlgende Code basiert auf dem Medium Post 
# 'LightGBM on Home Credit Default Risk Prediction' von Muhammad Ardi (2020)
# https://medium.com/becoming-human/lightgbm-on-home-credit-default-risk-prediction-5b17e68a6e9

# TEST TRAIN SPLIT 
X_train, X_test, y_train, y_test = getTrainRestSplit(df_reduced) 
# print('x_train, X_test, y_train, y_test')
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# NORMALISATION
X_train_scaled, X_test_scaled = normalizeData(X_train, X_test)

# TRAIN MODEL
model = trainModel(X_train_scaled, X_test_scaled, y_train, y_test)

# TEST MODEL
testModel(model, X_train_scaled, X_test_scaled, y_train, y_test)
