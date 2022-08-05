# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# Loading dataset
df = pd.read_csv('./application_train.csv')

# Displaying class distribution
print("0: will repay on time")
print("1: will have difficulty repaying loan")

class_dist = df['TARGET'].value_counts()

plt.figure(figsize=(12,3))
plt.title('Class Distribution')
plt.barh(class_dist.index, class_dist.values)
plt.yticks([0, 1])

for i, value in enumerate(class_dist.values):
    plt.text(value-2000, i, str(value), fontsize=12, color='white',
             horizontalalignment='right', verticalalignment='center')

plt.show()

# A function to fix age representation
def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive/365
    return age_years

# Applying convert_age() function to to the data frame
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(convert_age)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(convert_age)

# Displaying age distribution using kdeplot()
plt.figure(figsize=(10,8))
plt.title('Age Distribution')
plt.xlabel('Age')
sns.kdeplot(df[df['TARGET']==1]['DAYS_BIRTH'], label='Target=1')
sns.kdeplot(df[df['TARGET']==0]['DAYS_BIRTH'], label='Target=0')
plt.grid()
plt.show()

# Features that we are going to use to train the model
used_features = [
    'TARGET',
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'CNT_FAM_MEMBERS',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

# Create a new data frame which only consists of those selected columns
reduced_df = df[used_features]

# Displaying correlation matrix based on reduced_df
plt.figure(figsize=(12,10))
sns.heatmap(reduced_df.corr(), annot=True, cmap='RdBu')

# Displaying the number of unique values in each column
for column in reduced_df.columns:
    print("{}\t: {}".format(column, len(np.unique(reduced_df[column]))))
plt.show()

# A function to convert categorical data into one-hot representation (more than 2 categories)
columns = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

def create_one_hot(reduced_df, columns):
    for column in columns:
        reduced_df = pd.concat([reduced_df, pd.get_dummies(df[column])], axis=1, join='inner')
        reduced_df = reduced_df.drop([column], axis=1)
    
    return reduced_df

reduced_df = create_one_hot(reduced_df, columns)

# Using label encoder to encode columns which consists of only 2 categories
le_name_contract_type = LabelEncoder()
reduced_df['NAME_CONTRACT_TYPE'] = le_name_contract_type.fit_transform(reduced_df['NAME_CONTRACT_TYPE'])

le_flag_own_car = LabelEncoder()
reduced_df['FLAG_OWN_CAR'] = le_flag_own_car.fit_transform(reduced_df['FLAG_OWN_CAR'])

le_flag_own_realty = LabelEncoder()
reduced_df['FLAG_OWN_REALTY'] = le_flag_own_realty.fit_transform(reduced_df['FLAG_OWN_REALTY'])

# Filling missing values with the mean of the corresponding column
reduced_df.loc[:,'AMT_GOODS_PRICE'] = reduced_df['AMT_GOODS_PRICE'].fillna(reduced_df['AMT_GOODS_PRICE'].mean())
reduced_df.loc[:,'CNT_FAM_MEMBERS'] = reduced_df['CNT_FAM_MEMBERS'].fillna(reduced_df['CNT_FAM_MEMBERS'].mean())
reduced_df.loc[:,'EXT_SOURCE_1'] = reduced_df['EXT_SOURCE_1'].fillna(reduced_df['EXT_SOURCE_1'].mean())
reduced_df.loc[:,'EXT_SOURCE_2'] = reduced_df['EXT_SOURCE_2'].fillna(reduced_df['EXT_SOURCE_2'].mean())
reduced_df.loc[:,'EXT_SOURCE_3'] = reduced_df['EXT_SOURCE_3'].fillna(reduced_df['EXT_SOURCE_3'].mean())

# Split the data into train/test
X = reduced_df.iloc[:,1:].values
y = reduced_df['TARGET'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Value normalization
min_max_scaler = MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

# Initializing LightGBM classifier
model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=22)

# Training the LightGBM model
model.fit(X_train_scaled, y_train, eval_metric='auc', 
          eval_set=[(X_train_scaled, y_train),(X_test_scaled, y_test)])


# Predict the probability score
prob_train = model.predict_proba(X_train_scaled)
prob_test = model.predict_proba(X_test_scaled)

# Create train and test curve
fpr_train, tpr_train, thresh_train = roc_curve(y_train, prob_train[:,1])
fpr_test, tpr_test, thresh_test = roc_curve(y_test, prob_test[:,1])

# Create the straight line (how the graph looks like if the model does random guess instead)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs)

# Plot the ROC graph
plt.figure(figsize=(8,6))
plt.title('ROC Curve')
plt.plot(fpr_train, tpr_train, label='Train')
plt.plot(fpr_test, tpr_test, label='Test')
plt.plot(p_fpr, p_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Calculating the train and test AUC score
auc_score_train = roc_auc_score(y_train, prob_train[:,1])
auc_score_test = roc_auc_score(y_test, prob_test[:,1])

print(auc_score_train)
print(auc_score_test)

#### THE CODE BELOW IS NOT EXPLAINED IN THE MEDIUM ARTICLE
# Predict train and test data
pred_train = model.predict(X_train_scaled)
pred_test = model.predict(X_test_scaled)

# Constructing the confusion matrix based on train data
cm_train = confusion_matrix(y_train, pred_train)

# Display the train confusion matrix
plt.figure(figsize=(6,6))
plt.title('Confusion matrix on train data')
sns.heatmap(cm_train, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Constructing the confusion matrix based on test data
cm_test = confusion_matrix(y_test, pred_test)

# Display the test confusion matrix
plt.figure(figsize=(6,6))
plt.title('Confusion matrix on test data')
sns.heatmap(cm_test, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
