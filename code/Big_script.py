#!/usr/bin/env python3
# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

# load loan data
df = pd.read_csv('LoanData.csv', low_memory=False)
df_clean = df.copy(deep=True)

def addTargetColumn(): 
    TARGET = []
    for row in df['Status']:
        if row == 'Late' : TARGET.append(1)
        else: TARGET.append(0)
    
    df['TARGET'] = TARGET
    
    # if boolean should be prefered
    # df['TARGET'] = df.Status == 'Late'   

    
def removeOngoingLoans():
    global df_clean
    df_clean = df[df['Status'] != 'Current']
    
    
def removeIncompleteLines():
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
    
    print(incompleteRows)
    print('\n', len(incompleteRows), 'incomplete lines. if equal to 50 data is missing in the same lines across different columns.')
    
    df_clean = df_clean.drop(incompleteRows)
    # df_clean = df_clean.drop(df_clean[df_clean.index in incompleteRows].index)





    
# Excecute code

print(df_clean.shape)

addTargetColumn()
removeOngoingLoans()

removeIncompleteLines()

print(df_clean.shape)
