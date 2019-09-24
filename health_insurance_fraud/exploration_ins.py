import re
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# https://www.kaggle.com/rohitrox/healthcare-provider-fraud-detection-analysis/downloads/healthcare-provider-fraud-detection-analysis.zip/1
df_ipd = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
df_fraud = pd.read_csv('Train-1542865627584.csv')

df = df_ipd.merge(df_fraud, on='Provider').head(1000)
# ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
#        'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
#        'OtherPhysician', 'AdmissionDt', 'ClmAdmitDiagnosisCode',
#        'DeductibleAmtPaid', 'DischargeDt', 'DiagnosisGroupCode',
#        'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
#        'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
#        'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
#        'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
#        'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
#        'ClmProcedureCode_6', 'PotentialFraud']

df.PotentialFraud.replace(to_replace={'Yes': '1', 'No': '0'}, inplace=True)
df_fraud.PotentialFraud.replace(to_replace={'Yes': '1', 'No': '0'}, inplace=True)

f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='PotentialFraud', data=df_fraud)
_ = plt.title('# Fraud vs NonFraud')
_ = plt.xlabel('Class (1==Fraud)')
# plt.show()

df_subset = df[['Provider', 'InscClaimAmtReimbursed', 'ClmAdmitDiagnosisCode', 'DeductibleAmtPaid', 'PotentialFraud']]
df_subset = df_subset[~df_subset['ClmAdmitDiagnosisCode'].astype(str).str.contains("[a-zA-Z]").fillna(False)]
df_subset = df_subset.dropna()

enc = OneHotEncoder()
X = df_subset[['Provider']]
X_provider = enc.fit_transform(X).toarray()
X_provider_columns = enc.get_feature_names()

X_train_provider = pd.DataFrame(X_provider)
X_train_provider.columns = X_provider_columns

df_subset = df_subset.drop(columns='Provider')
df_all = df_subset.merge(X_train_provider, left_index=True, right_index=True)
X_train = df_all.drop(columns='PotentialFraud')
y_train = df_all['PotentialFraud']

lr = LogisticRegression()
scaler = StandardScaler()
model1 = Pipeline([('standardize', scaler), ('log_reg', lr)])
model1.fit(X_train, y_train)

y_train_hat = model1.predict(X_train)
y_train_hat_probs = model1.predict_proba(X_train)[:,1]
train_accuracy = accuracy_score(y_train, y_train_hat)*100
train_auc_roc = roc_auc_score(y_train, y_train_hat_probs)*100
print('Confusion matrix:\n', confusion_matrix(y_train, y_train_hat))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)