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

pd.set_option('display.max_columns', 30)

# https://www.kaggle.com/rohitrox/healthcare-provider-fraud-detection-analysis/downloads/healthcare-provider-fraud-detection-analysis.zip/1
df_ipd = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
df_bene = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
df_fraud = pd.read_csv('Train-1542865627584.csv')

cols = ['ChronicCond_stroke', 'ChronicCond_rheumatoidarthritis',
        'ChronicCond_Osteoporasis', 'ChronicCond_IschemicHeart',
        'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
        'ChronicCond_Diabetes', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_Alzheimer']

df_bene[cols] = df_bene[cols].replace(2, 0)

# Bar Chart of # of Each Diagnosis
df_bar = df_bene[cols].sum(axis=0, numeric_only=True)
y = df_bar.values
x = []
for i in cols:
    j = i.split('_')[1]
    x.append(j)
plt.barh(x, y)

df = df_ipd.merge(df_fraud, on='Provider')
df_full = df.merge(df_bene, on='BeneID')

# print(df_full.isna().sum() * 100 / df_full.shape[0])
df_full_cols = df_full.columns
df_full_values = (df_full.isna().sum() * 100 / df_full.shape[0]).values
for i in range(len(df_full_values)):
    if df_full_values[i] > 50.0:
        label = df_full_cols[i]
        df_full.drop(labels=label, axis=1)

df_full.PotentialFraud.replace(to_replace={'Yes': '1', 'No': '0'}, inplace=True)
df_fraud.PotentialFraud.replace(to_replace={'Yes': '1', 'No': '0'}, inplace=True)

# Figures exploring data visualizations
plt.figure()
f, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='PotentialFraud', data=df_fraud)
_ = plt.title('# Fraud vs NonFraud, Based on Provider')
_ = plt.xlabel('Class (1==Fraud)')
plt.show()

plt.figure()
g, ay = plt.subplots(figsize=(7, 5))
sns.countplot(x='PotentialFraud', data=df_full)
_ = plt.title('# Fraud vs NonFraud, Based on Claim')
_ = plt.xlabel('Class (1==Fraud)')

sns.set(rc={'figure.figsize': (12, 8)}, style='white')
plt.figure()
sns.countplot(x='State', hue='PotentialFraud', data=df_full,
              order=df_full.State.value_counts().iloc[:10].index, orient="h")
plt.figure()
sns.countplot(x='ChronicCond_IschemicHeart', hue='PotentialFraud', data=df_full,
              order=df_full.ChronicCond_IschemicHeart.value_counts().iloc[:10].index, orient="h")
plt.figure()
sns.countplot(x='Provider', data=df_full,
              order=df_full.Provider.value_counts().iloc[:10].index, orient="h")

df_full["ClmCount_Provider"] = \
    df_full.groupby(['Provider'])['ClaimID'].transform('count')
print(df_full["ClmCount_Provider"])

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
