"""
Bank Churn Personal Project - Machine Learning Classification Algorithm

Analysing and predicting if customers will stay with the bank
Costs more to take new customer than retaining one
"""

__date__ = "2023-01-16"
__author__ = "Thomas Mullan"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# %% --------------------------------------------------------------------------
# Generate Random State 
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Load Data 
# -----------------------------------------------------------------------------
file_path = "C:/mle06/Machine Learning/Data/archive/Bank Customer Churn Prediction.csv"
imported_df = pd.read_csv(file_path)

# %% --------------------------------------------------------------------------
# Exploratory Data Analysis
# -----------------------------------------------------------------------------
#imported_df.describe()
#imported_df.info()
#Check for null values
imported_df.isnull().sum()
print(imported_df.dtypes)

# Drop Customer ID column (Holds no significance)
df = imported_df.drop(columns="customer_id")
# %% --------------------------------------------------------------------------
# Data Visualisation 
# -----------------------------------------------------------------------------
#Count Plots for Products_number, Tenure 
fig, ax = plt.subplots(1,2, figsize=(8,10))
sns.countplot(x="gender", data=df, hue='churn', ax = ax[0])
sns.countplot(x="credit_card", data=df, hue='churn', ax=ax[1])
fig, ax = plt.subplots(1,2, figsize=(8,10))
sns.countplot(x="active_member", hue='churn', data=df, ax=ax[0])
sns.countplot(x="country", data=df, hue='churn', ax=ax[1])

#Heatmap - Correlation between features
fig, ax = plt.subplots(1, 1, figsize = (8,12))
sns.heatmap(df.corr(), fmt='.2g')

#Histograms of Contineuous Variables
cont_var = ['age', 'tenure', 'balance', 'products_number', 'credit_score', 'estimated_salary']
for col in cont_var:
    sns.histplot(data=df, x=col, color='#1f77b4', kde=False, edgecolor='black', bins=20)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(col + ' Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#Churn Proportion
# Data is Imbalanced (0:7963, 1:2037)
value_counts = df['churn'].value_counts()
labels = ["Retained", "Left"]
plt.pie(value_counts, labels=labels, startangle=90, autopct='%1.1f%%', 
        explode=[0.1, 0], shadow = True, colors=['#ff9999', '#66b3ff'])
plt.axis('equal')
plt.title('Churn Distribution')
plt.show()

#%%
# Change country, gender col to numeric 
df['country'] = df['country'].map({'France': 2, 'Germany': 1, 'Spain': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Convert dtypes to ints
print(df.dtypes)
df[['balance', 'estimated_salary']] = df[['balance', 'estimated_salary']].astype(int)

# %% --------------------------------------------------------------------------
# Train Test Split 
# -----------------------------------------------------------------------------
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=rng, stratify=df['churn']
)

X_train = df_train.drop(columns='churn')
X_test = df_test.drop(columns='churn')
y_train = df_train['churn']
y_test = df_test['churn']

# %% --------------------------------------------------------------------------
# Linear Regression Model
# -----------------------------------------------------------------------------
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
# %% --------------------------------------------------------------------------
# Decision Tree Model
# -----------------------------------------------------------------------------
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred)*100, "%")
# %% --------------------------------------------------------------------------
# Random Forest Model
# -----------------------------------------------------------------------------
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred)*100, "%")
# %% --------------------------------------------------------------------------
# XG Boost Model
# -----------------------------------------------------------------------------
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred)*100, "%")
# %% --------------------------------------------------------------------------
# KNN Model
# -----------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred)*100, "%")
