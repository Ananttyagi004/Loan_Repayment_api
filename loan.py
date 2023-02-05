import pandas as pd
import numpy as np



from fastapi import FastAPI
import uvicorn
from main import request_body

app = FastAPI()

train=pd.read_csv("train.csv")
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
from sklearn.preprocessing import LabelEncoder
ls=LabelEncoder()
colums=[ 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Property_Area', 'Loan_Status']

for column in colums:
    train[column]=ls.fit_transform(train[column])


train=train.drop(columns='Loan_ID', axis=1)
X=train.drop(columns="Loan_Status", axis=1)
Y=train['Loan_Status']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)




