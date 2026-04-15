import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('titanic.csv')    #to read the file

##Replacing Null
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

x=df.drop(columns=['Survived','Name','PassengerId','Ticket','Cabin'])
y=df['Survived']      #to create the variable

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)   #split the val

from sklearn.naive_bayes import GaussianNB  
NB = GaussianNB()

NB.fit(x_train, y_train)

y_pred=NB.predict(x_test)

#['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

testPrediction = NB.predict([[3,0,22.0,1,1,6,2]])
if testPrediction==1:
    print("Survived")
else:
    print("Not survived")