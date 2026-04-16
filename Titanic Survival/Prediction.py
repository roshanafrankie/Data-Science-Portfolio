import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

df = pd.read_csv('titanic.csv')

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

X = df.drop(columns=['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'])
y = df['Survived']

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Sex', 'Embarked'])
    ], 
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=12)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)

def get_user_input_and_predict():
    print("\n" + "="*20 + " TITANIC SURVIVAL PREDICTOR (ONE-HOT + LOG) " + "="*20)
    try:
        pclass = int(input("Enter Pclass (1, 2, or 3): "))
        sex = input("Enter Sex (male or female): ").strip().lower()
        age = float(input("Enter Age: "))
        sibsp = int(input("Enter Siblings/Spouses aboard: "))
        parch = int(input("Enter Parents/Children aboard: "))
        fare = float(input("Enter Fare: "))
        embarked = input("Enter Embarked Port (C, Q, or S): ").strip().upper()

        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        input_data = [pclass, sex, age, sibsp, parch, fare, embarked]
        input_df = pd.DataFrame([input_data], columns=feature_cols)

        print("\n" + "-"*15 + " YOUR RAW INPUT DATA " + "-"*15)
        print(input_df.to_string(index=False))
        print("-" * 52)

        input_encoded = ct.transform(input_df)

        prediction = log_model.predict(input_encoded)
        probability = log_model.predict_proba(input_encoded)[0][1]

        result = "✅ SURVIVED" if prediction[0] == 1 else "❌ NOT SURVIVED"
        print(f"\nFINAL PREDICTION: {result}")
        print(f"SURVIVAL CHANCE: {round(probability * 100, 2)}%")
        print("="*52)

    except Exception as e:
        print(f"\nError: {e}. Please ensure inputs are valid.")

get_user_input_and_predict()