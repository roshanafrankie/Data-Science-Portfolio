# Titanic Survival Prediction: A Statistical Approach

This project integrates **Exploratory Data Analysis (EDA)** with a **Gaussian Naive Bayes** machine learning model to predict passenger survival on the Titanic.

## 📊 Project Overview
The objective is to analyze passenger demographics and identify survival patterns. By leveraging statistical cleaning methods, we ensure the data is prepared correctly for probabilistic modeling.

## 🛠️ Key Features
* **Data Visualization**: Uses `missingno` for mapping data gaps and `seaborn` for survival distribution plots.
* **Statistical Imputation**:
    * **Age**: Filled with the **Median** (28.0) to remain robust against outliers identified during EDA.
    * **Embarked**: Filled with the **Mode** ("S") to represent the most statistically probable port.
* **Feature Engineering**:
    * Converts categorical text (`Sex`, `Embarked`) into integers using `LabelEncoder`.
    * Drops non-informative columns: `PassengerId`, `Name`, `Ticket`, and `Cabin`.
* **Machine Learning**: Implements a **Gaussian Naive Bayes** classifier, which assumes features follow a normal distribution.

## 📂 Code Workflow
1. **Exploration**: Identifying missing values and analyzing the "shape" of the data.
2. **Preprocessing**: Handling null values **before** encoding to maintain data integrity.
3. **Data Split**: Dividing the dataset into **80% Training** and **20% Testing**.
4. **Training**: Fitting the model to the training features (X) and target (y).
5. **Evaluation**: Calculating the **Accuracy Score** and generating a **Confusion Matrix**.

## 🚀 How to Run
Ensure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib missingno