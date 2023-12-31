# -*- coding: utf-8 -*-
"""bank_churn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yc0wTo4ge7w_arlA26fJMwHbcuw7defv

**Customer Churn prediction:-**

 Customer Churn prediction means knowing which customers are likely to leave or unsubscribe from your service. For many companies, this is an important prediction. This is because acquiring new customers often costs more than retaining existing ones. Once you’ve identified customers at risk of churn, you need to know exactly what marketing efforts you should make with each customer to maximize their likelihood of staying.




Customers have different behaviors and preferences, and reasons for cancelling their subscriptions. Therefore, it is important to actively communicate with each of them to keep them on your customer list. You need to know which marketing activities are most effective for individual customers and when they are most effective.
"""

# importing the require libarary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from google.colab import drive
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, classification_report, ConfusionMatrixDisplay

#importing the drive from the google
#from google.colab import drive
#drive.mount('/content/drive')

#load the data sheet from the drive
churn= pd.read_csv("C:\Users\ADMIN\Desktop\model churn\Churn_Modelling.csv")

# Display information about the dataset, including data types and missing values
churn.info()

# Display information about the dataset, including data types and missing values
churn.info()

# Display summary statistics of the dataset
churn.describe()

"""#Auto EDA process"""

'''!pip install matplotlib
!pip install pandas-profiling
!pip install --upgrade pandas
!pip install --upgrade numpy
!pip install --upgrade matplotlib
!pip install --upgrade seaborn
!pip install --upgrade scikit-learn
!pip install sweetviz'''

"""Restart the kernal once again"""

import matplotlib.pyplot as plt
import sweetviz as sv

'''# Analyze the dataset
report = sv.analyze(churn)

# Display the report
report.show_html("churn_eda_report.html")'''

"""#EDA  (Exploratory Data Analysis)"""

# Get the dimensions of the dataset (rows, columns)
churn.shape

# Get the total number of elements in the dataset
churn.size

# Get the names of the columns in the dataset
churn.columns

# Count missing values in each column
churn.isnull().sum()

# Identify and display duplicate rows based on the 'customer_id' column
duplicates = churn[churn.duplicated(subset=['CustomerId'], keep=False)]
print(duplicates)

"""# Data Visualization"""

# Set the figure size for the following visualization
plt.figure(figsize=(15, 5))

# Create a count plot to visualize the distribution of the 'exited' variable in the original dataset
sns.countplot(data=churn, x='Exited')

!pip install plotly

# Display the count of each class in the 'exited' variable
churn['Exited'].value_counts().to_frame()

# Class Imbalance Resampling
# Select the majority and minority classes
churn_majority = churn[churn['Exited'] == 0]
churn_minority = churn[churn['Exited'] == 1]

from sklearn.utils import resample

# Downsample the majority class to match the size of the minority class
churn_majority_downsample = resample(churn_majority, n_samples=2037, replace=False, random_state=42)

# Combine the resampled majority class with the minority class
churn_df = pd.concat([churn_majority_downsample, churn_minority])

# Set the figure size for the following visualization
plt.figure(figsize=(15, 5))

# Create a count plot to visualize the distribution of the 'exited' variable in the resampled dataset
sns.countplot(data=churn_df, x='Exited')

# Display the column names in the 'churn_df' DataFrame
churn_df.columns

# Remove specific columns from the 'churn_df' DataFrame
# These columns include 'rownumber,' 'customerid,' 'surname,' 'geography,' and 'gender.'
churn_df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography','Gender',], axis=1, inplace=True)

# Compute the correlation matrix for the remaining columns in the 'churn_df' DataFrame
churn_df.corr()

# Create a heatmap to visualize the correlation between different features
# The 'annot=True' parameter adds values to the heatmap
plt.figure(figsize=(15, 5))
sns.heatmap(churn_df.corr(), annot=True)

# Calculate the correlation of each feature with the 'exited' variable and store it in 'df_corr_exit'
df_corr_exit = churn_df.corr()['Exited'].to_frame()

# Create a bar plot to visualize the correlation of each feature with the 'exited' variable
plt.figure(figsize=(15, 5))
sns.barplot(data=df_corr_exit, x=df_corr_exit.index, y='Exited')

# Separate the feature columns (independent variables) into 'x'
x = churn_df.drop(['Exited'], axis=1)
# Separate the target variable ('exited') into 'y'
y = churn_df['Exited']

"""#Spliting the Data Set"""

# Import the necessary function from scikit-learn
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# x: Features (independent variables)
# y: Target variable (dependent variable)
# test_size: The proportion of the data to include in the test split (in this case, 30% for testing)
# random_state: A random seed for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Print the dimensions of the resulting datasets
x_train.shape, x_test.shape, y_train.shape, y_test.shape

"""#Modeling and Evaluation"""

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
x_train_scaled = scaler.fit_transform(x_train)
#y_train_scaled = scaler.fit_transform(y_train)

# Transform the test data using the same scaler
x_test_scaled = scaler.transform(x_test)
#y_test_scaled = scaler.transform(y_test)

# Import the logistic regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model with a specified maximum number of iterations
lr = LogisticRegression(max_iter=500)

# Train the logistic regression model on the training data
lr.fit(x_train_scaled, y_train)
#lr.fit(x_train_scaled, y_train_scaled)

# Calculate the accuracy score on the training set
train_accuracy = lr.score(x_train_scaled, y_train)
print("Training Accuracy:", train_accuracy)

# Predict outcomes on the test set
y_pred = lr.predict(x_test_scaled)

# Import necessary functions for performance evaluation
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, ConfusionMatrixDisplay

# Calculate the precision score on the test set
test_precision = precision_score(y_test, y_pred)
print("Test Precision Score:", test_precision)

# Calculate the recall score on the test set
test_recall = recall_score(y_test, y_pred)
print("Test Recall Score:", test_recall)

# Calculate the accuracy score on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Score:", test_accuracy)

from sklearn.metrics import classification_report

# Calculate the F1 score on the test set
#test_f1 = f1_score(y_test, y_pred)
print(classification_report(y_pred,y_test))
#print("Test F1 Score:", test_f1)

# Create a ConfusionMatrixDisplay object for visualization
cmd = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred, labels=lr.classes_),
    display_labels=lr.classes_
)
# Plot the confusion matrix
cmd.plot()

"""#k-Nearest Neighbors (KNN)"""

# Import and create a KNN classifier with k=3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model on the training data
knn.fit(x_train, y_train)

# Calculate the accuracy score on the training set for KNN
knn_train_accuracy = knn.score(x_train_scaled, y_train)
print("KNN Training Accuracy:", knn_train_accuracy)

# Calculate the accuracy score on the test set for KNN
knn_test_accuracy = knn.score(x_test_scaled, y_test)
print("KNN Test Accuracy:", knn_test_accuracy)

"""#Support Vector Classifier (SVC):"""

# Import and create an SVC classifier
from sklearn.svm import SVC
svc = SVC()

# Train the SVC model on the training data
svc.fit(x_train_scaled, y_train)

# Calculate the accuracy score on the training set for SVC
svc_train_accuracy = svc.score(x_train_scaled, y_train)
print("SVC Training Accuracy:", svc_train_accuracy)

# Calculate the accuracy score on the test set for SVC
svc_test_accuracy = svc.score(x_test_scaled, y_test)
print("SVC Test Accuracy:", svc_test_accuracy)

"""#Random Forest"""

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,  # Cross-validation folds
                           verbose=2,  # Higher values give more information
                           n_jobs=-1)  # Use all available CPU cores for parallelization

# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Fit the GridSearchCV object
grid_search.fit(x_train_scaled, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Predictions using the best model
y_pred_rf = best_rf.predict(x_test_scaled)
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Print the best cross-validation score
print("Best cross-validation score:", grid_search.best_score_)

"""# Feature Engineering"""

# Drop the columns you want to exclude
columns_to_drop = ['Surname', 'Geography', 'Gender']
churn = churn.drop(columns=columns_to_drop)

# Calculate the 'BalanceSalaryRatio'
churn['BalanceSalaryRatio'] = churn['Balance'] / churn['EstimatedSalary']

# Create a new feature 'CreditScoreAgeRatio'
churn['CreditScoreAgeRatio'] = churn['CreditScore'] / churn['Age']

# Calculate 'TenureAgeRatio'
churn['TenureAgeRatio'] = churn['Tenure'] / churn['Age']

# Create a new feature 'HasCrCardIsActiveMember'
churn['HasCrCardIsActiveMember'] = churn['HasCrCard'] * churn['IsActiveMember']

# Calculate 'CreditScoreGivenSalary'
churn['CreditScoreGivenSalary'] = churn['CreditScore'] / churn['EstimatedSalary']

# Calculate 'NumOfProductsGivenAge'
churn['NumOfProductsGivenAge'] = churn['NumOfProducts'] / churn['Age']

# Calculate 'BalanceGivenAge'
churn['BalanceGivenAge'] = churn['Balance'] / churn['Age']

# Calculate 'BalanceGivenCreditScore'
churn['BalanceGivenCreditScore'] = churn['Balance'] / churn['CreditScore']

# Create a new feature 'TenureGivenAge'
churn['TenureGivenAge'] = churn['Tenure'] / churn['Age']

# Recreate 'Exited' variable
churn['Exited'] = churn['Exited'].astype(int)

# Re-split the data into features and target
x = churn.drop(['Exited'], axis=1)
y = churn['Exited']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train a Logistic Regression model
lr = LogisticRegression(max_iter=500)
lr.fit(x_train_scaled, y_train)
y_pred = lr.predict(x_test_scaled)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Test Accuracy:", test_accuracy)
# Calculate the accuracy score on the training set
train_accuracy = accuracy_score(y_train, lr.predict(x_train_scaled))
print("Logistic Regression Training Accuracy:", train_accuracy)

"""#comparesion of all model after feather engineering"""



# Logistic Regression Model
# Train a Logistic Regression model
lr = LogisticRegression(max_iter=500)
lr.fit(x_train_scaled, y_train)
y_pred = lr.predict(x_test_scaled)

# Evaluate the Logistic Regression model
test_accuracy_lr = accuracy_score(y_test, y_pred)
train_accuracy_lr = accuracy_score(y_train, lr.predict(x_train_scaled))
print("Logistic Regression Training Accuracy:", train_accuracy_lr)
print("Logistic Regression Test Accuracy:", test_accuracy_lr)

# K-Nearest Neighbors (KNN) Model
# Import and create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_scaled, y_train)
y_pred_knn = knn.predict(x_test_scaled)

# Evaluate the KNN model
test_accuracy_knn = accuracy_score(y_test, y_pred_knn)
train_accuracy_knn = accuracy_score(y_train, knn.predict(x_train_scaled))
print("KNN Training Accuracy:", train_accuracy_knn)
print("KNN Test Accuracy:", test_accuracy_knn)

# Support Vector Classifier (SVC) Model
# Import and create an SVC classifier
svc = SVC()
svc.fit(x_train_scaled, y_train)
y_pred_svc = svc.predict(x_test_scaled)

# Evaluate the SVC model
test_accuracy_svc = accuracy_score(y_test, y_pred_svc)
train_accuracy_svc = accuracy_score(y_train, svc.predict(x_train_scaled))
print("SVC Training Accuracy:", train_accuracy_svc)
print("SVC Test Accuracy:", test_accuracy_svc)

# Random Forest Model
# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_scaled, y_train)
y_pred_rf = rf.predict(x_test_scaled)

# Evaluate the Random Forest model
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
train_accuracy_rf = accuracy_score(y_train, rf.predict(x_train_scaled))
print("Random Forest Training Accuracy:", train_accuracy_rf)
print("Random Forest Test Accuracy:", test_accuracy_rf)

import pickle

# Save the KNN model to a pickle file
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

print("KNN model saved as 'knn_model.pkl'")

"""conclution :-
the best model for the data sheet is knn with test accuracy of 83.16% and train accuracy of 89.52%
"""