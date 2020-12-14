"""
Michael Molnar - 100806823
This model will take the cleaned datasets from Notebook 3 and create
the Count Vectorizer and Logistic Regression Classifier.  It will 
pickle these two items for use in the Flask App that will follow.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

# Import cleaned datasets
X_train = pd.read_csv('clean_training_data.csv')
X_test = pd.read_csv('clean_testing_data.csv')
y_train = pd.read_csv('training_labels.csv')
y_test = pd.read_csv('testing_labels.csv')

# Prepare for vectorizing
X_train.drop(columns=['Unnamed: 0'], inplace=True)
X_test.drop(columns=['Unnamed: 0'], inplace=True)
X_train = X_train['reviewText']
X_test = X_test['reviewText']
y_train = y_train['label']
y_test = y_test['sentiment']

# Fit a count vectorizer 
vectorizer = CountVectorizer(ngram_range=(1,3))
vectorizer.fit(X_train)
X_train_cv = vectorizer.transform(X_train)
X_test_cv = vectorizer.transform(X_test)

# Fit a Logistic Regression Model 
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train_cv, y_train) 

# Get predictions and check accuracies
lr_train_preds = lr.predict(X_train_cv)
lr_preds = lr.predict(X_test_cv)
lr_train_acc = accuracy_score(y_train, lr_train_preds)
lr_test_acc = accuracy_score(y_test, lr_preds)
print("Training Accuracy:", lr_train_acc)
print("Testing Accuracy:", lr_test_acc)

# Pickle the model and vectorizer for use in app
pickle.dump(lr, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vector.pkl', 'wb'))