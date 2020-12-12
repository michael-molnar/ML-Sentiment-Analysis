import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

X_train = pd.read_csv('clean_training_data_no_stem.csv')
y_train = pd.read_csv('train_labels_no_stem.csv')

X_train.drop(columns=['Unnamed: 0'], inplace=True)
X_train = X_train['reviewText']
y_train = y_train['label']

vectorizer = CountVectorizer(ngram_range=(1,3))
vectorizer.fit(X_train)
X_train_cv = vectorizer.transform(X_train)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train_cv, y_train) 

pickle.dump(lr, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vector.pkl', 'wb'))