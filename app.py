"""
Michael Molnar - 100806823
In this module I will build my Flask App for deployment.  This will be 
combined with the index.html file to be hosted on Heroku.  
"""

import pandas as pd
import numpy as np
import re
import string
from num2words import num2words
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import time
import os


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from flask import Flask, request, jsonify, render_template 
import pickle

# Create the flask app
app = Flask(__name__)
# Import model and vectorizer from model.py
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vector.pkl', 'rb'))

"""
In this section I will define all of the functions that must be used to
preprocess input text to make predictions.
"""
# Function to expand contractions
def decontract(sentence):
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

# Function to remove linebreaks
def remove_linebreaks(input):
    text = re.compile(r'\n')
    return text.sub(r' ',input)

# Function to remove punctuation
def remove_punctuation(input):
    no_punc = [char for char in input if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return no_punc

# Function to replace numbers with text
def replace_numbers(text):
    words = []
    for word in text.split():
        if word.isdigit():
            words.append(num2words(word))
        else:
            words.append(word)
    return " ".join(words)

# Generate the list of stopwords
stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
"you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
"doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
"mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', 
"wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords_list.remove('no')
stopwords_list.remove('not')
stopwords_list.remove('very')
stopwords_list.remove('only')

# Function to remove stopwords
def remove_stopwords(input):
    no_stop = [word for word in input.split() if word not in stopwords_list]
    no_stop = ' '.join(no_stop)
    return no_stop

# Function to stem text
def stem_text(input):
    stemmer = SnowballStemmer('english')
    text = input.split()
    words = ""
    for i in text:
        words += (stemmer.stem(i))+' '
    return words 

"""
Next, I combine all of these into one function to process text for prediction.
"""
def process_input(text):
    text = decontract(text)
    text = remove_linebreaks(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = replace_numbers(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

@app.route('/')
def home():
	return render_template('index.html')
# Define what will happen when the Predict Sentiment Button is pressed
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['your_text']
    processed_text = [process_input(input_text)]
    #Vectorize text with previously fit Count Vectorizer
    vec_text = vectorizer.transform(processed_text)

    # Generate predictions and probabilities
    prediction = model.predict(vec_text)
    confidence = model.predict_proba(vec_text)

    # Define the output
    output = prediction[0].upper()
    neg_prob = np.around(confidence[0][0]*100, 2)
    neu_prob = np.around(confidence[0][1]*100, 2)
    pos_prob = np.around(confidence[0][2]*100, 2)

    pred_text0 = 'Your Text:'
    pred_text1 = 'The Predicted Sentiment is: {}'.format(output)
    head_text = 'Probabilities:'
    pred_text2 = 'Positive:  {}%'.format(pos_prob)
    pred_text3 = 'Neutral:  {}%'.format(neu_prob)
    pred_text4 = 'Negative:  {}%'.format(neg_prob)

    # Create a bar graph
    heights = [pos_prob, neu_prob, neg_prob]
    bars = ('Positive', 'Neutral', 'Negative')
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax = plt.bar(bars, heights, color=['gold', 'royalblue', 'tomato'])
    plt.tight_layout(pad=0)
    
    # Delete previous graph and save this one with a unique timestamp in name
    new_graph_name = "graph" + str(time.time()) + ".png"
    for filename in os.listdir('static/images'):
        if filename.startswith('graph'): 
            os.remove('static/images/' + filename)
    plt.savefig('static/images/' + new_graph_name, bbox_inches='tight')
    plt.cla()
    plt.close(fig)

    return render_template('index.html', your_text_header=pred_text0,  your_text = input_text, prediction_text = pred_text1, bottom_header = head_text, neg_text = pred_text4, neu_text = pred_text3, 
        pos_text = pred_text2, url='static/images/' + new_graph_name)

if __name__ == '__main__':
	app.run(debug=True)