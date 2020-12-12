import pandas as pd
import numpy as np
import re
import string
from num2words import num2words
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from flask import Flask, request, jsonify, render_template 
import pickle

app = Flask(__name__)
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
    return text

"""
I want to use the model to generate few words and phrases.
To do this I use the coefficients of hte model and the terms of the vectorizer.
"""
# The coefficients for the negative class
negative_coef = {
    word: coef for word, coef in zip(
        vectorizer.get_feature_names(), model.coef_[0])
    }

# The coefficients for the neutral class
neutral_coef = {
    word: coef for word, coef in zip(
        vectorizer.get_feature_names(), model.coef_[1])
    }

# The coefficients for the positive class
positive_coef = {
    word: coef for word, coef in zip(
        vectorizer.get_feature_names(), model.coef_[2])
    }




@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['your_text']
    processed_text = [process_input(input_text)]
    vec_text = vectorizer.transform(processed_text)

    prediction = model.predict(vec_text)
    confidence = model.predict_proba(vec_text)

    output = prediction[0].upper()
    neg_prop = np.around(confidence[0][0]*100, 2)
    neu_prop = np.around(confidence[0][1]*100, 2)
    pos_prop = np.around(confidence[0][2]*100, 2)

    pred_text1 = 'The Predicted Sentiment is: {}'.format(output)
    pred_text2 = 'Probability for Negative: {}%'.format(neg_prop)
    pred_text3 = 'Probability for Neutral: {}%'.format(neu_prop)
    pred_text4 = 'Probability for Positive: {}%'.format(pos_prop)

    return render_template('index.html', your_text = input_text, prediction_text = pred_text1, neg_text = pred_text2, neu_text = pred_text3, pos_text = pred_text4)

@app.route('/predict_analyze', methods=['POST', 'GET'])
def predict_analyze():
    input_text = request.form['your_text']
    processed_text = [process_input(input_text)]
    vec_text = vectorizer.transform(processed_text)

    prediction = model.predict(vec_text)
    confidence = model.predict_proba(vec_text)
    output = prediction[0].upper()
    neg_prop = np.around(confidence[0][0]*100, 2)
    neu_prop = np.around(confidence[0][1]*100, 2)
    pos_prop = np.around(confidence[0][2]*100, 2)

    pred_text1 = 'The Predicted Sentiment is: {}'.format(output)
    pred_text2 = 'Probability for Negative: {}%'.format(neg_prop)
    pred_text3 = 'Probability for Neutral: {}%'.format(neu_prop)
    pred_text4 = 'Probability for Positive: {}%'.format(pos_prop)

    # Get the features of the input text and store the English words or phrases
    phrases = []
    for item in vec_text.indices:
        phrases.append(vectorizer.get_feature_names()[item])
    
    #Create a dictionary of the model's coefficients for these features 
    importances = dict()
    for phrase in phrases:
        if prediction == 'negative':
            importances[phrase] = negative_coef.get(phrase)
        elif prediction == 'neutral':
            importances[phrase] = neutral_coef.get(phrase)
        elif prediction == 'positive':
            importances[phrase] = positive_coef.get(phrase)
            
    # Sort this dictionary according to its values
    importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    importances = dict(importances[:20])
    pred_text1 = 'The Predicted Sentiment is: {}'.format(output)
    pred_text2 = 'Probability for Negative: {}%'.format(neg_prop)
    pred_text3 = 'Probability for Neutral: {}%'.format(neu_prop)
    pred_text4 = 'Probability for Positive: {}%'.format(pos_prop)
       
    pred_text5 = 'Top Ten Phrases'
    #cloud = WordCloud(background_color='black').generate_from_frequencies(importances)
    #plt.figure(figsize=(6, 6), facecolor=None)
    #plt.imshow(cloud, interpolation="bilinear")
    #plt.axis("off")
    #plt.tight_layout(pad=0)

    #new_graph_name = "graph" + str(time.time()) + ".png"

    #for filename in os.listdir('static/images'):
     #   if filename.startswith('graph_'):  # not to remove other images
      #      os.remove('static/' + filename)

    #plt.savefig('static/images/' + new_graph_name, bbox_inches='tight')


    #plt.savefig('static\images\word_cloud.png', bbox_inches="tight")
    
    pred_text6 = importances[0][0]
    pred_text7 = importances[1][0]
    pred_text8 = importances[2][0]
    pred_text9 = importances[3][0]
    pred_text10 = importances[4][0]
    pred_text11 = importances[5][0]
    pred_text12 = importances[6][0]
    pred_text13 = importances[7][0]
    pred_text14 = importances[8][0]
    pred_text15 = importances[9][0]
    



    return render_template('index.html', prediction_text = pred_text1, your_text = input_text, neg_text = pred_text2, neu_text = pred_text3, pos_text = pred_text4,
        analysis_text = pred_text5, + , analysis_text2 = pred_text6, analysis_text3 = pred_text7, analysis_text4 = pred_text8, 
        analysis_text5 = pred_text9, analysis_text6 = pred_text10, analysis_text7 = pred_text11, analysis_text8 = pred_text12, analysis_text9 = pred_text13,
        analysis_text10 = pred_text14, analysis_text11 = pred_text15)
 #url='static/images/'#
if __name__ == '__main__':
	app.run(debug=True)