# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset

import neuralcoref
import spacy

neg_file = open("..\\data\\opinion-lexicon-English\\neg_words.txt",encoding = "ISO-8859-1")
pos_file = open("..\\data\\opinion-lexicon-English\\pos_words.txt",encoding = "ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos

word2vec = pickle.load(open("..\\pickled_files\\word2vec_google.pkl", 'rb'))
mlb = pickle.load(open("..\\pickled_files\\mlb.pkl", 'rb'))
NB_model=pickle.load(open("..\\pickled_files\\NB_model.pkl", 'rb'))

nlp = spacy.load('en')

neuralcoref.add_to_pipe(nlp)
def check_similarity(aspects, word):
    '''
    checks for word2vec similarity values between aspects and terms
    returns most similar aspect
    '''
    similarity = []
    for aspect in aspects:
        similarity.append(word2vec.n_similarity([aspect], [word]))
    # set threshold for max value
    if max(similarity) > 0.25:
        return aspects[np.argmax(similarity)]
    else:
        return None
    
def assign_term_to_aspect(aspect_sent, terms_dict, sent_dict, pred):
    '''
    function: takes in a sentiment dictionary (sent_dict) and appends the aspect dictionary
    inputs: sent_dict is a Counter in the form Counter(term:sentiment value)
            aspect_sent is a Counterm (aspect: total value)
            terms_dict: dictionary with terms associated with corresponding aspecys
    output: updated terms_dict and aspect_sent
    '''
    aspects = ['ambience', 'food', 'price', 'service']
    
    
    
    # First, check word2vec
    # Note: the .split() is used for the term because word2vec can't pass compound nouns
    for term in sent_dict:
        try:
            # The conditions for when to use the NB classifier as default vs word2vec
            
            if check_similarity(aspects, term.split()[-1]):
                terms_dict[check_similarity(aspects, term.split()[-1])][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent[check_similarity(aspects, term.split()[-1])]["pos"] += sent_dict[term]
                else:
                    aspect_sent[check_similarity(aspects, term.split()[-1])]["neg"] += abs(sent_dict[term])
            
            elif (len(pred) == 1):
                terms_dict[pred[0]][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent[pred[0]]["pos"] += sent_dict[term]
                else:
                    aspect_sent[pred[0]]["neg"] += abs(sent_dict[term])
            # if unable to classify via NB or word2vec, then put them in misc. bucket
            else:
                terms_dict["misc"][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent["misc"]["pos"] += sent_dict[term]
                else:
                    aspect_sent["misc"]["neg"] += abs(sent_dict[term])
        except:
            print(term, "not in vocab")
            continue
    return aspect_sent, terms_dict

def find_sentiments(text):
    """
    this function checks whether token can contain positive or negative opinion word
    if token is positive sentiment is 1; if token is negative sentiment is -1
    after that we check for token dependency
    """
    sentiment_dict = Counter()
    sentiment=0
    sentence = nlp(text)
    for token in sentence:
        if (token.dep_ == 'advmod'):
            continue
        if (token.text in opinion_words):
            if (token.text in pos):
                sentiment = 1
            else:
                sentiment = -1
            sentiment_dict =check_for_dep(token, sentiment, sentiment_dict)
    return sentiment_dict

def check_for_dep(token, sentiment, sentiment_dict):
    """
    if token is adjective modifier we append it to term dictionary
    otherwise we check if token has a weight modifier such as adverb or adjective
    we review the case when token is verb
    we check for negation words existence in sentence
    we check nouns in sentence
    returns sentiment dict
    """
    if (token.dep_=='amod'):
        sentiment_dict[token.head.text] +=sentiment
        return sentiment_dict
    else:
        sentiment = check_for_weight_modifier(token, sentiment)
        sentiment_dict = check_for_verb(token, sentiment, sentiment_dict)
        sentiment= check_for_negations(token, sentiment)
        sentiment_dict = check_for_nouns(token, sentiment, sentiment_dict)
        return sentiment_dict

def check_for_weight_modifier(token, sentiment):
    """
    if token has adjective modifier or adverb modifier child which is in opinion words,
    we increase weight by multiplying sentiment by 1.5
    if child is negative opinion word we flip sign
    returns sentiment
    """
    for child in token.children:
        if (child.text in opinion_words and (child.dep_ =='amod') or child.dep_=='advmod'):
            sentiment*=1.5
        if (child.dep_ =='neg'):
            sentiment*=-1
    return sentiment
def check_for_verb(token, sentiment, sentiment_dict):
    """
    if token is verb and it has direct object we append direct object to terms dictionary
    besides that we check if direct object has conjunction
    returns sentiment_dict
    """
    for child in token.children:
        if (token.pos_ =='VERB' and child.dep_ =='dobj'):
            sentiment_dict[child.text] +=sentiment
            sentiment_dict=check_for_conjunction(child, sentiment, sentiment_dict)
    return sentiment_dict
            
def check_for_conjunction(token, sentiment, sentiment_dict):
    """
    this function checks for conjunction for direct object and if it exists appends to terms dictionary
    returns sentiment dict
    """
    for child in token.children:
        if (child.dep_ == 'conj'):
            sentiment_dict[child.text] += sentiment
    return sentiment_dict

def check_for_negations(token, sentiment):
    """
    this function checks for negation words in sentence and flips the sign
    returns sentiment
    """
    for child in token.head.children:
        if (child.text in opinion_words and (child.dep_ =='amod') or child.dep_=='advmod'):
            sentiment*=1.5
        if (child.dep_ =='neg'):
            sentiment*=-1
    return sentiment
def check_for_nouns(token, sentiment, sentiment_dict):
    """this function checks for nouns in the sentence and also checks for compound nouns and appends to term dictionary
    returns sentiment_dict
    """
    for child in token.head.children:
        noun = ''
        if (child.pos_ =='NOUN' and child.text not in sentiment_dict):
            noun = child.text
            for subchild in child.children:
                if (subchild.dep_ == 'compound'):
                    noun = subchild.text + " " + noun
            sentiment_dict[noun] += sentiment
    return sentiment_dict
def classify(sentence):

    predicted = mlb.inverse_transform(NB_model.predict([sentence]))
    return predicted

def replace_pronouns(text):
    doc = nlp(text)
    return doc._.coref_resolved
def split_sentence(text):
    review = nlp(text)
    sentences = []
    start = 0
    for token in review:
        if token.sent_start: #boolean value if token starts the sentence
            sentences.append(review[start:(token.i-1)])
            start = token.i
        if token.i == len(review)-1:
            sentences.append(review[start:(token.i+1)])
    return sentences

#remove special characters using regex
def remove_special_chars(text):
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', text)

def review_pipe(review, aspect_sent, terms_dict={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}):
    review = replace_pronouns(review)
    sentences = split_sentence(review)
    sentiment_dict = Counter()
    for sentence in sentences:
        sentence = remove_special_chars(str(sentence))
        predicted_aspect = classify(sentence.lower())
        sentiment_dict = find_sentiments(sentence.lower())
       
        aspect_sent, terms_dict = assign_term_to_aspect(aspect_sent, terms_dict, sentiment_dict, predicted_aspect[0])
     
    return aspect_sent, terms_dict

terms_dict={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}
aspect_sent={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}
@app.route("/process", methods=["POST"])
def prob():
    data = request.get_json()
    sentence = data['review']
    aspect, terms= review_pipe(sentence, aspect_sent, terms_dict)
    

    return jsonify({'aspect': aspect, 'terms': terms})

if __name__ == "__main__":
    app.run()
