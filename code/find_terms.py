# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:48:09 2020

@author: Nika
"""
from collections import Counter, defaultdict
import spacy
nlp = spacy.load('en')

"""
neg_file contains dictionary of negative opinion words
pos_file contains dictionary of positive opinion words
opinion words is merging of those two dictionaries
"""
neg_file = open("..\\data\\opinion-lexicon-English\\neg_words.txt",encoding = "ISO-8859-1")
pos_file = open("..\\data\\opinion-lexicon-English\\pos_words.txt",encoding = "ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos

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