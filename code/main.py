# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:45:06 2020

@author: Nika
"""
import os
import pickle
from preprocessing import replace_pronouns, split_sentence, remove_special_chars
from classify import classify
from find_terms import find_sentiments
from assign_terms_to_aspects import assign_term_to_aspect
from collections import Counter, defaultdict
from lemmatize import lemmatize_sentence, fix_output


def review_pipe(review, aspect_sent, terms_dict={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}):

    """

    Args:
        review: string
            Restaurant review
        aspect_sent: defaultdict
            Dictionary of aspects
        terms_dict: defaultdict
            Dictionary of aspects

    Returns:
        aspect_sent: defaultdict
            Dictionary of aspects with total positive and negative sentiments
            Examples ambience': Counter({'pos': 568.75, 'neg': 251.0})
        terms_dict: defaultdict
            Dicionary of aspects with respective terms and their values
            Examples 'ambience': Counter({'atmosphere': 59.25, 'location': 33.75

    review pipe fixes correferencing, splits review into sentences removes special characters from sentences, does lematization,
    classifys sentence using pretrained model
    finds sentiments in each sentence and assigns it to aspects

    """
    review = replace_pronouns(review)
    sentences = split_sentence(review)
    sentiment_dict = Counter()
    for sentence in sentences:
        sentence = remove_special_chars(str(sentence))
        sentence = lemmatize_sentence(sentence)
        sentence = fix_output(sentence)
        predicted_aspect = classify(sentence.lower())
        sentiment_dict = find_sentiments(sentence.lower())
       
        aspect_sent, terms_dict = assign_term_to_aspect(aspect_sent, terms_dict, sentiment_dict, predicted_aspect[0])
     
    return aspect_sent, terms_dict

import pandas as pd

#importing restaurant dataset
dt = pd.read_csv("..\data\yelp_reviews_primanti.csv")

#drop unnecessary columns
dt = dt.drop(["address", "city", "state", "postal_code", "latitude", "longitude", "attributes", "user_id", "review_stars"], 1)

#business id of a restaurant primanti bros
dt = dt[dt.business_id == 'lKom12WnYEjH5FFemK3M1Q']
terms_dict={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}
aspect_sent={'ambience':Counter(), 'food':Counter(), 'price':Counter(), 'service':Counter(),'misc':Counter()}

#sending data in review pipe
for review in dt.text:
    aspect_sent, terms_dict = review_pipe(review, aspect_sent, terms_dict)

print(terms_dict)
print(aspect_sent)

#saving aspects and terms for visualization

pickle.dump(aspect_sent, open("..\\pickled_files\\primanti_aspect.pkl", 'wb'))
pickle.dump(terms_dict, open("..\\pickled_files\\primanti_terms.pkl", 'wb'))