# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:41:13 2020

@author: Nika
"""

import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
import gensim

#google_vec_file = '..\\data\\GoogleNews-vectors-negative300.bin'
#word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)

#pickle.dump(word2vec, open("..\\pickled_files\\word2vec_google.pkl", 'wb'))

#loading pretrained word2vec mpdel (commented code above)
word2vec = pickle.load(open("..\\pickled_files\\word2vec_google.pkl", 'rb'))


def check_similarity(aspects, word):
    """

    Args:
        aspects: list
        word: string

    Returns:
        aspect: string

    checks for word2vec similarity values between aspects and terms
    returns most similar aspect or nothing if similarity score <0.25
    """
    similarity = []
    for aspect in aspects:
        similarity.append(word2vec.n_similarity([aspect], [word]))
    # set threshold for max value
    if max(similarity) > 0.2:
        return aspects[np.argmax(similarity)]
    else:
        return None
    
def assign_term_to_aspect(aspect_sent, terms_dict, sent_dict, pred):
    """

    Args:
        aspect_sent: dictionary
        terms_dict: dictionary
        sent_dict: dictionary
        pred: list

    Returns:
        aspect_sent: dictionary
        Dictionary of aspects with total positive and negative sentiments
        Examples ambience': Counter({'pos': 568.75, 'neg': 251.0})
        terms_sent: dictionary
            Dicionary of aspects with respective terms and their values
            Examples 'ambience': Counter({'atmosphere': 59.25, 'location': 33.75
    the function assigns terms to respective aspects  according to prediction made by pretrained model
    the finction assigns total value to aspects which is the sum of term values

    """

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

