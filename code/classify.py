# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:58:42 2020

@author: Nika
"""
import os
import pickle
#loading naive bayes model
mlb = pickle.load(open("..\\pickled_files\\mlb.pkl", 'rb'))
NB_model=pickle.load(open("..\\pickled_files\\NB_model.pkl", 'rb'))
#classify sentence and inverse trandsform from vector to string
def classify(sentence):

    predicted = mlb.inverse_transform(NB_model.predict([sentence]))
    return predicted


