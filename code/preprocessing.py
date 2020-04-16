# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:01:40 2020

@author: Nika
"""

import os
import neuralcoref
import spacy
import re
nlp = spacy.load('en')

neuralcoref.add_to_pipe(nlp)

#fixing correference
def replace_pronouns(text):
    doc = nlp(text)
    return doc._.coref_resolved
#split sentences
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

