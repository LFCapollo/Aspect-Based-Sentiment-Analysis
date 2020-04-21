# -*- coding: utf-8 -*-

from collections import Counter

import spacy

nlp = spacy.load('en')

"""
neg_file: contains dictionary of negative opinion words.
pos_file: contains dictionary of positive opinion words.
opinion words is a merge of those two dictionaries.
"""

neg_file = open("..\\data\\opinion-lexicon-English\\neg_words.txt", encoding="ISO-8859-1")
pos_file = open("..\\data\\opinion-lexicon-English\\pos_words.txt", encoding="ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos


def find_sentiments(text: str) -> dict:
    """
    This function checks whether token can contain positive or negative opinion word.
    If token is positive then sentiment is 1.
    if token is negative then sentiment is -1.

    Returns:
        sentiment_dict: dictionary
    """

    sentiment_dict = Counter()
    sentiment = 0
    sentence = nlp(text)
    for token in sentence:
        if (token.dep_ == 'advmod'):
            continue
        if (token.text in opinion_words):
            if (token.text in pos):
                sentiment = 1
            else:
                sentiment = -1
            sentiment_dict = check_for_dep(token, sentiment, sentiment_dict)
    return sentiment_dict


def check_for_dep(token, sentiment: int, sentiment_dict: dict) -> dict:
    """
    Function checks for token dependency.
    If token is adjective modifier function appends it to term dictionary,
    otherwise function checks if token has a weight modifier such as adverb or adjective.

    Returns:
        sentiment dictionary
    """

    if (token.dep_=='amod'):
        sentiment_dict[token.head.text] += sentiment
        return sentiment_dict
    else:
        sentiment = check_for_weight_modifier(token, sentiment)
        sentiment_dict = check_for_verb(token, sentiment, sentiment_dict)
        sentiment = check_for_negations(token, sentiment)
        sentiment_dict = check_for_nouns(token, sentiment, sentiment_dict)
        return sentiment_dict


def check_for_weight_modifier(token, sentiment: int) -> int:
    """
    If token has adjective modifier or adverb modifier child, which is in opinion words,
    function increases weight by multiplying sentiment by 1.5.
    Ff child is negative opinion word function flips the sign.

    Returns:
        sentiment

    """

    for child in token.children:
        if (child.text in opinion_words and (child.dep_ == 'amod') or child.dep_ == 'advmod'):
            sentiment *= 1.5
        if (child.dep_ == 'neg'):
            sentiment *= -1
    return sentiment


def check_for_verb(token, sentiment: int, sentiment_dict: dict) -> dict:
    """
    If token is verb and it has direct object function appends direct object to terms dictionary.
    Examples: "I like tennis". In this example, tennis is a direct object, Like is a verb
    Besides that function checks if direct object has conjunction.

    Returns:
        sentiment dictionary
    """

    for child in token.children:
        if (token.pos_ == 'VERB' and child.dep_ == 'dobj'):
            sentiment_dict[child.text] += sentiment
            sentiment_dict = check_for_conjunction(child, sentiment, sentiment_dict)
    return sentiment_dict


def check_for_conjunction(token, sentiment: int, sentiment_dict: dict) -> dict:
    """
    This function checks for conjunction for direct object and if it exists appends to terms dictionary.
    Example: "I like tennis and basketball". Basketball is a conjunction.

    Returns:
        sentiment dictionary
    """

    for child in token.children:
        if (child.dep_ == 'conj'):
            sentiment_dict[child.text] += sentiment
    return sentiment_dict


def check_for_negations(token, sentiment: int) -> int:
    """
    This function checks for negation words in sentence and flips the sign of the sentiment

    Returns:
        sentiment
    """

    for child in token.head.children:
        if (child.dep_ == 'neg'):
            sentiment *= -1
    return sentiment


def check_for_nouns(token, sentiment: int, sentiment_dict: dict) -> dict:
    """
    This function checks for nouns and compound nouns in the sentence and appends to term dictionary.
    Examples: compound noun "full moon"

    Returns:
        sentiment dictionary
    """

    for child in token.head.children:
        noun = ''
        if (child.pos_ == 'NOUN' and child.text not in sentiment_dict):
            noun = child.text
            for subchild in child.children:
                if (subchild.dep_ == 'compound'):
                    noun = subchild.text + " " + noun
            sentiment_dict[noun] += sentiment
    return sentiment_dict
