# -*- coding: utf-8 -*-

import pickle


# Loading naive bayes model

mlb = pickle.load(open("pickled_files/mlb.pkl", 'rb'))
NB_model = pickle.load(open("pickled_files/NB_model.pkl", 'rb'))

# Classify sentence and inverse transform from vector to string


def classify(sentence: str) -> list:
    """
    Takes one sentence from review and classifies it into aspect using pre-trained model

    Returns:
        predicted: list
        Example: ["ambience"]
    """

    predicted = mlb.inverse_transform(NB_model.predict([sentence]))
    return predicted
