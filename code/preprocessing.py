# -*- coding: utf-8 -*-

import re

import neuralcoref
import spacy

nlp = spacy.load('en')

neuralcoref.add_to_pipe(nlp)


# fixing co-reference
def replace_pronouns(text: str) -> str:
    """
    Function resoles co-reference dependency.
    Example: "I drove Joe home because he lives near my apartment" ->
    -> "I drove Joe home because Joe lives near my apartment"

    Returns:
        text with resolved co-reference
    """

    doc = nlp(text)
    return doc._.coref_resolved


def split_sentence(text: str) -> list:
    """
    Splits review into list

    Returns:
        list of sentences in restaurant review
    """
    review = nlp(text)
    sentences = []
    start = 0
    for token in review:
        if token.sent_start:  # boolean value if token starts the sentence
            sentences.append(review[start:(token.i - 1)])
            start = token.i
        if token.i == len(review) - 1:
            sentences.append(review[start:(token.i+1)])
    return sentences


def remove_special_chars(text: str) -> str:
    """
    Removes numbers and punctuations from the text

    Returns:
        The text without numbers and punctuations
    """
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', text)
