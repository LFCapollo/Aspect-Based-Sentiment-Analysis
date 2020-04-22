# -*- coding: utf-8 -*-
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

import pickle
from collections import Counter
import re

import neuralcoref
import numpy as np
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

neg_file = open("data/opinion-lexicon-English/neg_words.txt", encoding="ISO-8859-1")
pos_file = open("data/opinion-lexicon-English/pos_words.txt", encoding="ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos

word2vec = pickle.load(open("pickled_files/word2vec_google.pkl", 'rb'))
mlb = pickle.load(open("pickled_files/mlb.pkl", 'rb'))
NB_model = pickle.load(open("pickled_files/NB_model.pkl", 'rb'))

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)


def nltk_tag_to_wordnet_tag(nltk_tag: str) -> str:
    """

    Args:
        nltk_tag: nltk tags
        token tag (adjective, verb, Noun, Adverb)

    Returns:
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence: str) -> str:
    """
    Tokenize the sentence and find POS tag for each token.

    Returns:
        Lemmatized sentence
    """

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def fix_output(text: str) -> str:
    """
    The function fixes mistakes of lematization output, such as do n't -> don't

    Returns:
        text
    """

    text = text.replace(" n't", "n't")
    return text


def check_similarity(aspects: list, word: str) -> str:
    """
    Checks for word2vec similarity values between aspects and terms.

    Returns:
        The most similar aspect or nothing if similarity score <0.2
    """
    similarity = []
    for aspect in aspects:
        similarity.append(word2vec.n_similarity([aspect], [word]))
    # set threshold for max value
    if max(similarity) > 0.2:
        return aspects[np.argmax(similarity)]
    else:
        return None


def assign_term_to_aspect(aspect_sent: dict, terms_dict: dict, sent_dict: dict, pred: list) -> tuple:
    """
    The function assigns terms to respective aspects according to the prediction made by pre-trained model.
    The function assigns total value to aspects which is the sum of term values.

    Returns:
        aspect_sent: dictionary
        Dictionary of aspects with total positive and negative sentiments
        Example: {'ambience': Counter({'pos': 568.75, 'neg': 251.0})}

        terms_sent: dictionary
            Dictionary of aspects with respective terms and their values
            Example: {'ambience': Counter({'atmosphere': 59.25, 'location': 33.75})}
    """
    aspects = ['ambience', 'food', 'price', 'service']

    # First, check word2vec
    for term in sent_dict:
        try:
            # The conditions for when to use the NB classifier as default vs word2vec
            # Note: the .split() is used for the term because word2vec can't pass compound nouns
            if check_similarity(aspects, term.split()[-1]):
                terms_dict[check_similarity(aspects, term.split()[-1])][term] += sent_dict[term]
                if sent_dict[term] > 0:
                    aspect_sent[check_similarity(aspects, term.split()[-1])]["pos"] += sent_dict[term]
                else:
                    aspect_sent[check_similarity(aspects, term.split()[-1])]["neg"] += abs(sent_dict[term])

            elif (pred[0] == "anecdotes/miscellaneous"):
                continue

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


def find_sentiments(text: str) -> dict:
    """
    This function checks whether sentence can contain positive or negative opinion word(s).
    If token is positive then sentiment is 1.
    if token is negative then sentiment is -1.

    Returns:
        sentiment_dict: dictionary
    """

    sentiment_dict = Counter()
    sentiment = 1
    sentence = nlp(text)
    for token in sentence:
        if (token.dep_ == 'advmod'):
            continue
        if (token.text in opinion_words):
            if (token.text in neg):
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

    if (token.dep_ == 'amod'):
        if token.head.text not in sentiment_dict:
            sentiment_dict[token.head.text] += sentiment * 1.5
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
    If child is negative opinion word function flips the sign.

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
            if child.text not in sentiment_dict:
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
            if child.text not in sentiment_dict:
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
            if noun not in sentiment_dict:
                sentiment_dict[noun] += sentiment
    return sentiment_dict


def classify(sentence: str) -> list:
    """
    Takes one sentence from review and classifies it into aspect using pre-trained model

    Returns:
        predicted: list
        Example: ["ambience"]
    """

    predicted = mlb.inverse_transform(NB_model.predict([sentence]))
    return predicted


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
            sentences.append(review[start:(token.i + 1)])
    return sentences


def remove_special_chars(text: str) -> str:
    """
    Removes numbers and punctuations from the text

    Returns:
        The text without numbers and punctuations
    """
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', text)


def review_pipe(review: str,
                aspect_sent: dict,
                terms_dict={'ambience': Counter(), 'food': Counter(), 'price': Counter(),
                            'service': Counter(), 'misc': Counter()}) -> tuple:
    """
    The function fixes co-referencing, splits review into sentences, removes special characters from sentences,
    does lematization, and classify sentence using pre-trained model.
    Finds sentiments in each sentence and assigns it to aspects.

    Returns:
        aspect_sent: defaultdict
            Dictionary of aspects with total positive and negative sentiments
            Example: {'ambience': Counter({'pos': 568.75, 'neg': 251.0})}
        terms_dict: defaultdict
            Dictionary of aspects with respective terms and their values
            Example: {'ambience': Counter({'atmosphere': 59.25, 'location': 33.75})}
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


terms_dict = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(), 'misc': Counter()}
aspect_sent = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(), 'misc': Counter()}


@app.route("/process", methods=["POST"])
def prob():
    data = request.get_json()
    sentence = data['review']
    aspect, terms = review_pipe(sentence, aspect_sent, terms_dict)

    return jsonify({'aspect': aspect, 'terms': terms})


if __name__ == "__main__":
    app.run()
