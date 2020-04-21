# -*- coding: utf-8 -*-

import pickle

import numpy as np

# google_vec_file = '..\\data\\GoogleNews-vectors-negative300.bin'
# word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)
#
# pickle.dump(word2vec, open("..\\pickled_files\\word2vec_google.pkl", 'wb'))
#
# loading pre-trained word2vec model (commented code above)

word2vec = pickle.load(open("..\\pickled_files\\word2vec_google.pkl", 'rb'))


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
