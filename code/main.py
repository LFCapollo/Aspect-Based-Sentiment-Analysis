# -*- coding: utf-8 -*-

import pickle
from collections import Counter

import pandas as pd
from assign_terms_to_aspects import assign_term_to_aspect
from classify import classify
from find_terms import find_sentiments
from lemmatize import fix_output
from lemmatize import lemmatize_sentence
from preprocessing import remove_special_chars
from preprocessing import replace_pronouns
from preprocessing import split_sentence


def review_pipe(review, aspect_sent, terms_dict={'ambience':Counter(), 'food':Counter(), 'price':Counter(),
                                                 'service':Counter(),'misc':Counter()}):
    """
    The function fixes co-referencing, splits review into sentences, removes special characters from sentences,
    does lematization, and classify sentence using pre-trained model.
    Finds sentiments in each sentence and assigns it to aspects.

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


# read restaurant data-set
dt = pd.read_csv("..\data\yelp_reviews_primanti.csv")

# drop unnecessary columns
dt = dt.drop(["address", "city", "state",
              "postal_code", "latitude", "longitude",
              "attributes", "user_id", "review_stars"], axis=1)

# business id of a restaurant primanti bros
dt = dt[dt.business_id == 'lKom12WnYEjH5FFemK3M1Q']
terms_dict = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(),'misc': Counter()}
aspect_sent = {'ambience': Counter(), 'food': Counter(), 'price': Counter(), 'service': Counter(), 'misc': Counter()}

# sending data in review pipe
for review in dt.text:
    aspect_sent, terms_dict = review_pipe(review, aspect_sent, terms_dict)

print(terms_dict)
print(aspect_sent)

# saving aspects and terms for visualization
pickle.dump(aspect_sent, open("..\\pickled_files\\primanti_aspect.pkl", 'wb'))
pickle.dump(terms_dict, open("..\\pickled_files\\primanti_terms.pkl", 'wb'))
