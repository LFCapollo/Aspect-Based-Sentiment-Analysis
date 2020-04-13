## Aspect-Based-Sentiment-Analysis

# Notebooks
Notebooks contain data preprocessing part and model training using multinomialNB. Different models were tested on the training data
results are in models directory.

# code
code section contains modules for sentiment analysis
main.py creates a pipeline for sentiment analysis
preprocessing.py fixes coreferencing, divides reviews into sentences and removes special characters
classify.py classifies sentences into aspects using pretrained multinomialNB model
find_terms.py finds sentiments in sentences and gives them sentiment score
assign_terms_to_aspects.py assigns sentiment scores to respective aspects using word2vec and multinomialNB predict

# plots
visualization of people's sentiments with respect to aspects

# flask
Placeholder for the flask app

# Data
restaurant_train.xml was used for multinomialNB training
yelp_reviews_primanti.csv was used to test sentiment analysis model
opinion lexicon contains positive and negative opinion words
!missing word2vec pretrained model


