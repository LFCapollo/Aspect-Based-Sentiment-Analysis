## Aspect-Based-Sentiment-Analysis

The aspect based sentiment analysis project aims to classify restaurant reviews into several aspects on the sentence level, find useful terms in sentences and evaluate with sentiment scores. The terms are assigned to respective aspects and their sentiment scores are summed up. Finally we can evaluate total aspect sentiment score and find positive and negative terms associated with each aspects.

# Notebooks
Notebooks contain data preprocessing part and model training using multinomialNB. Different models were tested on the training data
results are in models directory.

# code
code section contains modules for sentiment analysis <br/>
main.py creates a pipeline for sentiment analysis <br/>
preprocessing.py fixes coreferencing, divides reviews into sentences and removes special characters <br/>
classify.py classifies sentences into aspects using pretrained multinomialNB model <br/>
find_terms.py finds sentiments in sentences and gives them sentiment score <br/>
assign_terms_to_aspects.py assigns sentiment scores to respective aspects using word2vec and multinomialNB predict <br/>

# plots
visualization of people's sentiments with respect to aspects <br/>

# flask
Placeholder for the flask app <br/>

# Data
restaurant_train.xml was used for multinomialNB training <br/>
yelp_reviews_primanti.csv was used to test sentiment analysis model <br/>
opinion lexicon contains positive and negative opinion words <br/>
!missing word2vec pretrained model

# Run the code using $python main.py


