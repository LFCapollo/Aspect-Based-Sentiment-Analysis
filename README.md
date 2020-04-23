# Aspect Based Sentiment Analysis

The aspect based sentiment analysis project aims to classify restaurant reviews into several aspects on the sentence level, find useful terms in sentences and evaluate with sentiment scores. The terms are assigned to respective aspects and their sentiment scores are summed up. Finally we can evaluate total aspect sentiment score and find positive and negative terms associated with each aspect.

## Notebooks

Contain data pre-processing part and model training using MultiNomialNB. Different models were tested on the training data and results are in models directory.

## code

This directory contains modules for sentiment analysis.

* ```main.py``` creates a pipeline for sentiment analysis.

* ```preprocessing.py``` fixes co-referencing, divides reviews into sentences, and removes special characters.

* ```classify.py``` classifies sentences into aspects using pre-trained MultiNomialNB model.

* ```find_terms.py``` finds sentiments in sentences and gives them sentiment score.
 
* ```assign_terms_to_aspects.py``` assigns sentiment scores to respective aspects using word2vec and MultiNomialNB prediction.

* ```lemmatize.py``` performs lemmatization of sentences and fixes output

## Plots

Visualization of people's sentiments with respect to aspects.

## Flask

The directory contains web service of the model. Run the following in the terminal.

```shell
python app.py
```

request.py can be used to send data to the service and get aspects and terms dictionaries by running the following in the console:

```shell
python request.py
```

## Data

Data directory contains data files for model building processes.

* restaurant_train.xml was used for MultiNomial Naive Bayes training.

* yelp_reviews_primanti.csv was used to test sentiment analysis model.

* opinion lexicon contains positive and negative opinion words

* !missing word2vec pre-trained model

## Possible improvements

* Preprocessing and balancing data for classification model improvement.

* Exploring stemming algorithms and comparing their performance with lemmatization algorithms.

* Defining more complex rules for sentiment extraction and designing test cases for performance evaluation.

### Run the code using 

```shell
python main.py
```
