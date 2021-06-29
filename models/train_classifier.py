'''

Disaster Response Tweets NLP Machine Learning Project
With Udacity.

Model attempts to predict relevance of tweets and if
they are related to evolving disasters or not related.

'''

# load necessary libraries
import sys
from time import time
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import pickle

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import joblib


def load_data(database_filepath):

    # creates engine for sqlite
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # loads cleaned database from sql
    df_clean = pd.read_sql_table('disaster_response_table',engine)
 
    #make copy of loaded dataframe
    df = df_clean.copy()

    # create message dataframe and target dataframe
    X = df['message'].copy()
    y = df.iloc[:,4:].copy()

    return X, y


def tokenize(text):

    # this gives us a list of words from the messages
    tokens = word_tokenize(text)
    # this groups inflected forms of words
    lemmatizer = WordNetLemmatizer()
    # drop words like "and, but, or, the"
    stopWords = set(stopwords.words('english'))

    # Get clean tokens after lemmatization, normalization,
    # and stripping and stop words removal
    clean_tokens = []
    for tok in tokens:
        # make sure all words are lowercase
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)
 
    return clean_tokens


def build_model():

    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # select parameters for grid search #1
    parameters = {
        'vect__min_df': [1, 5],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10, 25], 
        'clf__estimator__min_samples_split':[2, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, y_test):

    # make model prediction
    y_pred = model.predict(X_test) 

    # print general model accuracy at the top
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    # basic scikit learn classification report by column
    for ind, cat in enumerate(y_test.keys()): 
        print("Classification report for {}".format(cat))
        print(classification_report(y_test.iloc[:,ind], y_pred[:,ind]))
 
    # tuned model grid search results
    grid_scores = pd.DataFrame(model.cv_results_)
    print(grid_scores)

    # best parameters for tuned model grid search
    print(model.best_params_)
    

def save_model(model, model_filepath):

    # Pickle best model
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        
        t0 = time()
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
 
        # load dataframe from sql database
        X, y = load_data(database_filepath)

        # split into test and train data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
         test_size=0.33, random_state=42)

        # build the model
        print('Building model...')
        model = build_model()

        # train the model
        print('Training model...')
        model.fit(X_train, y_train)

        # evaluate the model outputs
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        # save model in pickle format
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        print("done in %0.3fs" % (time() - t0))
              
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()