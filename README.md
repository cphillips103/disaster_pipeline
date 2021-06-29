# Disaster Response Pipeline Project

# Udacity Data Science Nanodegree
## Disaster Response Pipeline
### A Figure 8 dataset

![img1](https://github.com/cphillips103/disaster_pipeline/blob/main/images/splash.png)


Step 1: ETL Pipeline
- Loads the `messages` and `categories` dataset provided by Figure 8
- Merges the two datasets
- Cleans the data
  - Initial column descriptions are concatenated into one column
  - Splits categories into individual columns
  - Creates ones and zeros for yes or no.
  - removes duplications

- Stores clean data frame it in a SQLite database named "disaster_response.db"

Step 2: ML Pipeline Preparation
- Loads data from Sql dataframe
- Tokenizer
  - Splits tweet messages into individual words
  - Leminizes plurals or "ing" words into root words
  - Removes stop words "and, but, or, the"
  - Makes everything lower case
- Creates Pipeline model
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()
- Splits data into training and test (33%) data, random state 42.
- Fits model to data with grid search to tune the parameters
- Prints accuarcy of the results of the model fit.
- Saves best model in Pickle format.

Step 3. Run the Flask Web App
- Uses database and saved model to start Flask based web app.

![img2](https://github.com/cphillips103/disaster_pipeline/blob/main/images/overview_graph.png)

### Notes about the Data:
There are some categories in the provided data that have few examples
like "child_alone", which can create an imbalance in the test and training
sets.

![img3](https://github.com/cphillips103/disaster_pipeline/blob/main/images/distribution.png)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ for local server only

Environmental requiremnets
Flask==0.12.5
ipython==6.5.0
joblib==0.11
nltk==3.2.5
numpy==1.12.1
pandas==0.23.3
plotly==2.0.15
scikit-learn==0.19.1
SQLAlchemy==1.2.19