'''

Data process automation for disaster pipeline machine
learning project with Udacity.

Command line takes 4 arguments:

messages_filepath: The path of messages dataset to be cleaned
categories_filepath: The path of categories dataset
Sqlite3 database filepath for merged dataframe.

'''


#importing necessary libraries

import pandas as pd
import sys
import sqlite3
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):

    # load messages dataset
    messages = pd.read_csv (messages_filepath)

    # load categories dataset
    categories = pd.read_csv (categories_filepath)

    # merge data sets based on id column
    df = pd.merge(messages, categories, on=["id"])

    return df



def clean_data(df):
    # process merged dataframe
    
    # data received with categores concatenated in one column
    # we need to process the column and seperate it into
    # one category per column.
    
    # use original category data to create column headers
    categories = df['categories'].str.split(';',expand=True)    

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # using first row to extract a list of new column names
    category_colnames = row.str.split('-').apply(lambda x:x[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # select the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    # make copy of merged dataframe
    df_merged = df.copy()

    return df_merged

def save_data(df_merged, database_filename):
    
    # create the engine sqllite
    # save the table in the database mentioned above
    
    engine = create_engine('sqlite:///' + database_filename)
    # assuming that  we are saving table in same directory
    # as the messages, categories, and the process_data.py file
    df_merged.to_sql('disaster_response_table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()