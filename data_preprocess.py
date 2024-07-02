import pandas as pd
from data_analysis import analyze_data

def preprocess_data():
    data = analyze_data()
    data.reset_index(drop=True,inplace=True)
    
    data.drop(['Unnamed: 0', 'condition'], axis=1, inplace=True)
    data.dropna(axis=0, how='any', inplace=True) # Drop any rows with null values

    return data

preprocess_data()
