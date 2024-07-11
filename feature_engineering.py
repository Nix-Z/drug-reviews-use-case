import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datavisualization import visualize_data

# Download VADER lexicon
# nltk.download('vader_lexicon')

def engineer_features():
    data = visualize_data()

    # Initialize the sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Function to get sentiment scores
    def get_sentiment_scores(text):
        if isinstance(text, str):
            sentiment = sia.polarity_scores(text)
            return sentiment['compound']
        return np.nan

    # Concatenate reviews into a single text column for sentiment analysis
    data['combined_review'] = data['benefitsReview'].fillna('') + ' ' + data['sideEffectsReview'].fillna('') + ' ' + data['commentsReview'].fillna('')

    # Applying sentiment analysis using VADER on concatenated reviews
    data['combined_sentiment'] = data['combined_review'].apply(get_sentiment_scores)
    
    # Drop rows that were used for sentiment analysis
    data.drop(['benefitsReview', 'sideEffectsReview', 'commentsReview', 'combined_review', 'urlDrugName'], axis=1, inplace=True)

    # Convert categorical columns to numerical
    data.replace({'effectiveness': {
                                    'Highly Effective':0,
                                    'Considerably Effective':1,
                                    'Moderately Effective':2,
                                    'Marginally Effective':3,
                                    'Ineffective':4
                                    }
                  }, inplace=True)
    
    data.replace({'sideEffects': {
                                    'No Side Effects':0,
                                    'Mild Side Effects':1,
                                    'Moderate Side Effects':2,
                                    'Severe Side Effects':3,
                                    'Extremely Severe Side Effects':4
                                    }
                  }, inplace=True)

    '''
    le = LabelEncoder()
    data['urlDrugName'] = le.fit_transform(data['urlDrugName'])
    '''
    
    print(data.head())
    
    data.to_csv('drug_reviews_cleansed_data_2.csv', index=False)

    return data

engineer_features()
