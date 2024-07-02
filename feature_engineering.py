import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datavisualization import visualize_data

# Download VADER lexicon
nltk.download('vader_lexicon')

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

    # Applying sentiment analysis
    data['benefits_sentiment'] = data['benefitsReview'].apply(get_sentiment_scores)
    data['sideEffects_sentiment'] = data['sideEffectsReview'].apply(get_sentiment_scores)
    data['comments_sentiment'] = data['commentsReview'].apply(get_sentiment_scores)
    
    # Drop rows that were used for sentiment analysis
    data.drop(['urlDrugName', 'benefitsReview', 'sideEffectsReview', 'commentsReview'], axis=1, inplace=True)

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

    # Scale the Data
    new_data = data.drop(columns=['rating'])
    scaler = StandardScaler()
    scaler.fit(new_data)
    scaled_features = scaler.transform(new_data)
    scaled_data = pd.DataFrame(scaled_features,columns=new_data.columns[:])
    selected_columns = data[['rating']]
    scaled_data[['rating']] = selected_columns.copy()
    data = scaled_data
    '''

    data.dropna(axis=0, how='any', inplace=True) # Drop any rows with null values
    
    print(data.head())
    
    data.to_csv('drug_reviews_cleansed_data.csv', index=False)

    return data

engineer_features()
