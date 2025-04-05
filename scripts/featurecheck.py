import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('/app/data/truckerpath_1star_3620.csv')

# Convert all columns to strings
df = df.astype(str)

# Check the column names and first few rows
print(df.columns)
print(df.head())

# Get info about data types and missing values
df.info()

# Check for nulls in each column
print(df.isnull().sum())

# Drop rows with no review content
df = df.dropna(subset=['content'])

# Convert date columns to datetime
df['at'] = pd.to_datetime(df['at'])
df['repliedAt'] = pd.to_datetime(df['repliedAt'], errors='coerce')  # some may be NaT

# Fill missing values in optional columns (only if the column exists)
if 'reviewCreatedVersion' in df.columns:
    df['reviewCreatedVersion'] = df['reviewCreatedVersion'].fillna('Unknown')

if 'appVersion' in df.columns:
    df['appVersion'] = df['appVersion'].fillna('Unknown')

if 'replyContent' in df.columns:
    df['replyContent'] = df['replyContent'].fillna('No Reply')

# (Optional) Drop columns you don't need
df = df.drop(columns=['userImage'])

# Reset index
df.reset_index(drop=True, inplace=True)

# Display a preview after cleaning
print("Cleaning complete. Here's a preview:")
print(df.head())

# Check again after cleaning
df.info()
print(df.isnull().sum())

# Sort reviews by thumbsUpCount (descending)
top_helpful_reviews = df.sort_values(by='thumbsUpCount', ascending=False)

# Display the top 500 most "helpful" 1-star reviews
print(top_helpful_reviews[['userName', 'content', 'thumbsUpCount', 'at']].head(500))

# Plot top 10 helpful reviews
top_10 = top_helpful_reviews.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_10['userName'], top_10['thumbsUpCount'], color='crimson')
plt.xlabel('Thumbs Up Count')
plt.title('Top 10 Most Helpful 1-Star Reviews')
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.show()

# Get the top 500 reviews sorted by thumbsUpCount
top_500_reviews = df.sort_values(by='thumbsUpCount', ascending=False).head(500)

# Export to CSV
top_500_reviews.to_csv('top_500_1star_reviews.csv', index=False)

print("Top 500 1-star reviews with most likes exported successfully!")

# Clean the content column
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Tokenize and remove stopwords
        text_tokens = word_tokenize(text)
        text = ' '.join([word for word in text_tokens if word not in stop_words])
        return text
    else:
        # Handle non-string values (e.g., return as is or handle differently)
        return str(text)  # converting non-string to string if found

# Apply text cleaning
df['clean_content'] = df['content'].apply(clean_text)

# Check cleaned data
print(df[['content', 'clean_content', 'score']].head(10))

# Apply TF-IDF vectorization on the cleaned content to track bigrams or trigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=50, ngram_range=(2,3))  # For bigrams and trigrams

# Fit and transform the cleaned content into TF-IDF features
X_tfidf = vectorizer.fit_transform(df['clean_content'])

# Get the feature names (i.e., bigrams and trigrams)
feature_names = vectorizer.get_feature_names_out()

# Sum the TF-IDF values across all documents to get the importance of each bigram/trigram
tfidf_scores = X_tfidf.sum(axis=0).A1  # .A1 converts the matrix to a 1D array

# Create a DataFrame to display bigrams/trigrams with their corresponding scores
tfidf_df = pd.DataFrame(list(zip(feature_names, tfidf_scores)), columns=['Phrase', 'TF-IDF Score'])

# Sort the DataFrame to show the most prominent bigrams/trigrams
tfidf_df = tfidf_df.sort_values(by='TF-IDF Score', ascending=False)

# Display the top 30 most prominent phrases
print(tfidf_df.head(30))

# Save top 30 TF-IDF phrases to CSV
top_30_tfidf = tfidf_df.head(30)
top_30_tfidf.to_csv('/app/data/top_30_tfidf_phrases.csv', index=False)

# Plot the top 10 most prominent features
top_tfidf_df = tfidf_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(top_tfidf_df['Phrase'], top_tfidf_df['TF-IDF Score'], color='skyblue')
plt.xlabel('TF-IDF Score')
plt.title('Top 10 Most Prominent Features (Words) in 1-Star Reviews')
plt.gca().invert_yaxis()  # Reverse the order for a better visual
plt.tight_layout()
plt.show()

print("Process Complete!")