import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/ChitranjanUpadhayay/ML_Projects/main/Datasets/Movies%20Recommendation%20System/dataset.csv")

# Data Preprocessing
df['tags'] = df['genre'] + df['overview']
df = df[['title', 'tags']].dropna()
df['tags'] = df['tags'].str.lower()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vec = tfidf.fit_transform(df['tags'].values.astype('U'))  

# Compute Cosine Similarity
sim = cosine_similarity(vec)

# Function to Recommend Movies
def recommend(movie_name):
    if movie_name not in df['title'].values:
        return ["Movie not found in database!"]
    
    movie_index = df[df['title'] == movie_name].index[0]
    distances = sim[movie_index]
    movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]  # Top 5 movies

    return [df.iloc[i[0]]['title'] for i in movie_list]

# Streamlit UI
st.title("Movie Recommender System 🎬")
movie_input = st.text_input("Enter a Movie Name", "")

if st.button("Recommend"):
    if movie_input:
        recommendations = recommend(movie_input)
        st.write("### Recommended Movies:")
        for movie in recommendations:
            st.write(f"✔ {movie}")
