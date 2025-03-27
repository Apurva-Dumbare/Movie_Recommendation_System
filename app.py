# # using .pkl files but it causing issue while uploading to github as it allows only upto size of 100 mb and those pickle files are excedding the limit 


# import pandas as pd
# import numpy as np
# import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle

# # Load dataset 
# df = pd.read_csv("https://raw.githubusercontent.com/ChitranjanUpadhayay/ML_Projects/refs/heads/main/Datasets/Movies%20Recommendation%20System/dataset.csv")  # Ensure your dataset has 'title' and 'tags' columns

# df['tags'] = df['genre']+ df['overview']
# df = df[['title', 'tags']].dropna()

# # Convert tags to lowercase for consistency
# df['tags'] = df['tags'].str.lower()

# # **TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
# vec = tfidf.fit_transform(df['tags'].values.astype('U'))  

# # Compute Cosine Similarity
# sim = cosine_similarity(vec)

# # Save the similarity matrix (to avoid recalculating each time)
# pickle.dump(sim, open("similarity.pkl", "wb"))
# pickle.dump(df, open("movies.pkl", "wb"))

# # Function to Recommend Movies
# def recommend(movie_name):
#     if movie_name not in df['title'].values:
#         return ["Movie not found in database!"]
    
#     movie_index = df[df['title'] == movie_name].index[0]
#     distances = sim[movie_index]
#     movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]  # Top 5 movies

#     return [df.iloc[i[0]]['title'] for i in movie_list]

# # Streamlit UI
# st.title("Movie Recommender System 🎬")
# movie_input = st.text_input("Enter a Movie Name", "")

# if st.button("Recommend"):
#     if movie_input:
#         recommendations = recommend(movie_input)
#         st.write("### Recommended Movies:")
#         for movie in recommendations:
#             st.write(f"✔ {movie}")

