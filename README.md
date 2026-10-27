# Movie Recommendation System

Content-based movie recommendation system built with Python, Pandas, scikit-learn and Streamlit, deployed on Render.

## Dataset

- Name: TMDB 5000 Movie Dataset
- Kaggle link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- File used: tmdb_5000_movies.csv
- Item identifier: title
- Text features: overview, genres, keywords

## Preprocessing Steps

1. Combine overview, genres and keywords into one text.
2. Convert to lowercase.
3. Remove punctuation and special characters.
4. Remove stopwords.
5. Replace missing values with empty string.
6. Store result in clean_text.

## Why Cosine Similarity

Cosine similarity measures the angle between two TF-IDF vectors, not their length. So two movies with similar words get a high score even if one description is much longer than the other. Its value ranges from 0 to 1, which makes ranking similar movies simple.

## Run Locally

pip install -r requirements.txt
streamlit run app.py

## Deploy on Render

- Build Command: pip install -r requirements.txt
- Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

## Deployed URL

https://your-app-name.onrender.com
