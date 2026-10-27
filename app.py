
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_data():
    df = pd.read_csv("movies_clean.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    return df, tfidf_matrix


def recommend(item_name, top_n=5):
    index = df.index[df["title"] == item_name][0]
    scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    top = [i for i in scores.argsort()[::-1] if i != index][:top_n]
    return df["title"].iloc[top].tolist()


df, tfidf_matrix = load_data()

st.title("Movie Recommendation System")
selected = st.selectbox("Select a movie", df["title"].values)

if st.button("Recommend"):
    st.subheader(f"Movies similar to {selected}:")
    for i, movie in enumerate(recommend(selected), 1):
        st.write(f"{i}. {movie}")
