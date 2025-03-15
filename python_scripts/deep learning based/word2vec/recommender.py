import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import re

def load_models():
    title_model = Word2Vec.load("title_word2vec.model")
    desc_model = Word2Vec.load("desc_word2vec.model")
    return title_model, desc_model

# Load dataset
def load_data():
    df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")
    return df

df = load_data()

def get_description_vector(description_text, model):
    if not isinstance(description_text, str) or not description_text.strip():
        return None
    words = word_tokenize(description_text.lower())
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else None

def recommend_books(title_model, desc_model, book_title, df, top_n=5,
                    title_weight=0.6, desc_weight=0.4, category_filter=True,
                    author_filter=False, ratings_filter=True):

    book_info = df[df['Title'] == book_title]
    print("Checking if book info exists in data frame")
    if book_info.empty:
        print(f"Book title '{book_title}' not found in dataset")
        return pd.DataFrame()
    print("checking if the book can be found by similar titles")
    book_category = book_info['Categories'].values[0]
    book_author = book_info['Authors'].values[0]
    book_rating = book_info['Average Rating'].values[0]

    title_similar, desc_similar = [], []

    book_title_tokens = word_tokenize(book_title.lower())
    for token in book_title_tokens:
        if token in title_model.wv:
            title_similar.extend(title_model.wv.most_similar(token, topn=top_n))
    
    if not title_similar:
        print("No title-based recommendations found.")
    
    print(title_similar)

    description_text = df.loc[df["Title"] == book_title, "Description"].values
    if len(description_text) == 0:
        return pd.DataFrame()
    print("Checking description based recommendations are possible...")
    desc_vector = get_description_vector(description_text[0], desc_model)
    if desc_vector is not None:
        desc_similar = desc_model.wv.similar_by_vector(desc_vector, topn=top_n)
    else:
        print("No description-based recommendations found.")
    print(desc_similar)

    combined_similarities = {}
    for word, score in title_similar:
        combined_similarities[word] = combined_similarities.get(word, 0) + score * title_weight
    for word, score in desc_similar:
        combined_similarities[word] = combined_similarities.get(word, 0) + score * desc_weight
    print(combined_similarities)
    print("Checking similarities for book recommendations")
    recommend_titles = sorted(combined_similarities.items(), key=lambda x: x[1], reverse=True)
    recommend_titles = [rec[0] for rec in recommend_titles[:top_n]]
    print(recommend_titles)
    pattern = "|".join(map(re.escape, recommend_titles))
    recommended_books = df[df['Title'].str.contains(pattern, case=False, na=False, regex=True)]
    print(recommended_books)

    print("applying filters...")
    # Apply filters
    if category_filter:
        recommended_books = recommended_books[recommended_books['Categories'] == book_category]
    if author_filter:
        recommended_books = recommended_books[recommended_books['Authors'] == book_author]
    if ratings_filter:
        recommended_books = recommended_books[recommended_books['Average Rating'] >= book_rating]

    return recommended_books[['Title', 'Authors', 'Categories', 'Average Rating']].head(top_n)
