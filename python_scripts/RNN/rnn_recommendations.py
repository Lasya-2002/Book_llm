import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import streamlit as st

df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv", low_memory=False)

# Load stored embeddings and similarity matrix
book_embeddings = np.load("book_embeddings.npy")
category_embeddings = np.load("category_embeddings.npy")
similarity_matrix = np.load("similarity_matrix.npy")

# Function to recommend books based on title
def recommend_books_by_title(book_title, df, similarity_matrix, top_n=5):
    if book_title not in df['Title'].values:
        return None
    
    book_index = df[df['Title'] == book_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[book_index]))
    sorted_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_books = [df.iloc[i[0]]['Title'] for i in sorted_books]
    return recommended_books

# Function to recommend books based on category
def recommend_books_by_category(category, df, category_embeddings, top_n=5):
    if category not in df['Category'].values:
        return ["Category not found in dataset."]
    
    category_index = df[df['Category'] == category].index[0]
    category_embedding = category_embeddings[category_index]
    
    # Calculate similarity between the category embedding and all book embeddings
    similarity_scores = cosine_similarity([category_embedding], book_embeddings)
    sorted_books = sorted(enumerate(similarity_scores[0]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_books = [df.iloc[i[0]]['Title'] for i in sorted_books]
    return recommended_books

# Streamlit UI
st.title("📚 Book Recommendation System")

# User input for book title and category
book_name = st.text_input("Enter a book title:")
category_name = st.text_input("Enter a book category:")

# Generate recommendations if input is provided
if st.button("Get Recommendations"):
    if book_name.strip():
        recommended_books = recommend_books_by_title(book_name.strip(), df, similarity_matrix)
        if recommended_books is None:
            st.write(f"'{book_name}' not found in the database. Trying category-based recommendations...")
            recommended_books = recommend_books_by_category(category_name.strip(), df, category_embeddings)
    elif category_name.strip():
        recommended_books = recommend_books_by_category(category_name.strip(), df, category_embeddings)
    else:
        recommended_books = []

    # Save recommendations to JSON
    with open("recommended_books.json", "w") as f:
        json.dump(recommended_books, f)
    
    st.subheader("📖 Recommended Books:")
    if recommended_books:
        for book in recommended_books:
            st.write(f"- {book}")
    else:
        st.write("No recommendations found.")
    
    print(f"\nRecommended books for '{book_name}':")
    for book in recommended_books:
        print("- " + book)
