import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import streamlit as st

df=pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/Google_books_api_dataset.csv", low_memory=False)

# Load stored embeddings and similarity matrix
book_embeddings = np.load("book_embeddings.npy")
similarity_matrix = np.load("similarity_matrix.npy")

# Function to recommend books
def recommend_books(book_title, df, similarity_matrix, top_n=5):
    if book_title not in df['Title'].values:
        return ["Book not found in dataset."]
    
    book_index = df[df['Title'] == book_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[book_index]))
    sorted_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_books = [df.iloc[i[0]]['Title'] for i in sorted_books]
    return recommended_books

# Streamlit UI
st.title("📚 Book Recommendation System")

# User input for book title
book_name = st.text_input("Enter a book title:")

# Generate recommendations if input is provided
if st.button("Get Recommendations") and book_name.strip():
    recommended_books = recommend_books(book_name.strip(), df, similarity_matrix)
    
    # Save recommendations to JSON
    with open("recommended_books.json", "w") as f:
        json.dump(recommended_books, f)
    
    st.subheader(f"📖 Recommended Books for '{book_name}':")
    for book in recommended_books:
        st.write(f"- {book}")


with open("recommended_books.json", "w") as f:
    json.dump(recommended_books, f)

print(f"\nRecommended books for '{book_name}':")
for book in recommended_books:
    print("- " + book)