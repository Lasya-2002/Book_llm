import streamlit as st
import numpy as np
import faiss
import sqlite3

# Load FAISS indices
book_index = faiss.read_index("book_faiss.index")
title_index = faiss.read_index("title_faiss.index")
category_index = faiss.read_index("category_faiss.index")


# Connect to SQLite database
conn = sqlite3.connect("book_recommendations.db")
cursor = conn.cursor()

# Fetch book metadata
cursor.execute("SELECT title, author, description, average_rating, ratings_count, categories FROM books")
records = cursor.fetchall()

# Convert records into a structured list
books = []
for record in records:
    books.append({
        "title": record[0],
        "author": record[1],
        "description": record[2],
        "average_rating": record[3],
        "ratings_count": record[4],
        "categories": record[5]
    })

# Save to a NumPy file
np.save("books_metadata.npy", books)


# Load metadata
books = np.load("books_metadata.npy", allow_pickle=True).tolist()

# Streamlit app
st.title("Book Recommendation System")

# User input
book_title = st.text_input("Enter book title:")
book_category = st.text_input("Enter book category (required):")

def get_recommendations(book_title, book_category, top_n=10):
    try:
        if not book_category:
            st.warning("Please enter a book category.")
            return []

        category_embedding = category_index.reconstruct(0).reshape(1, -1)
        category_distances, category_indices = category_index.search(category_embedding, top_n * 5)
        candidate_indices = category_indices.flatten().tolist()

        if book_title:
            title_embedding = title_index.reconstruct(0).reshape(1, -1)
            title_distances, title_indices = title_index.search(title_embedding, top_n * 5)
            candidate_indices.extend(title_indices.flatten().tolist())
            candidate_indices = list(set(candidate_indices))  # Remove duplicates

        recommendations = []
        for idx in candidate_indices:
            book = books[idx]
            avg_rating, ratings_count = book["average_rating"], book["ratings_count"]
            
            title_embedding = book_index.reconstruct(idx).reshape(1, -1)
            category_embedding = category_index.reconstruct(idx).reshape(1, -1)

            title_similarity = 1.0 if book_title else 0.0
            category_similarity = 1.0  # Precomputed from FAISS

            weight_rating, weight_count, weight_title, weight_category = 0.5, 0.3, 0.1, 0.1
            weighted_score = (weight_rating * avg_rating) + (weight_count * np.log1p(ratings_count)) + (weight_title * title_similarity) + (weight_category * category_similarity)
            
            recommendations.append((book["title"], book["author"], book["description"], book["categories"], weighted_score))

        recommendations.sort(key=lambda x: x[4], reverse=True)
        return recommendations[:top_n]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

if st.button("Get Recommendations"):
    recommendations = get_recommendations(book_title, book_category)

    if recommendations:
        st.subheader("Recommended Books:")
        for book_name, author, description, categories, score in recommendations:
            st.write(f"**{book_name}** by {author}")
            st.write(f"Categories: {categories}")
            st.write(f"Description: {description}")
            st.write(f"Score: {score:.2f}")
            st.write("---")
