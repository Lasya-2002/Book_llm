# Functions for the UI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import warnings as wr
from BERT_database_operations import load_faiss_index
import faiss

wr.filterwarnings('ignore')

# Function to extract embeddings from the BERT model
def get_input_embedding(text):
    tokenizer = AutoTokenizer.from_pretrained("sri-lasya/book-recommender-bert")
    model = AutoModel.from_pretrained("sri-lasya/book-recommender-bert")
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Pass tokens through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the last hidden state (you can use other layers as well)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Convert embeddings to numpy array and reshape it to 2D
    return embeddings

# Function to fetch category embedding for a specific book
def get_category_embedding(book_title):
    try:
        conn = sqlite3.connect("book_recommendations.db")
        cursor = conn.cursor()

        query = """
        SELECT category_embedding
        FROM books
        WHERE title = ?
        """
        cursor.execute(query, (book_title,))
        result = cursor.fetchone()
        conn.close()

        if result:
            # Convert BLOB category embedding to numpy array
            category_embedding = np.frombuffer(result[0], dtype=np.float32)
            return category_embedding
        else:
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to get input embeddings for the book title and category
def get_input_embeddings_with_category(book_title, category_title):
    # Extract the title embedding (using your existing logic)
    title_embedding = get_input_embedding(book_title)  # Title embedding from BERT
    
    if title_embedding is None:
        return None  # If title embedding is not found
    
    # Now, get the category embedding from the database
    category_embedding = get_category_embedding(category_title)
    
    if category_embedding is None:
        category_embedding = get_input_embedding('category_title') # Use zero vector as a placeholder
    
    category_embedding=category_embedding.reshape(1,-1)
    
    author_embedding=np.zeros_like(title_embedding)
    author_embedding=author_embedding.reshape(1,-1)

    book_embedding=np.zeros_like(title_embedding)
    book_embedding=book_embedding.reshape(1,-1)
    # Combine embeddings for title and category
    combined_embedding = np.concatenate([title_embedding, category_embedding,book_embedding,author_embedding])

    return combined_embedding

# Function to fetch embeddings for all books in the database
def get_all_book_embeddings():
    try:
        conn = sqlite3.connect("book_recommendations.db")
        cursor = conn.cursor()

        query = """
        SELECT title, title_embedding, category_embedding, book_embedding, author_embedding
        FROM books
        """
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        all_embeddings = []
        for row in results:
            title, title_embedding, category_embedding, book_embedding, author_embedding = row
            # Convert BLOB embeddings to numpy arrays
            title_embedding = np.frombuffer(title_embedding, dtype=np.float32)
            category_embedding = np.frombuffer(category_embedding, dtype=np.float32)
            book_embedding = np.frombuffer(book_embedding, dtype=np.float32)
            author_embedding = np.frombuffer(author_embedding, dtype=np.float32)
            # Combine embeddings into a single vector
            combined_embedding = np.concatenate([title_embedding, category_embedding, book_embedding, author_embedding])
            all_embeddings.append((title, combined_embedding))

        return all_embeddings
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

import numpy as np
import faiss
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def get_recommendations_by_embeddings(input_embedding, top_n=5, rating_weight=0.5, count_weight=0.5):
    try:
        # Ensure input_embedding is 2D (1, N)
        input_embedding = input_embedding.reshape(1, -1)
        
        # Normalize the input embedding for cosine similarity
        faiss.normalize_L2(input_embedding)

        # Fetch all book embeddings from the database
        all_embeddings = get_all_book_embeddings()

        if not all_embeddings:
            return []

        # Extract titles and embeddings
        titles = [item[0] for item in all_embeddings]
        db_embeddings = [item[1] for item in all_embeddings]

        # Ensure all embeddings are 2D
        db_embeddings = np.array(db_embeddings)
        if db_embeddings.ndim == 1:
            db_embeddings = db_embeddings.reshape(-1, db_embeddings.shape[1])

        # Normalize database embeddings
        faiss.normalize_L2(db_embeddings)

        # Compute cosine similarity between input embedding and all book embeddings
        similarity_scores = cosine_similarity(input_embedding, db_embeddings)[0]

        # Fetch book details for weighting
        conn = sqlite3.connect("book_recommendations.db")
        cursor = conn.cursor()
        
        books_data = {}
        for title in titles:
            query = """
            SELECT title, average_rating, ratings_count FROM books WHERE title = ?
            """
            cursor.execute(query, (title,))
            result = cursor.fetchone()
            if result:
                books_data[title] = {
                    "average_rating": result[1] if result[1] else 0,
                    "ratings_count": result[2] if result[2] else 0
                }
        conn.close()

        # Normalize ratings and counts
        ratings = np.array([books_data[t]["average_rating"] for t in titles])
        counts = np.array([books_data[t]["ratings_count"] for t in titles])
        
        if np.max(ratings) > 0:
            ratings /= np.max(ratings)  # Normalize to range [0,1]
        if np.max(counts) > 0:
            counts /= np.max(counts)  # Normalize to range [0,1]

        # Compute weighted similarity score
        weighted_scores = similarity_scores + rating_weight * ratings + count_weight * counts

        # Sort books by weighted scores (descending order)
        sorted_indices = np.argsort(weighted_scores)[::-1]
        top_indices = sorted_indices[:top_n]

        # Fetch book details for the top recommendations
        recommendations = []
        conn = sqlite3.connect("book_recommendations.db")
        cursor = conn.cursor()
        
        for idx in top_indices:
            title = titles[idx]
            query = """
            SELECT title, author, description, categories, average_rating, ratings_count
            FROM books
            WHERE title = ?
            """
            cursor.execute(query, (title,))
            result = cursor.fetchone()
            if result:
                recommendations.append(result)

        conn.close()
        return recommendations
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []
