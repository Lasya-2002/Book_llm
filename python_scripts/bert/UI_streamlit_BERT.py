from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import warnings as wr

wr.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained("sri-lasya/book-recommender-bert")
model = AutoModel.from_pretrained("sri-lasya/book-recommender-bert")

# Function to extract embeddings from the BERT model
def get_input_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Pass tokens through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the last hidden state (you can use other layers as well)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pooling over all tokens

    # Convert embeddings to numpy array and reshape it to 2D
    return embeddings.numpy().reshape(1, -1)


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
        st.warning(f"Could not find category embedding for the category: {category_title}")
        category_embedding = np.zeros_like(title_embedding)  # Use zero vector as a placeholder
    
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


def get_recommendations_by_embeddings(input_embedding, top_n=5):
    try:
        # Ensure input_embedding is 2D (1, N)
        input_embedding = input_embedding.reshape(1, -1)
        
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

        # Compute cosine similarity between input embedding and all book embeddings
        similarity_scores = cosine_similarity(input_embedding, db_embeddings)[0]

        # Sort books by similarity scores (descending order)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        top_indices = sorted_indices[:top_n]

        # Fetch book details for the top recommendations
        conn = sqlite3.connect("book_recommendations.db")
        cursor = conn.cursor()

        recommendations = []
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

# Function to create a book recommendation card
def create_card(title, author, description, categories, average_rating, ratings_count):
    return f"""
    <div style="
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        color: black;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;">
        <h4 style="margin: 0;color: black; ">📖 <b>{title}</b></h4>
        <p style="margin: 5px 0;"><b>Author:</b> {author}</p>
        <p style="margin: 5px 0;"><b>Categories:</b> {categories}</p>
        <p style="margin: 5px 0;"><b>Description:</b> {description[:200]}...</p>
        <p style="margin: 5px 0;">
            <b>⭐ Rating:</b> {average_rating} | <b>Reviews:</b> {ratings_count}
        </p>
    </div>
"""

# Streamlit UI setup
st.set_page_config(page_title="Book Recommendation System", page_icon="📚")
st.title("📚 Book Recommendation System")

book_title = st.text_input("Enter book title (required):")
book_category = st.text_input("Enter book category (required):")

if st.button("Get Recommendations"):
    if not book_category:
        st.warning("Please enter a book category.")
    else:
        # Fetch embeddings for the input book (if title is provided)
        if book_title:
            combined_embeddings = get_input_embeddings_with_category(book_title, book_category)
            
            if combined_embeddings is None:
                st.write("Could not generate embeddings. Please check the input.")
            else:
                recommendations = get_recommendations_by_embeddings(combined_embeddings)
                if recommendations:
                    st.subheader("📌 Recommended Books (based on title and category):")
                    for title, author, description, categories, avg_rating, ratings_count in recommendations:
                        st.markdown(create_card(title, author, description, categories, avg_rating, ratings_count), unsafe_allow_html=True)
                else:
                    st.write("No recommendations found.")
        else:
            st.warning("Please enter a book title.")
