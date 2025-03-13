import torch
import faiss
import numpy as np
import sqlite3
import os

# Function to fetch embeddings for all books in the database and create the FAISS index
def create_faiss_index():
    try:
        conn = sqlite3.connect("book_recommendations.db")
        cursor = conn.cursor()

        query = """
        SELECT title, author, description, categories, average_rating, ratings_count, 
               title_embedding, category_embedding, book_embedding, author_embedding
        FROM books
        """
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        all_embeddings = []
        book_details = []  # To store book details along with embeddings
        for row in results:
            title, author, description, categories, average_rating, ratings_count, \
            title_embedding, category_embedding, book_embedding, author_embedding = row
            
            # Convert BLOB embeddings to numpy arrays
            title_embedding = np.frombuffer(title_embedding, dtype=np.float32)
            category_embedding = np.frombuffer(category_embedding, dtype=np.float32)
            book_embedding = np.frombuffer(book_embedding, dtype=np.float32)
            author_embedding = np.frombuffer(author_embedding, dtype=np.float32)
            
            # Combine embeddings into a single vector
            combined_embedding = np.concatenate([title_embedding, category_embedding, book_embedding, author_embedding])
            all_embeddings.append(combined_embedding)
            
            # Save the book details
            book_details.append((title, author, description, categories, average_rating, ratings_count))
        
        # Convert the list of embeddings to a numpy array
        all_embeddings = np.array(all_embeddings).astype(np.float32)

        # Create a FAISS index
        d = all_embeddings.shape[1]  # The dimension of the embeddings
        index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity
        index.add(all_embeddings)  # Add the embeddings to the FAISS index

        # Save the index to a file
        faiss.write_index(index, 'book_recommender.index')  # Write to file
        print("FAISS index saved to disk.")
        return index, book_details
    except Exception as e:
        print(f'Exception : {e}')
        return None, []

# Function to load FAISS index from disk
def load_faiss_index():
    try:
        if os.path.exists('book_recommender.index'):
            index = faiss.read_index('book_recommender.index')  # Load the FAISS index from disk
            print("FAISS index loaded from disk.")
            return index
        else:
            print('there is an error in index loading')
            return None
    except Exception as e:
        print(f"An error occurred while loading the FAISS index: {e}")
        return None

if __name__ == '__main__':
    books_metadata=create_faiss_index()