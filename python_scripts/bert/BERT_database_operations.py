import torch
import faiss
import numpy as np
import sqlite3
from transformers import AutoModel, AutoTokenizer

# Load embeddings from the database
def load_embeddings():
    conn1 = sqlite3.connect("file:book_recommendations.db?mode=ro", uri=True)
    cursor = conn1.cursor()
    cursor.execute("SELECT title, author, description, average_rating, ratings_count, categories, book_embedding, author_embedding, title_embedding, category_embedding FROM books")
    records = cursor.fetchall()
    books = []
    book_embeddings = []
    author_embeddings = []
    title_embeddings = []
    category_embeddings = []
    for record in records:
        books.append({
            "Name": record[0],
            "Author": record[1],
            "Description": record[2],
            "AverageRating": record[3],
            "RatingsCount": record[4],
            "Categories": record[5]
        })
        book_embeddings.append(np.frombuffer(record[6], dtype=np.float32))
        author_embeddings.append(np.frombuffer(record[7], dtype=np.float32))
        title_embeddings.append(np.frombuffer(record[8], dtype=np.float32))
        category_embeddings.append(np.frombuffer(record[9], dtype=np.float32))
    conn1.close()
    return books, np.array(book_embeddings), np.array(author_embeddings), np.array(title_embeddings), np.array(category_embeddings)

# Create and save FAISS index
def create_and_save_faiss_index(embeddings, index_file_path):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # Using L2 distance
    index.add(embeddings)
    faiss.write_index(index, index_file_path)
    print(f"FAISS index saved to {index_file_path}")

# Load FAISS index from disk
def load_faiss_index(index_file_path):
    index = faiss.read_index(index_file_path)
    print(f"FAISS index loaded from {index_file_path}")
    return index

# Search for similar vectors using the FAISS index
def search_similar_vectors(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return distances, indices

if __name__ == "__main__":
    books, book_embeddings, author_embeddings, title_embeddings, category_embeddings = load_embeddings()
    
    # Create and save FAISS index
    create_and_save_faiss_index(book_embeddings, "book_faiss.index")
    create_and_save_faiss_index(author_embeddings, "author_faiss.index")
    create_and_save_faiss_index(title_embeddings, "title_faiss.index")
    create_and_save_faiss_index(category_embeddings, "category_faiss.index")
    
    # Load FAISS index
    book_index = load_faiss_index("book_faiss.index")

