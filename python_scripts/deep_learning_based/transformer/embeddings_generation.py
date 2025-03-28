from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm

# Load dataset
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate embeddings
def generate_embedding(text):
    return embedding_model.encode(str(text), convert_to_numpy=True)

tqdm.pandas()

# Encode title, description, and category separately
df["title_embedding"] = df["Title"].progress_apply(generate_embedding)
df["description_embedding"] = df["Description"].progress_apply(generate_embedding)
df["category_embedding"] = df["Categories"].progress_apply(generate_embedding)

# Convert to FAISS format
def create_faiss_index(embeddings):
    embeddings_matrix = np.vstack(embeddings.values)
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)
    return index

# Create and save FAISS indexes
indexes = {
    "title": create_faiss_index(df["title_embedding"]),
    "description": create_faiss_index(df["description_embedding"]),
    "category": create_faiss_index(df["category_embedding"])
}

faiss.write_index(indexes["title"], "books_faiss_title.index")
faiss.write_index(indexes["description"], "books_faiss_description.index")
faiss.write_index(indexes["category"], "books_faiss_category.index")

# Save dataset with embeddings
df.to_csv("books_with_embeddings.csv", index=False)

print("Title, description, and category embeddings stored in FAISS!")
