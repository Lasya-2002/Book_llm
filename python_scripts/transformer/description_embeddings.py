from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/Google_books_api_dataset.csv")

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode book descriptions into embeddings
df["description_embedding"] = df["Description"].apply(lambda x: embedding_model.encode(str(x), convert_to_numpy=True))

# Convert to FAISS format
embeddings_matrix = np.vstack(df["description_embedding"].values)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

# Save FAISS index for later use
faiss.write_index(index, "books_faiss_description.index")
df.to_csv("books_with_embeddings.csv", index=False)

print("Book description embeddings stored in FAISS!")
