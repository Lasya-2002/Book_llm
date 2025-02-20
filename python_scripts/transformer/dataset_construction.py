import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reload the FAISS index
index = faiss.read_index("books_faiss_description.index")
df = pd.read_csv("books_with_embeddings.csv")

def find_similar_books_by_description(query_embedding, df, top_k=2):
    """Find books with the most similar descriptions using FAISS, handling index errors."""
    D, I = index.search(np.array([query_embedding]), top_k + 1)  # +1 to exclude itself
    # Ensure valid indices within DataFrame bounds
    valid_indices = [idx for idx in I[0][1:] if 0 <= idx < len(df)]

    if not valid_indices:
        print("Warning: No valid similar books found.")
        return []  # Return an empty list if no valid indices

    similar_books = df.iloc[valid_indices]["Title"].tolist()
    return similar_books


def create_instruction(row, df, embedding_model):
    # Filter books by the same language
    same_language_books = df[df["Language"] == row["Language"]]

    # Prioritize books with high ratings and sufficient rating count
    high_rated_books = same_language_books[
        (same_language_books["Average Rating"] >= 4.0) & (same_language_books["Ratings Count"] >= 100)
    ]
    
    # Find books with similar categories (filtered by language)
    similar_category_books = same_language_books[same_language_books["Categories"] == row["Categories"]]
    similar_category_books = similar_category_books.sample(min(2, len(similar_category_books)))["Title"].tolist()

    # Find books by the same author (filtered by language)
    same_author_books = same_language_books[same_language_books["Authors"] == row["Authors"]]
    same_author_books = same_author_books.sample(min(2, len(same_author_books)))["Title"].tolist()

    # Find books with high ratings (filtered by language)
    high_rated_recommendations = high_rated_books.sample(min(2, len(high_rated_books)))["Title"].tolist()

    # Get books with similar descriptions using FAISS
    query_embedding = embedding_model.encode(str(row["Description"]), convert_to_numpy=True)
    similar_description_books = find_similar_books_by_description(query_embedding, same_language_books)

    return {
        "instruction": "Recommend books based on user query",
        "input": f"I like books similar to '{row['Title']}' by {row['Authors']} in {row['Language']}. Can you recommend more?",
        "output": {
            "book_itself": row["Title"],
            "categories": row["Categories"],
            "average_rating": row["Average Rating"],
            "ratings_count": row["Ratings Count"],
            "language": row["Language"],
            "similar_category_books": similar_category_books,
            "same_author_books": same_author_books,
            "similar_description_books": similar_description_books,
            "high_rated_recommendations": high_rated_recommendations
        }
    }


# Process all books
instructions = [create_instruction(row, df, embedding_model) for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Instructions")]

# Save as JSONL
output_file = "instruction_dataset.jsonl"
with open(output_file, "w") as f:
    for entry in instructions:
        f.write(json.dumps(entry) + "\n")

print(f"Instruction dataset saved as {output_file} ✅")


print("Conversion complete! Your JSONL dataset is ready for fine-tuning.")
