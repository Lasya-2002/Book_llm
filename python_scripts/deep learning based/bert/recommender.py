import numpy as np
import faiss
import sqlite3
import pandas as pd


# Load embeddings from SQLite database
def load_embeddings():
    conn = sqlite3.connect("book_embeddings.db")
    c = conn.cursor()
    c.execute("SELECT id, title, title_embedding, description_embedding, category_embedding FROM book_embeddings")
    data = c.fetchall()
    conn.close()

    book_titles = []
    title_embeddings = []
    desc_embeddings = []
    cat_embeddings = []

    for row in data:
        book_titles.append(row[1].strip().lower())
        title_embeddings.append(np.frombuffer(row[2], dtype=np.float32))
        desc_embeddings.append(np.frombuffer(row[3], dtype=np.float32))
        cat_embeddings.append(np.frombuffer(row[4], dtype=np.float32))

    # Stack embeddings into numpy arrays
    title_embeddings = np.vstack(title_embeddings)
    desc_embeddings = np.vstack(desc_embeddings)
    cat_embeddings = np.vstack(cat_embeddings)

    return book_titles, title_embeddings, desc_embeddings, cat_embeddings


# Build or load multiple FAISS indexes
def build_faiss_indexes(embeddings_dict):
    indexes = {}

    for key, embeddings in embeddings_dict.items():
        dimension = embeddings.shape[1]

        # Define FAISS index configurations
        index_configs = {
            f"{key}_FlatL2": faiss.IndexFlatL2(dimension),
            f"{key}_HNSW": faiss.IndexHNSWFlat(dimension, 32),  # M = 32
            f"{key}_IVFFlat": create_ivf_index(embeddings, dimension)
        }

        for index_type, index in index_configs.items():
            index_file = f"book_index_{index_type.lower()}.faiss"

            try:
                # Try to load existing index if available
                index = faiss.read_index(index_file)
                print(f"{index_type} FAISS index loaded successfully!")
            except Exception:
                print(f"{index_type} FAISS index not found. Building a new one...")
                index.add(embeddings)
                faiss.write_index(index, index_file)
                print(f"{index_type} FAISS index built and saved successfully!")

            indexes[index_type] = index

    return indexes


# Create an IVF Index with training
def create_ivf_index(embeddings, dimension, nlist=100):
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index_ivf.train(embeddings)
    return index_ivf


def calculate_mrr(results_list, target_title):
    target_title = target_title.strip().lower()

    for rank, rec in enumerate(results_list, start=1):
        if isinstance(rec, dict) and rec.get("title", "").strip().lower() == target_title:
            return 1 / rank

    return 0.0


def calculate_map(results_list, target_title):
    target_title = target_title.strip().lower()
    relevant_count, ap_sum = 0, 0.0

    for rank, rec in enumerate(results_list, start=1):
        if isinstance(rec, dict) and rec.get("title", "").strip().lower() == target_title:
            relevant_count += 1
            ap_sum += relevant_count / rank

    return ap_sum / relevant_count if relevant_count > 0 else 0.0



# Get recommendations with separate scores for each embedding
def get_separate_scores_recommendations(book_title, book_titles, embeddings_dict, indexes_dict, k=5):
    recommendations = {}

    try:
        # Get the index of the book title
        book_idx = book_titles.index(book_title)

        # Get query vectors for title, description, and category embeddings
        query_vectors = {key: embeddings[book_idx].reshape(1, -1) for key, embeddings in embeddings_dict.items()}

        for index_type in ["FlatL2", "HNSW", "IVFFlat"]:
            results = []

            for i, (key, query_vector) in enumerate(query_vectors.items()):
                index_name = f"{key}_{index_type}"

                if index_name in indexes_dict:
                    index = indexes_dict[index_name]
                    distances, indices = index.search(query_vector, k)
                    similarity_scores = 1 / (1 + distances[0])  # Convert L2 distance to similarity

                    for j, idx in enumerate(indices[0]):
                        if idx < len(book_titles):
                            # Check if this recommendation already exists
                            if idx >= len(results):
                                results.append({
                                    "title": book_titles[idx],
                                    "Title_Score": 0,
                                    "Description_Score": 0,
                                    "Category_Score": 0,
                                    "Average_Score": 0
                                })

                            # Assign scores to appropriate columns
                            if key == "title":
                                results[j]["Title_Score"] = round(similarity_scores[j], 4)
                            elif key == "description":
                                results[j]["Description_Score"] = round(similarity_scores[j], 4)
                            elif key == "categories":
                                results[j]["Category_Score"] = round(similarity_scores[j], 4)

            # Calculate average score for final ranking
            for result in results:
                result["Average_Score"] = round(
                    (result["Title_Score"] + result["Description_Score"] + result["Category_Score"]) / 3, 4
                )

            # Sort results by average score
            results = sorted(results, key=lambda x: x["Average_Score"], reverse=True)

            # Calculate MRR and MAP for evaluation
            mrr = calculate_mrr(results, book_title)
            map_score = calculate_map(results, book_title)

            print(f"\n{index_type} - MRR: {round(mrr, 4)}, MAP: {round(map_score, 4)}")

            # Create a DataFrame from results
            df = pd.DataFrame(results)
            recommendations[index_type] = df

    except ValueError:
        print(f"Book '{book_title}' not found in the dataset.")
        return {}

    return recommendations
