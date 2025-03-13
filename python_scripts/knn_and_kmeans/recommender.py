import pickle
import numpy as np
import pandas as pd

# Load the trained model
def load_trained_model():
    with open("trained_model.pkl", "rb") as model_file:
        df, title_vectorizer, desc_vectorizer, kmeans, knn, book_vectors = pickle.load(model_file)
    return df, title_vectorizer, desc_vectorizer, kmeans, knn, book_vectors

# Recommend books using the trained model
def recommend_books(book_title, n=5, filter_category="Any", filter_author="Any", min_rating=0.0):
    df, title_vectorizer, desc_vectorizer, kmeans, knn, book_vectors = load_trained_model()

    book_title = book_title.lower()

    # ✅ Check if the book exists in the dataset
    if book_title not in df['Title'].values:
        print(f"⚠️ Book '{book_title}' not found! Suggesting category-based recommendations...")

        # Get most relevant category (if available)
        category = df['Categories'].value_counts().idxmax() if not df['Categories'].isna().all() else None

        if category:
            category_books = df[df['Categories'] == category].nlargest(n, 'Average Rating')
            recommendations = [
                (row['Title'], row['Authors'] if pd.notna(row['Authors']) else "Unknown", 
                 row['Average Rating'] if pd.notna(row['Average Rating']) else 0, 0.0)
                for _, row in category_books.iterrows()
            ]
        else:
            # If no category is found, return top-rated books
            top_books = df.nlargest(n, 'Average Rating')
            recommendations = [
                (row['Title'], row['Authors'] if pd.notna(row['Authors']) else "Unknown", 
                 row['Average Rating'] if pd.notna(row['Average Rating']) else 0, 0.0)
                for _, row in top_books.iterrows()
            ]

        return recommendations if recommendations else [("No recommendations available!", "N/A", 0, 0.0)]

    # If book is found, proceed with KNN recommendations
    book_index = df[df['Title'] == book_title].index[0]
    book_cluster = df.iloc[book_index]['Cluster']

    # Get books from the same cluster
    cluster_books = df[df['Cluster'] == book_cluster]

    if len(cluster_books) < n:
        return [("No books found in this cluster!", "N/A", 0, 0.0)]

    # Find KNN recommendations
    distances, indices = knn.kneighbors([book_vectors[book_index]])
    recommendations = []

    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        title = df.iloc[idx]['Title']
        author = df.iloc[idx]['Authors'] if pd.notna(df.iloc[idx]['Authors']) else "Unknown"
        rating = df.iloc[idx]['Average Rating'] if pd.notna(df.iloc[idx]['Average Rating']) else 0
        similarity = 1 - distances[0][i]  # Convert distance to similarity

        # Apply filters
        if filter_category != "Any" and df.iloc[idx]['Categories'] != filter_category:
            continue
        if filter_author != "Any" and author != filter_author:
            continue
        if rating < min_rating:
            continue

        recommendations.append((title, author, rating, similarity))

        if len(recommendations) >= n:
            break

    return recommendations if recommendations else [("No books found after filtering!", "N/A", 0, 0.0)]
