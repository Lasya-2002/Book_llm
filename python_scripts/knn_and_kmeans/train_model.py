import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from joblib import parallel_backend
import warnings as wr

wr.filterwarnings('ignore')

# Disable loky multiprocessing on Windows
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# Load dataset
def load_dataset():
    df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")
    df.drop(['Published Date', 'Page Count'], axis=1, inplace=True)
    df.dropna(subset=['Title', 'Description', 'Categories'], inplace=True)  # Drop missing values

    # Convert to lowercase
    df['Title'] = df['Title'].str.lower()
    df['Description'] = df['Description'].str.lower()
    df['Categories'] = df['Categories'].str.lower()

    return df

# Vectorization
def vectorize_text(df):
    title_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1,2), dtype=np.float32)
    desc_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(4,5), max_features=100000, dtype=np.float32)
    
    title_x = title_vectorizer.fit_transform(df['Title'])
    desc_x = desc_vectorizer.fit_transform(df['Description'])

    sparse_vectors = hstack([title_x, desc_x], format='csr')
    
    batch_size = 500000  # Process 500K rows at a time
    svd = TruncatedSVD(n_components=100)

    dense_vectors = []
    for i in range(0, sparse_vectors.shape[0], batch_size):
        batch = sparse_vectors[i:i + batch_size]
        transformed_batch = svd.fit_transform(batch)
        dense_vectors.append(transformed_batch)

    book_vectors = np.vstack(dense_vectors)
    
    return title_vectorizer, desc_vectorizer, book_vectors

# Elbow Method for Optimal K
def find_optimal_k(book_vectors, max_k=15):
    inertia_values = []
    silhouette_scores = []

    k_values = range(2, max_k + 1)  # Start from k=2 to avoid trivial cases

    for k in k_values:
        with parallel_backend('threading'):  # Use threading backend
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(book_vectors)

        inertia_values.append(kmeans.inertia_)

        # Calculate silhouette score
        if k > 1:
            silhouette_avg = silhouette_score(book_vectors, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='b', label="Inertia")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.legend()
    plt.show()

    # Plot Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(k_values[1:], silhouette_scores, marker='s', linestyle='-', color='g', label="Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Analysis")
    plt.legend()
    plt.show()

# Train final model
def train_model(optimal_k):
    df = load_dataset()
    title_vectorizer, desc_vectorizer, book_vectors = vectorize_text(df)

    # Train K-Means
    with parallel_backend('threading'):  # Use threading backend
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(book_vectors)

    # Train KNN for recommendations
    knn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=1)  # Set n_jobs=1 to prevent issues
    knn.fit(book_vectors)

    # Save trained model
    with open("trained_model.pkl", "wb") as model_file:
        pickle.dump((df, title_vectorizer, desc_vectorizer, kmeans, knn, book_vectors), model_file)

    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    df = load_dataset()
    _, _, book_vectors = vectorize_text(df)
    
    print("🔍 Running Elbow Method to find optimal k...")
    find_optimal_k(book_vectors)

    optimal_k = int(input("Enter the optimal number of clusters (k) from the Elbow Method: "))
    if optimal_k < 2 or optimal_k > 15:
        print("Invalid value for k. Please choose a value between 2 and 15.")
    else:
        train_model(optimal_k)