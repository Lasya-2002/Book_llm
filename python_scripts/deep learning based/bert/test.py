from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from BERT_recommendations import get_all_book_embeddings
import numpy as np


def visualize_embeddings(embeddings, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(set(labels)):
        indices = [j for j in range(len(labels)) if labels[j] == label]
        plt.scatter(reduced[indices, 0], reduced[indices, 1], label=label)
    
    plt.legend()
    plt.show()

# Example usage:
all_embeddings = get_all_book_embeddings()
embeddings = np.array([emb[1] for emb in all_embeddings])
labels = [emb[0] for emb in all_embeddings]  # Book titles or categories
visualize_embeddings(embeddings, labels)
