import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv('C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/google_books_api_dataset/Google_books_api_dataset.csv')


# Ensure no missing values in required columns
df = df.dropna(subset=['Title', 'Authors', 'Language', 'Categories', 'Description', 'Average Rating', 'Ratings Count'])

# Combine text columns into a single column for tokenization
df['Combined_Text'] = df['Title'].str.cat(df['Authors'], sep=' ').str.cat(df['Language'], sep=' ').str.cat(df['Description'], sep=' ')

# Tokenization and Sequence Preparation
max_vocab_size = 5000  # Limit vocabulary size
max_sequence_length = 100  # Maximum sequence length

# Tokenizer
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Combined_Text'])  # Fit on combined text
sequences = tokenizer.texts_to_sequences(df['Combined_Text'])  # Convert text to sequences

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Frequency Encoding
category_freq = df['Categories'].value_counts(normalize=True)
df['Category_Freq'] = df['Categories'].map(category_freq)

# Normalize Average Ratings, Ratings Count, and Category Labels
scaler = MinMaxScaler()
df[['Average Rating', 'Ratings Count', 'Category_Freq']] = scaler.fit_transform(df[['Average Rating', 'Ratings Count', 'Category_Freq']])

# Convert to PyTorch tensors
X = torch.tensor(padded_sequences, dtype=torch.long).to(device)
ratings = torch.tensor(df['Average Rating'].values, dtype=torch.float16).to(device)
ratings_count = torch.tensor(df['Ratings Count'].values, dtype=torch.float16).to(device)
category_labels = torch.tensor(df['Category_Freq'].values, dtype=torch.float16).to(device)

# Define the LSTM-based Embedding model
class LSTMEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

# Initialize the embedding model
embedding_dim = 128
hidden_size = 256
embedding_model = LSTMEmbeddingModel(max_vocab_size, embedding_dim, hidden_size).to(device)

torch.cuda.empty_cache()

# Extract book embeddings
with torch.no_grad():
    book_embeddings = embedding_model(X)

torch.cuda.empty_cache()

# Convert embeddings and other features to numpy for similarity computation
book_embeddings = book_embeddings.cpu().numpy()
ratings = ratings.cpu().numpy()
ratings_count = ratings_count.cpu().numpy()
category_labels = category_labels.cpu().numpy()

# Function to recommend books
def recommend_books(book_index, top_k=5, content_weight=0.6, rating_weight=0.2, count_weight=0.1, category_weight=0.1):
    """
    Recommend books based on a weighted combination of:
    - Content similarity (from embeddings)
    - Average Rating
    - Ratings Count
    - Category similarity
    """
    # Compute cosine similarity between the target book and all other books
    content_similarity = cosine_similarity(book_embeddings[book_index].reshape(1, -1), book_embeddings)
    
    # Normalize content similarity to [0, 1]
    content_similarity = (content_similarity - content_similarity.min()) / (content_similarity.max() - content_similarity.min())
    
    # Compute category similarity (absolute difference between category labels)
    category_similarity = 1 - np.abs(category_labels[book_index] - category_labels)
    
    # Combine similarities with weighted scores
    combined_scores = (
        content_weight * content_similarity +
        rating_weight * ratings +
        count_weight * ratings_count +
        category_weight * category_similarity
    )
    
    # Get the top-k most similar books
    top_k_indices = np.argsort(combined_scores[0])[-top_k-1:-1][::-1]
    return top_k_indices

# Example: Recommend books similar to the first book in the dataset
target_book_index = 0
recommended_indices = recommend_books(target_book_index)
print("Target Book:", df.iloc[target_book_index]['Title'])
print("Recommended Books:")
for idx in recommended_indices:
    print(f"{df.iloc[idx]['Title']} (Rating: {df.iloc[idx]['Average Rating']:.2f}, Ratings Count: {df.iloc[idx]['Ratings Count']:.2f}, Category: {df.iloc[idx]['Categories']})")