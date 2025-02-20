import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
file_path = "C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv"
df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
df_original = df.copy()

# Text columns for tokenization
text_columns = ['Title', 'Authors', 'Categories', 'Description', 'Publisher']
max_vocab_size = 5000
max_length = 50

# Tokenization
tokenizers = {}
tokenized_data = {}
for col in text_columns:
    tokenizer = {word: min(i+1, max_vocab_size-1) for i, word in enumerate(set(" ".join(df[col].astype(str)).split()))}
    tokenizers[col] = tokenizer
    tokenized_data[col] = [[tokenizer.get(word, 0) for word in str(text).split()] for text in df[col]]

# Pad sequences
def pad_sequences(sequences, maxlen):
    return [seq[:maxlen] + [0] * (maxlen - len(seq)) for seq in sequences]

for col in text_columns:
    tokenized_data[col] = pad_sequences(tokenized_data[col], max_length)

# Scale numerical features
scaler = MinMaxScaler()
df[['Page Count', 'Average Rating', 'Ratings Count']] = scaler.fit_transform(
    df[['Page Count', 'Average Rating', 'Ratings Count']]
)

# Encode language
df['Language'] = LabelEncoder().fit_transform(df['Language'])

# Encode categories
df['Categories'] = LabelEncoder().fit_transform(df['Categories'])

# Split dataset (80% training, 10% validation, 10% testing)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_tokenized_data = {}
val_tokenized_data = {}
test_tokenized_data = {}

for col in text_columns:
    tokenizer = {word: min(i+1, max_vocab_size-1) for i, word in enumerate(set(" ".join(train_df[col].astype(str)).split()))}
    train_tokenized_data[col] = [[tokenizer.get(word, 0) for word in str(text).split()] for text in train_df[col]]
    val_tokenized_data[col] = [[tokenizer.get(word, 0) for word in str(text).split()] for text in val_df[col]]
    test_tokenized_data[col] = [[tokenizer.get(word, 0) for word in str(text).split()] for text in test_df[col]]

# Pad sequences for train, validation, and test sets
for col in text_columns:
    train_tokenized_data[col] = pad_sequences(train_tokenized_data[col], max_length)
    val_tokenized_data[col] = pad_sequences(val_tokenized_data[col], max_length)
    test_tokenized_data[col] = pad_sequences(test_tokenized_data[col], max_length)

# Convert data to tensors and move to device
X_train_texts = [torch.tensor(train_tokenized_data[col], dtype=torch.long, device=device) for col in text_columns]
X_val_texts = [torch.tensor(val_tokenized_data[col], dtype=torch.long, device=device) for col in text_columns]
X_test_texts = [torch.tensor(test_tokenized_data[col], dtype=torch.long, device=device) for col in text_columns]

X_train_numeric = torch.tensor(train_df[['Page Count', 'Average Rating', 'Ratings Count']].values, dtype=torch.float32, device=device)
X_val_numeric = torch.tensor(val_df[['Page Count', 'Average Rating', 'Ratings Count']].values, dtype=torch.float32, device=device)
X_test_numeric = torch.tensor(test_df[['Page Count', 'Average Rating', 'Ratings Count']].values, dtype=torch.float32, device=device)

X_train_language = torch.tensor(train_df['Language'].values, dtype=torch.long, device=device)
X_val_language = torch.tensor(val_df['Language'].values, dtype=torch.long, device=device)
X_test_language = torch.tensor(test_df['Language'].values, dtype=torch.long, device=device)

X_train_categories = torch.tensor(train_df['Categories'].values, dtype=torch.long, device=device)
X_val_categories = torch.tensor(val_df['Categories'].values, dtype=torch.long, device=device)
X_test_categories = torch.tensor(test_df['Categories'].values, dtype=torch.long, device=device)

# Create DataLoader for training, validation, and testing sets
batch_size = 8
train_dataset = TensorDataset(*X_train_texts, X_train_numeric, X_train_language, X_train_categories)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(*X_val_texts, X_val_numeric, X_val_language, X_val_categories)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(*X_test_texts, X_test_numeric, X_test_language, X_test_categories)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define PyTorch model
class BookEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_numeric, num_languages, num_categories):
        super(BookEmbeddingModel, self).__init__()
        self.text_embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(len(text_columns))])
        self.lstm_layers = nn.ModuleList([nn.LSTM(embed_dim, 64, batch_first=True) for _ in range(len(text_columns))])
        self.numeric_dense = nn.Linear(num_numeric, 32)
        self.language_embedding = nn.Embedding(num_languages, 8)
        self.category_embedding = nn.Embedding(num_categories, 8)
        self.fc = nn.Sequential(
            nn.Linear(64 * len(text_columns) + 32 + 8 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, texts, numeric, language, categories):
        text_features = [rnn(embed(text))[0][:, -1, :] for embed, rnn, text in zip(self.text_embeddings, self.rnn_layers, texts)]
        text_concat = torch.cat(text_features, dim=1)
        numeric_out = self.numeric_dense(numeric)
        language_out = self.language_embedding(language).squeeze(1)
        category_out = self.category_embedding(categories).squeeze(1)
        x = torch.cat([text_concat, numeric_out, language_out, category_out], dim=1)
        return self.fc(x)
# Model initialization
vocab_size = max_vocab_size
embed_dim = 128
num_numeric = X_train_numeric.shape[1]
num_languages = df['Language'].nunique()
num_categories = df['Categories'].nunique()

torch.cuda.empty_cache()
torch.cuda.synchronize()

model = BookEmbeddingModel(vocab_size, embed_dim, num_numeric, num_languages, num_categories).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Evaluate on test set
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
with torch.no_grad():
    for test_batch in test_data_loader:
        test_texts = [test_batch[i] for i in range(len(text_columns))]
        test_numeric = test_batch[len(text_columns)]
        test_language = test_batch[-2]
        test_categories = test_batch[-1]

        test_output = model(test_texts, test_numeric, test_language, test_categories)
        test_target = model(test_texts, test_numeric, test_language, test_categories).detach()

        test_loss += criterion(test_output, test_target).item()
        test_correct += (torch.argmax(test_output, dim=1) == torch.argmax(test_target, dim=1)).sum().item()
        test_total += test_target.size(0)

test_accuracy = test_correct / test_total * 100
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


torch.cuda.empty_cache()

def get_book_embeddings(model, X_texts, X_numeric, X_language, X_categories):
    model.eval()
    with torch.no_grad():
        return model(X_texts, X_numeric, X_language, X_categories).cpu().numpy()

book_embeddings = get_book_embeddings(model, X_train_texts, X_train_numeric, X_train_language, X_train_categories)
category_embeddings = model.category_embedding.weight.cpu().numpy()

torch.save(model.state_dict(), "book_embedding_model.pth")
np.save("book_embeddings.npy", book_embeddings)
np.save("category_embeddings.npy", category_embeddings)

# Compute similarity matrix
similarity_matrix = cosine_similarity(book_embeddings)
np.save("similarity_matrix.npy", similarity_matrix)

# Clear unused memory
torch.cuda.empty_cache()