import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")

# Preprocessing
df = df.dropna(subset=['Title', 'Authors', 'Language', 'Categories', 'Description', 'Average Rating', 'Ratings Count'])
df['Combined_Text'] = df['Title'] + " " + df['Authors'] + " " + df['Language'] + " " + df['Description']

# Normalize numeric columns
df['Ratings Count'] = np.log1p(df['Ratings Count'])  # Log transform to reduce skew
df['Average Rating'] = (df['Average Rating'] - df['Average Rating'].min()) / (df['Average Rating'].max() - df['Average Rating'].min())

# Label encoding categories
label_encoder = LabelEncoder()
df['Category_Labels'] = label_encoder.fit_transform(df['Categories'])

# Tokenizer
max_vocab_size = 5000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Combined_Text'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df['Combined_Text'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, df['Category_Labels'], test_size=0.2, random_state=42
)

# Further split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)  # 10% of the training data for validation

# Convert to PyTorch tensors
X_train, X_val, X_test = (torch.tensor(X_train, dtype=torch.long), 
                          torch.tensor(X_val, dtype=torch.long), 
                          torch.tensor(X_test, dtype=torch.long))

y_train, y_val, y_test = (torch.tensor(y_train.values, dtype=torch.long), 
                          torch.tensor(y_val.values, dtype=torch.long), 
                          torch.tensor(y_test.values, dtype=torch.long))



# DataLoader
batch_size = 32
train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_data = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# Define RNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        x = self.dropout(torch.relu(self.fc1(h_n[-1])))
        x = self.fc2(x)
        return x

# Model parameters
embedding_dim = 128
hidden_size = 256
output_size = len(label_encoder.classes_)

# Instantiate and train model
model = RNNModel(max_vocab_size, embedding_dim, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_data:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    train_accuracy = correct / total
    
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_data:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)
    
    val_accuracy = val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_data):.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss/len(test_data):.4f}, Val Acc: {val_accuracy:.4f}")

# Save trained model
torch.save(model.state_dict(), 'rnn_model.pth')

# Function to get RNN embeddings
# Function to get RNN embeddings
def get_embedding(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)
    with torch.no_grad():
        _, h_n = model.rnn(model.embedding(input_tensor))
    return h_n[-1].cpu().numpy().reshape(-1)  # Ensure the embedding is 1D

# Precompute embeddings for all books
df['Embedding'] = df['Combined_Text'].apply(lambda x: get_embedding(x))

def recommend_books(input_title, top_n=5):
    if input_title in df['Title'].values:
        input_embedding = df.loc[df['Title'] == input_title, 'Embedding'].values[0].reshape(1, -1)
    else:
        input_embedding = np.mean(np.stack(df['Embedding'].values), axis=0).reshape(1, -1)
    
    embeddings = np.vstack(df['Embedding'].values)  # Stack to ensure 2D array
    book_titles = df['Title'].values
    similarities = cosine_similarity(input_embedding, embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    return book_titles[similar_indices]

# Example usage
print(recommend_books("Harry Potter and the Sorcerer's Stone"))

