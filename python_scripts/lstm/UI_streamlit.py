import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/google_books_api_dataset/Google_books_api_dataset.csv')

# Preprocessing
df = df.dropna(subset=['Title', 'Authors', 'Language', 'Categories', 'Description'])
df['Combined_Text'] = df['Title'] + " " + df['Authors'] + " " + df['Language'] + " " + df['Description']

# Load label encoder
label_encoder = LabelEncoder()
df['Category_Labels'] = label_encoder.fit_transform(df['Categories'])

# Tokenizer
max_vocab_size = 5000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Combined_Text'])

# Load trained LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, 256)
        self.fc2 = torch.nn.Linear(256, output_size)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(torch.relu(self.fc1(h_n[-1])))
        x = self.fc2(x)
        return x

# Load trained model
vocab_size = max_vocab_size
embedding_dim = 128
hidden_size = 256
output_size = len(df['Category_Labels'].unique())

model = LSTMModel(vocab_size, embedding_dim, hidden_size, output_size).to(device)
model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
model.eval()

# Function to get LSTM embeddings
def get_embedding(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)
    with torch.no_grad():
        _, (h_n, _) = model.lstm(model.embedding(input_tensor))
    return h_n[-1].cpu().numpy()

# Precompute embeddings for all books
df['Embedding'] = df['Combined_Text'].apply(lambda x: get_embedding(x))

# Streamlit App
def main():
    st.title("Book Recommendation System with LSTM Embeddings")

    user_input = st.text_input("Enter a Book Title")
    user_category = st.selectbox("Select a Category", options=df['Categories'].unique())

    if user_input and user_category:
        # Get embedding for user input
        user_embedding = get_embedding(user_input)

        # Filter dataset by category
        filtered_df = df[df['Categories'] == user_category]

        # Compute similarity between user input and book embeddings
        similarities = filtered_df['Embedding'].apply(lambda x: np.dot(user_embedding, x.T).item())
        top_indices = similarities.nlargest(5).index

        # Display recommendations
        st.write("### Recommended Books")
        st.write(filtered_df.loc[top_indices, ['Title', 'Authors', 'Categories', 'Description']])

if __name__ == "__main__":
    main()
