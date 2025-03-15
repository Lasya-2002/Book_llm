import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('punkt_tab')

# Load dataset
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")  # Ensure books.csv has 'Title' and 'Description' columns

# Convert to lowercase and remove punctuation
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    return tokens

class TQDMProgress(tqdm):
    """Custom tqdm progress bar for gensim Word2Vec."""
    def update_epoch(self, epoch, loss):
        self.update(1)
        self.set_postfix(loss=loss)

# Preprocess text
df["Title"] = df["Title"].astype(str).apply(preprocess_text)
df["Description"] = df["Description"].astype(str).apply(preprocess_text)

# Wrap tqdm around tokenized sentences for progress tracking
title_sentences = list(tqdm(df["Title"].tolist(), desc="Processing Titles"))
desc_sentences = list(tqdm(df["Description"].tolist(), desc="Processing Descriptions"))

EPOCHS=20

#Train Word2Vec model for book titles
print("Training Word2Vec model for Titles...")
title_w2v = Word2Vec(vector_size=200, window=5, min_count=1, workers=4, sg=1)
title_w2v.build_vocab(df["Title"].tolist())

# Training with tqdm progress tracking
for epoch in tqdm(range(EPOCHS), desc="Title Model Training"):
    title_w2v.train(df["Title"].tolist(), total_examples=title_w2v.corpus_count, epochs=EPOCHS)
title_w2v.init_sims(replace=True)
print(title_w2v)

title_w2v.save("title_word2vec.model")

# Train Word2Vec model for descriptions
print("Training Word2Vec model for Descriptions...")
desc_w2v = Word2Vec(vector_size=200, window=10, min_count=1, workers=4, sg=1)
desc_w2v.build_vocab(df["Description"].tolist())

# Training with tqdm progress tracking
for epoch in tqdm(range(EPOCHS), desc="Description Model Training"):
    desc_w2v.train(df["Description"].tolist(), total_examples=desc_w2v.corpus_count, epochs=EPOCHS)
desc_w2v.init_sims(replace=True)
print(desc_w2v)

desc_w2v.save("desc_word2vec.model")

print("✅ Word2Vec models trained and saved successfully!")

