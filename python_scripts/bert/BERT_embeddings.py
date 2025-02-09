import streamlit as st
import torch
import faiss
import numpy as np
import pandas as pd
import sqlite3
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Load the fine-tuned BERT model
MODEL_NAME = 'sri-lasya/book-recommender-bert'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to("cuda")  # Move the model to GPU

# Load book dataset dynamically
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_api_dataset/google_books_api_dataset/Google_books_api_dataset.csv")
df = df.dropna()

# Generate batch embeddings
def get_batch_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move back to CPU for processing

# Create database connection
print('Creating sqlite database')
conn = sqlite3.connect("book_recommendations.db")
cursor = conn.cursor()

# Create table for storing embeddings
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author TEXT,
    description TEXT,
    average_rating REAL,
    ratings_count INTEGER,
    categories TEXT,
    book_embedding BLOB,
    author_embedding BLOB,
    title_embedding BLOB,
    category_embedding BLOB
)""")
conn.commit()

# Batch insert into database
BATCH_SIZE = 256

for i in tqdm(range(0, df.shape[0], BATCH_SIZE), desc="Processing books in batches"):
    batch = df.iloc[i:i + BATCH_SIZE]
    descriptions = batch["Description"].tolist()
    authors = batch["Authors"].tolist()
    titles = batch["Title"].tolist()
    categories = batch["Categories"].tolist()

    desc_embeddings = get_batch_embeddings(descriptions)
    author_embeddings = get_batch_embeddings(authors)
    title_embeddings = get_batch_embeddings(titles)
    category_embeddings = get_batch_embeddings(categories)

    buffer = []
    for j, row in batch.iterrows():
        buffer.append((
            row["Title"], 
            row["Authors"], 
            row["Description"], 
            row["Average Rating"], 
            row["Ratings Count"], 
            row["Categories"],
            desc_embeddings[j % BATCH_SIZE].tobytes(),
            author_embeddings[j % BATCH_SIZE].tobytes(),
            title_embeddings[j % BATCH_SIZE].tobytes(),
            category_embeddings[j % BATCH_SIZE].tobytes()
        ))
    
    # Insert the buffer in one transaction
    cursor.executemany("""
        INSERT INTO books (title, author, description, average_rating, ratings_count, categories, book_embedding, author_embedding, title_embedding, category_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", buffer)
    conn.commit()

print("Book, author, title, and category embeddings stored in database successfully.")
conn.close()
