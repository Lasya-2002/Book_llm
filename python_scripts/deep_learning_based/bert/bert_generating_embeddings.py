import pandas as pd
import torch
import numpy as np
import sqlite3
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset and fill missing values
df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")
df = df.fillna("")

# Define tokenizer and pre-trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased")
base_model.eval()  # set model to evaluation mode

# Setup device and move the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

# Preprocess text columns: lowercase and ensure string type
df['Title'] = df['Title'].astype(str).str.lower()
df['Description'] = df['Description'].astype(str).str.lower()
df['Categories'] = df['Categories'].astype(str).str.lower()

# Tokenize text columns separately with tailored max lengths
def tokenize_features(df):
    title_tokens = tokenizer(
        df['Title'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    desc_tokens = tokenizer(
        df['Description'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    cat_tokens = tokenizer(
        df['Categories'].tolist(), padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    return title_tokens, desc_tokens, cat_tokens

title_tokens, desc_tokens, cat_tokens = tokenize_features(df)

# Scale numeric features and encode language
def feature_scaling(df):
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    scaled_page_count = scaler.fit_transform(df[["Page Count"]])
    scaled_average_rating = scaler.fit_transform(df[["Average Rating"]])
    scaled_ratings_count = scaler.fit_transform(df[["Ratings Count"]])
    encoded_language = label_encoder.fit_transform(df['Language'])
    
    return scaled_page_count, scaled_average_rating, scaled_ratings_count, encoded_language

pg_count, avg_rating, rat_count, lan = feature_scaling(df)

# Prepare numeric feature tensor (each row: Page Count, Average Rating, Ratings Count, Language)
numeric_features = np.column_stack((pg_count, avg_rating, rat_count, lan))
numeric_features = torch.tensor(numeric_features, dtype=torch.float32)

# Create SQLite connection
conn = sqlite3.connect("book_embeddings.db")
c = conn.cursor()

# Create table to store embeddings separately
c.execute('''
    CREATE TABLE IF NOT EXISTS book_embeddings (
        id INTEGER PRIMARY KEY,
        title TEXT,
        description TEXT,
        categories TEXT,
        authors TEXT,
        title_embedding BLOB,
        description_embedding BLOB,
        category_embedding BLOB,
        numeric_features BLOB
    )
''')

# Generate and store embeddings separately
batch_size = 32
for i in tqdm(range(0, len(df), batch_size)):
    # Move tokenized inputs to device
    batch_title_tokens = {k: v[i:i+batch_size].to(device) for k, v in title_tokens.items()}
    batch_desc_tokens = {k: v[i:i+batch_size].to(device) for k, v in desc_tokens.items()}
    batch_cat_tokens = {k: v[i:i+batch_size].to(device) for k, v in cat_tokens.items()}
    batch_numeric_features = numeric_features[i:i+batch_size].to(device)
    
    with torch.no_grad():
        # Extract embeddings for each text field using the base BERT model
        title_embeds = base_model(**batch_title_tokens).pooler_output.cpu().numpy()
        desc_embeds = base_model(**batch_desc_tokens).pooler_output.cpu().numpy()
        cat_embeds = base_model(**batch_cat_tokens).pooler_output.cpu().numpy()
        batch_numeric_features = batch_numeric_features.cpu().numpy()

    # Store embeddings in database
    for j in range(len(title_embeds)):
        c.execute('''
            INSERT INTO book_embeddings 
            (id, title,description, categories, authors, title_embedding, description_embedding, category_embedding, numeric_features)
            VALUES (?, ?, ?, ?, ?, ?,?,?,?)
        ''', (
            i + j, 
            df['Title'].iloc[i + j],
            df['Description'].iloc[i + j],
            df['Categories'].iloc[i + j],
            df['Authors'].iloc[i + j],
            title_embeds[j].tobytes(),
            desc_embeds[j].tobytes(),
            cat_embeds[j].tobytes(),
            batch_numeric_features[j].tobytes()
        ))

# Commit and close connection
conn.commit()
conn.close()

print("Embeddings successfully generated and stored in SQLite!")
