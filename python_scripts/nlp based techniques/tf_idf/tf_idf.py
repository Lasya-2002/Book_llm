import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
def load_dataset():
    df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")
    df.drop(['Published Date', 'Page Count'], axis=1, inplace=True)
    df['Title'] = df['Title'].str.lower()
    df['Description'] = df['Description'].str.lower()
    df['Categories'] = df['Categories'].str.lower()

    title_vectorizer = TfidfVectorizer(stop_words='english', max_features=200000, ngram_range=(1,2))
    desc_vectorizer = TfidfVectorizer(stop_words='english', max_features=300000, ngram_range=(2,3))

    title_x = title_vectorizer.fit_transform(df['Title'])
    desc_x = desc_vectorizer.fit_transform(df['Description'])
    
    return df, title_vectorizer, desc_vectorizer, title_x, desc_x

# Category-based recommendation if the book is not found
def recommend_by_category(df, n=5):
    # Ask user for category input
    categories = df['Categories'].dropna().unique()
    print("📚 Available Categories: ", ", ".join(categories[:10]))  # Show top 10 categories
    user_category = input("Enter your preferred category: ").lower()
    
    # Filter books by selected category
    category_books = df[df['Categories'].str.contains(user_category, na=False)]
    
    if category_books.empty:
        print(f"⚠️ No books found in category '{user_category}'. Showing top-rated books instead.")
        return recommend_top_books(df, n)
    
    # Rank by highest-rated books
    category_books = category_books.sort_values(by=['Average Rating', 'Ratings Count'], ascending=False)
    recommendations = list(zip(category_books['Title'].values, 
                               category_books['Authors'].fillna("Unknown").values, 
                               category_books['Average Rating'].fillna(0).values))
    
    return recommendations[:n]

# Top-rated books fallback
def recommend_top_books(df, n=5):
    top_books = df.sort_values(by=['Average Rating', 'Ratings Count'], ascending=False)
    recommendations = list(zip(top_books['Title'].values, 
                               top_books['Authors'].fillna("Unknown").values, 
                               top_books['Average Rating'].fillna(0).values))
    
    return recommendations[:n]

# Compute recommendations
def recommend_books(book_title, n=5, filter_category=True, filter_author=False, weight_ratings=True, 
                     title_weight=0.7, desc_weight=0.3, similarity_threshold=0.1):
    
    df, title_vectorizer, desc_vectorizer, title_x, desc_x = load_dataset()
    
    user_title_cleaned = book_title.lower()
    
    # Check if book exists in the dataset
    if user_title_cleaned not in df['Title'].values:
        print(f"⚠️ Book '{book_title}' not found. Recommending books based on category...")
        return recommend_by_category(df, n)
    
    # If book exists, proceed with similarity calculation
    user_title_vector = title_vectorizer.transform([user_title_cleaned])
    user_desc_vector = desc_vectorizer.transform([""])  # Placeholder empty desc

    title_sim_scores = cosine_similarity(user_title_vector, title_x)[0]
    desc_sim_scores = cosine_similarity(user_desc_vector, desc_x)[0]
    
    combined_sim = (title_weight * title_sim_scores) + (desc_weight * desc_sim_scores)
    most_similar_idx = combined_sim.argmax()
    
    if combined_sim[most_similar_idx] < similarity_threshold:
        print("⚠️ No highly similar books found. Switching to category-based recommendations...")
        return recommend_by_category(df, n)
    
    similar_books = sorted(list(enumerate(combined_sim)), key=lambda x: x[1], reverse=True)
    recommendations = []
    
    for i, sim_score in similar_books:
        if filter_category and df.iloc[i]['Categories'] != df.iloc[most_similar_idx]['Categories']:
            continue
        if filter_author and df.iloc[i]['Authors'] != df.iloc[most_similar_idx]['Authors']:
            continue
        recommendations.append((
            df.iloc[i]['Title'],
            df.iloc[i]['Authors'] if pd.notna(df.iloc[i]['Authors']) else "Unknown",
            df.iloc[i]['Average Rating'] if pd.notna(df.iloc[i]['Average Rating']) else 0,
            sim_score
        ))
        if len(recommendations) >= n * 2:
            break
    
    # Apply weighted rating sorting
    if weight_ratings:
        recommendations.sort(
            key=lambda book: (
                book[2] if pd.notna(book[2]) else 0,  # Average Rating
                df[df['Title'] == book[0]]['Ratings Count'].values[0] if len(df[df['Title'] == book[0]]['Ratings Count'].values) > 0 else 0
            ),
            reverse=True
        )
    
    return recommendations[:n]
