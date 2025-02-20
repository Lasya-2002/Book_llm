import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
def load_dataset():
    df = pd.read_csv("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/google_books_dataset.csv")
    df.drop(['Published Date', 'Page Count'], axis=1, inplace=True)
    df['Title'] = df['Title'].str.lower()
    df['Description'] = df['Description'].str.lower()
    df['Categories'] = df['Categories'].str.lower()

    title_vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    desc_vectorizer = CountVectorizer(stop_words='english', max_features=3000)

    title_x = title_vectorizer.fit_transform(df['Title'])
    desc_x = desc_vectorizer.fit_transform(df['Description'])
    return df, title_vectorizer, desc_vectorizer, title_x, desc_x

def recommend_books(book_title, n=5, filter_category=True, filter_author=False, weight_ratings=True, title_weight=0.7, desc_weight=0.3, similarity_threshold=0.1):

    df,title_vectorizer,desc_vectorizer,title_x,desc_x=load_dataset()
    user_title_cleaned = book_title.lower()
    user_title_vector = title_vectorizer.transform([user_title_cleaned])
    user_desc_vector = desc_vectorizer.transform([""])  # Placeholder empty desc
    
    title_sim_scores = cosine_similarity(user_title_vector, title_x)[0]
    desc_sim_scores = cosine_similarity(user_desc_vector, desc_x)[0]
    
    combined_sim = (title_weight * title_sim_scores) + (desc_weight * desc_sim_scores)
    most_similar_idx = combined_sim.argmax()
    
    if combined_sim[most_similar_idx] < similarity_threshold:
        return [("No similar books found!", "N/A", 0, 0.0)]  # Ensure consistent return format
    
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
    
    if weight_ratings:
        recommendations.sort(
            key=lambda book: (
                book[2] if pd.notna(book[2]) else 0,  # Rating
                df[df['Title'] == book[0]]['Ratings Count'].values[0] if len(df[df['Title'] == book[0]]['Ratings Count'].values) > 0 else 0
            ),
            reverse=True
        )
    
    return recommendations[:n]

