import streamlit as st
from recommender import recommend_books, load_models, load_data
import pandas as pd

st.set_page_config(page_title="Book Recommendation System", layout="wide")
st.title("📚 Book Recommendation System Word2Vec")

title_model, desc_model = load_models()
df = load_data()

# User input
book_title = st.text_input("Enter Book Title:")
filter_category = st.checkbox("Filter by Category")
filter_author = st.checkbox("Filter by Author")
filter_rating = st.checkbox("Filter by Rating")
topn = st.number_input("Enter the number of recommendations", min_value=1, step=1, value=5)

if st.button("Get Recommendations"):
    recommendations = recommend_books(
        title_model, desc_model, book_title, df,
        top_n=topn, category_filter=filter_category,
        author_filter=filter_author, ratings_filter=filter_rating
    )
    
    if recommendations.empty:
        st.write("No recommendations found.")
    else:
        st.write("### Recommended Books:")
        st.dataframe(recommendations)
