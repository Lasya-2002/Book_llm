import streamlit as st
import pandas as pd
from recommender import recommend_books, load_trained_model

# Load model and data
df, _, _, _, _, _ = load_trained_model()

st.title("📚 Book Recommendation System")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
n_recommendations = st.sidebar.slider("🔢 Number of Recommendations", 1, 10, 5)

# Filters
st.sidebar.subheader("🔍 Filters")
selected_category = st.sidebar.selectbox("📚 Filter by Category", ["Any"] + df['Categories'].dropna().unique().tolist())
selected_author = st.sidebar.selectbox("✍️ Filter by Author", ["Any"] + df['Authors'].dropna().unique().tolist())
min_rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0)

# Main Input
book_title = st.text_input("🔎 Enter a Book Title", "")

if st.button("📌 Get Recommendations"):
    if book_title.strip() == "":
        st.warning("⚠️ Please enter a book title!")
    else:
        recommendations = recommend_books(book_title, n=n_recommendations, filter_category=selected_category, filter_author=selected_author, min_rating=min_rating)

        if recommendations[0][0] == "No similar books found!":
            st.warning("🚫 No similar books found!")
        elif recommendations[0][0] == "No books found after filtering!":
            st.warning("🚫 No books matched the selected filters!")
        else:
            st.write("### 📖 Recommended Books:")
            df_output = pd.DataFrame(recommendations, columns=['Title', 'Author', 'Average Rating', 'Similarity Score'])
            st.table(df_output)
