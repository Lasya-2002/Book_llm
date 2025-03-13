import streamlit as st
import pandas as pd
from tf_idf import recommend_books, recommend_by_category, recommend_top_books

# Streamlit UI
st.set_page_config(page_title="Book Recommendation System", layout="wide")
st.title("📚 Book Recommendation System")

# User Input
book_title = st.text_input("🔍 Enter Book Title", "")
n_recommendations = st.slider("📌 Number of Recommendations", 1, 10, 5)

# Filtering Options
st.sidebar.header("⚙️ Advanced Settings")
filter_category = st.sidebar.checkbox("📂 Filter by Category", True)
filter_author = st.sidebar.checkbox("✍️ Filter by Author", False)
weight_ratings = st.sidebar.checkbox("⭐ Weight by Ratings", True)

# Weight Adjustments
title_weight = st.sidebar.slider("📝 Title Similarity Weight", 0.0, 1.0, 0.7)
desc_weight = st.sidebar.slider("📖 Description Similarity Weight", 0.0, 1.0, 0.3)
similarity_threshold = st.sidebar.slider("🔍 Similarity Threshold", 0.0, 1.0, 0.1)

# Get Recommendations
if st.button("🎯 Get Recommendations"):
    recommendations = recommend_books(book_title, n_recommendations, filter_category, filter_author, 
                                      weight_ratings, title_weight, desc_weight, similarity_threshold)
    
    # Handle Case: Book Not Found
    if not recommendations or recommendations[0][0] == "No similar books found!":
        st.warning("⚠️ No similar books found! Switching to category-based recommendations...")
        
        # Get Category Recommendations
        category_recommendations = recommend_by_category(n=n_recommendations)
        
        if not category_recommendations:
            st.error("⚠️ No books found in the selected category. Showing top-rated books instead.")
            category_recommendations = recommend_top_books(n=n_recommendations)
        
        st.write("### 🔥 Popular Books Based on Category:")
        st.table(pd.DataFrame(category_recommendations, columns=['Title', 'Author', 'Average Rating']))
    
    else:
        st.write("### 📚 Recommended Books:")
        recommendations_df = pd.DataFrame(recommendations, columns=['Title', 'Author', 'Average Rating', 'Similarity Score'])
        
        # Display Recommendations with Progress Bars
        for index, row in recommendations_df.iterrows():
            st.subheader(f"📖 {row['Title']}")
            st.write(f"✍️ Author: {row['Author']} | ⭐ Rating: {row['Average Rating']:.2f}")
            st.progress(row['Similarity Score'])  # Progress bar for similarity score
            st.markdown("---")  # Divider for better UI
