import streamlit as st
import pandas as pd
from bag_of_words import recommend_books

# Streamlit UI
st.title("Book Recommendation System")
book_title = st.text_input("Enter Book Title", "")
n_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
filter_category = st.checkbox("Filter by Category", True)
filter_author = st.checkbox("Filter by Author", False)
weight_ratings = st.checkbox("Weight by Ratings", True)
title_weight = st.slider("Title Similarity Weight", 0.0, 1.0, 0.7)
desc_weight = st.slider("Description Similarity Weight", 0.0, 1.0, 0.3)
similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1)

if st.button("Get Recommendations"):
    recommendations = recommend_books(book_title, n_recommendations, filter_category, filter_author, weight_ratings, title_weight, desc_weight, similarity_threshold)

    if recommendations and recommendations[0][0] == "No similar books found!":
        st.warning("No similar books found!")
    else:
        st.write("### Recommended Books:")
        df_output = pd.DataFrame(recommendations, columns=['Title', 'Author', 'Average Rating', 'Similarity Score'])
        st.table(df_output)  # ✅ Corrected usage
