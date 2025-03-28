import streamlit as st
import pandas as pd
from recommender import (
    load_embeddings,
    build_faiss_indexes,
    get_separate_scores_recommendations,
    calculate_mrr,
    calculate_map,
)

# Load embeddings and build/load FAISS indexes
@st.cache_resource
def initialize_recommender():
    book_titles, title_embeddings, desc_embeddings, cat_embeddings = load_embeddings()

    # Build or load FAISS indexes
    embeddings_dict = {
        "title": title_embeddings,
        "description": desc_embeddings,
        "categories": cat_embeddings
    }

    indexes_dict = build_faiss_indexes(embeddings_dict)
    return book_titles, embeddings_dict, indexes_dict


# Initialize on app load
book_titles, embeddings_dict, indexes_dict = initialize_recommender()

# Title and header
st.title("📚 Book Recommendation System with Separate Scores and Evaluation Metrics")
st.markdown("Enter a book title to get recommendations using separate scores for each embedding and display MRR and MAP for different FAISS indexes!")

# Input for book title
book_title = st.text_input("Enter book title:").strip().lower()

# Number of recommendations slider
num_recommendations = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

# Button to generate recommendations
if st.button("Get Recommendations"):
    if not book_title:
        st.error("❗ Please enter a valid book title.")
    else:
        st.write(f"🔍 Generating recommendations for **'{book_title}'** using FAISS indexes...")

        # Get recommendations with separate scores
        recommendations = get_separate_scores_recommendations(
            book_title, book_titles, embeddings_dict, indexes_dict, num_recommendations
        )

        if recommendations:
            for index_type, df in recommendations.items():
                st.subheader(f"📚 Recommendations using **{index_type}** FAISS index:")
                if not df.empty:
                    st.dataframe(df, width=800)

                    # Calculate and display MRR and MAP using functions from recommender
                    mrr_score = calculate_mrr(df, book_title)
                    map_score = calculate_map(df, book_title)

                    st.markdown(
                        f"✅ **MRR:** `{round(mrr_score, 4)}`  |  📊 **MAP:** `{round(map_score, 4)}`"
                    )
                else:
                    st.warning(f"No recommendations found using {index_type}.")
        else:
            st.error("❌ No recommendations found. Please try another book title.")
