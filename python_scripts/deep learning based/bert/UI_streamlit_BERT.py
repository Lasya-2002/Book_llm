import streamlit as st
import warnings as wr
from BERT_recommendations import *
wr.filterwarnings('ignore')

# Function to create a book recommendation card
def create_card(title, author, description, categories, average_rating, ratings_count):
    return f"""
    <div style="
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        color: black;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;">
        <h4 style="margin: 0;color: black; ">📖 <b>{title}</b></h4>
        <p style="margin: 5px 0;"><b>Author:</b> {author}</p>
        <p style="margin: 5px 0;"><b>Categories:</b> {categories}</p>
        <p style="margin: 5px 0;"><b>Description:</b> {description[:200]}...</p>
        <p style="margin: 5px 0;">
            <b>⭐ Rating:</b> {average_rating} | <b>Reviews:</b> {ratings_count}
        </p>
    </div>
"""

# Streamlit UI setup
st.set_page_config(page_title="Book Recommendation System", page_icon="📚")
st.title("📚 Book Recommendation System")

book_title = st.text_input("Enter book title (required):")
book_category = st.text_input("Enter book category (required):")

if st.button("Get Recommendations"):
    if not book_category:
        st.warning("Please enter a book category.")
    else:
        # Fetch embeddings for the input book (if title is provided)
        if book_title:
            combined_embeddings = get_input_embeddings_with_category(book_title, book_category)
            
            if combined_embeddings is None:
                st.write("Could not generate embeddings. Please check the input.")
            else:
                recommendations = get_recommendations_by_embeddings(combined_embeddings)
                if recommendations:
                    st.subheader("📌 Recommended Books (based on title and category):")
                    for title, author, description, categories, average_rating, ratings_count in recommendations:
                        st.markdown(create_card(title, author, description, categories, average_rating, ratings_count), unsafe_allow_html=True)
                else:
                    st.write("No recommendations found.")
        else:
            st.warning("Please enter a book title.")