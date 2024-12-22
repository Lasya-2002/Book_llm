import requests
import pandas as pd
import time
from tqdm import tqdm

# Your Google Books API Key
API_KEY = "AIzaSyAsgBpVTVT3Nf8rV8s1RL9_zVXyUTJk2Wk"

# Function to fetch all books
def fetch_all_books(query, total_books, max_results=40):
    all_books = []
    start_index = 0
    iterations = (total_books + max_results - 1) // max_results  # Total requests needed

    with tqdm(total=iterations, desc=f"Fetching '{query}' data") as pbar:
        while start_index < total_books:
            url = f"https://www.googleapis.com/books/v1/volumes"
            params = {
                "q": query,
                "startIndex": start_index,
                "maxResults": max_results,
                "printType": "books",  # Focus only on books
                "orderBy": "relevance",  # Fetch relevant results
                "key": API_KEY
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if "items" in data:
                    all_books.extend(data["items"])
                    start_index += max_results
                    pbar.update(1)
                else:
                    print("No more results found for this query.")
                    break
            else:
                print(f"Error {response.status_code}: {response.reason}")
                break
            time.sleep(1)
    return all_books

# Function to parse books
def parse_books(items):
    books = []
    for item in items:
        book_info = item.get("volumeInfo", {})
        books.append({
            "Title": book_info.get("title", "N/A"),
            "Authors": ", ".join(book_info.get("authors", [])),
            "Description": book_info.get("description", "N/A"),
            "Categories": ", ".join(book_info.get("categories", [])),
            "Page Count": book_info.get("pageCount", 0),
            "Average Rating": book_info.get("averageRating", "N/A"),
            "Ratings Count": book_info.get("ratingsCount", 0),
            "Language": book_info.get("language", "N/A")
        })
    return books

# Save to CSV
def save_to_csv(data, query, filename="books_dataset.csv"):
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)  # Remove duplicates
    filename = f"{query}_books_dataset.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main script
if __name__ == "__main__":
    # Search terms for broader coverage
    queries = [
        "Fiction", "Novels", "Classic Fiction", "Modern Fiction",
        "Fiction books", "Literature Fiction"
    ]
    total_books_per_query = 5000  # Adjust based on observations
    all_data = []

    for query in queries:
        print(f"Fetching data for query: {query}")
        items = fetch_all_books(query, total_books=total_books_per_query, max_results=40)
        books = parse_books(items)
        all_data.extend(books)

    save_to_csv(all_data, "Fiction")
