import requests
import pandas as pd
import time  # To add delays between requests if needed
from tqdm import tqdm  # Progress bar

# Your Google Books API Key
API_KEY = "AIzaSyAsgBpVTVT3Nf8rV8s1RL9_zVXyUTJk2Wk"

# Function to fetch books with pagination
def fetch_books(query, max_results=40, total_books=2000):
    all_books = []
    start_index = 0
    iterations = total_books // max_results

    # Use tqdm for progress monitoring
    with tqdm(total=iterations, desc=f"Fetching '{query}' data") as pbar:
        while start_index < total_books:
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&startIndex={start_index}&maxResults={max_results}&key={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "items" in data:
                    all_books.extend(data["items"])
                    start_index += max_results
                    pbar.update(1)  # Update progress bar
                else:
                    print("No more results for this query.")
                    break
            else:
                print(f"Error {response.status_code}: {response.reason}")
                break
            time.sleep(1)  # Avoid hitting API limits
    return all_books

# Parse the book data
def parse_books(items):
    books = []
    for item in items:
        book_info = item.get("volumeInfo", {})
        books.append({
            "Title": book_info.get("title", "N/A"),
            "Authors": ", ".join(book_info.get("authors", [])),
            "Publisher": book_info.get("publisher", "N/A"),
            "Published Date": book_info.get("publishedDate", "N/A"),
            "Description": book_info.get("description", "N/A"),
            "Categories": ", ".join(book_info.get("categories", [])),
            "Page Count": book_info.get("pageCount", 0),
            "Average Rating": book_info.get("averageRating", "N/A"),
            "Ratings Count": book_info.get("ratingsCount", 0),
            "Language": book_info.get("language", "N/A")
        })
    return books

# Save data to CSV
def save_to_csv(data, filename="books_dataset.csv"):
    df = pd.DataFrame(data)
    df.drop_duplicates()
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main script
if __name__ == "__main__":
    search_queries = ["fiction", "science fiction", "history", "fantasy", "romance","Memoir","Narrative","Self help","Mystery","Non- Fiction","Young Adult Literature","Historical Fiction","Thriller","Graphic","Women Fiction","Horror Fiction","Biography","Historical Fantasy","Essay","Contemporary Romance","Travel","Poetry","Science","True Crime","Humour","Satire","Social Science","Adventure Fiction","New Adult Fiction","Speculative Fiction","Spirituality","Magical Realism","Fairy Tale","Philosophy","Alternate History"]
    search_queries_1=["Drama","Detective Fiction","Action","Paranormal Fantasy","Science Fantasy","Technology","Education"]
    all_data = []

    for query in search_queries:
        print(f"Fetching data for query: {query}")
        items = fetch_books(query, max_results=40, total_books=2000)  # Fetch up to 2000 books per query
        books = parse_books(items)
        all_data.extend(books)

    for query in search_queries_1:
        print(f"Fetching data for query: {query}")
        items=fetch_books(query,max_results=40,total_books=2000)
        books=parse_books(items)
        all_data.extend(books)

    save_to_csv(all_data)
