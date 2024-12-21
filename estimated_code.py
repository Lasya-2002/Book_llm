import requests

API_KEY = "AIzaSyAsgBpVTVT3Nf8rV8s1RL9_zVXyUTJk2Wk"
url = "https://www.googleapis.com/books/v1/volumes"

queries=['Graphic Novels','Family & Relationships','Foreign Language Study','Games and Activities','Games','Activities','Gardening','health and fitness','history','health','fitness']

for query in queries:
    params = {
        "q": query,  # Replace with your keyword
        "startIndex": 0,
        "maxResults": 1,  # Minimum request to get total count
        "key": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        total_items = data.get("totalItems", "Not available")
        print(f"Total books matching {query} : {total_items}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
