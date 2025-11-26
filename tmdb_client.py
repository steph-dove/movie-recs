import os
import time
import requests
from dotenv import load_dotenv


load_dotenv()

TMDB_API_KEY=os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

HEADERS = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "Accept": "application/json",
}

def discover_movies(page: int = 1):
    resp = requests.get(
        f"{BASE_URL}/discover/movie",
        headers=HEADERS,
        params={
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": page,
            "include_adult": "false",
        },
        timeout=10
    )
    
    resp.raise_for_status()
    return resp.json()

def get_movie_details(movie_id: int):
    resp = requests.get(
        f"{BASE_URL}/movie/{movie_id}",
        headers=HEADERS,
        params={
            "language": "en-US",
            "append_to_response": "keywords",
        },
        timeout=10, 
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()

def fetch_movies(num_pages: int = 5, sleep_sec: float = 0.25):
    movies = []
    for page in range(1, num_pages + 1):
        page_data = discover_movies(page)
        for m in page_data.get("results", []):
            details = get_movie_details(m["id"])
            if details is None:
                continue
            movies.append(details)
            time.sleep(sleep_sec)
    return movies
