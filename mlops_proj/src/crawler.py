import os
import json
import time
import requests

class TMDB_crawler:
    def __init__(
            self, 
            region = "KR", 
            language = "ko_KR", 
            image_language = "ko", 
            request_interval_seconds = 0.4
        ):
        self._base_url = os.environ.get("TMDB_BASE_URL")
        self._api_key = os.environ.get("TMDB_API_KEY")
        self._region = region
        self._language = language
        self._image_language = image_language
        self._request_interval_seconds = request_interval_seconds

    def get_popular_movies(self, page: int) -> list: # get popular movies from a page
        params = {
                "api_key": self._api_key, 
                "language": self._language, 
                "region": self._region, 
                "page": page, 
        }

        url = os.path.join(self._base_url, "popular")
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Request fails (status code: {response.status_code})")
            return []

        try:
            data = response.json()
            return data.get("results", [])

        except json.JSONDecodeError:
            print("JSON parsing fails")
            return []

    def get_popular_movie_pages(self, start_page: int, end_page: int) -> list: # get popular movie pages
        movies = []
        for page in range(start_page, end_page + 1):
            print(f"Getting page #{str(page).zfill(3)}...")
            movies.extend(self.get_popular_movies(page))
            time.sleep(self._request_interval_seconds) # interval to avoid crawler detector
        
        return movies

    @staticmethod
    def save_movies_to_json_file(movies: list, dst = "./result", filename = "popular") -> None:
        os.makedirs(dst, exist_ok=True)
        filepath = os.path.join(dst, f"{filename}.json")
        
        with open(filepath, 'w', encoding = "utf-8") as f:
            json.dump({"movies": movies}, f, ensure_ascii = False, indent = 2)

        print(f"File saved: {filepath}")
