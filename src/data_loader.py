"""
Data Loader Module
Handles downloading and loading the MovieLens dataset
"""

import os
import zipfile
import requests
import pandas as pd
from io import BytesIO


class MovieLensDataLoader:
    """
    Loads the MovieLens 100K dataset for the recommendation system.

    The MovieLens dataset is a standard benchmark dataset in recommendation
    system research, containing user ratings for movies.
    """

    MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(self, data_dir="data"):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Directory to store the dataset
        """
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, "ml-100k")

    def download_dataset(self):
        """
        Download the MovieLens 100K dataset if not already present.
        """
        if os.path.exists(self.dataset_path):
            print("Dataset already exists.  Skipping download.")
            return

        print("Downloading MovieLens 100K dataset...")
        os.makedirs(self.data_dir, exist_ok=True)

        response = requests.get(self.MOVIELENS_URL)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(self.data_dir)

        print("Dataset downloaded successfully!")

    def load_ratings(self):
        """
        Load the ratings data.

        Returns:
            pd.DataFrame: DataFrame containing user_id, movie_id, rating, timestamp
        """
        ratings_path = os.path.join(self.dataset_path, "u.data")

        ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )

        print(f"Loaded {len(ratings)} ratings")
        return ratings

    def load_movies(self):
        """
        Load the movies data with genre information.

        Returns:
            pd.DataFrame: DataFrame containing movie information
        """
        movies_path = os.path.join(self.dataset_path, "u.item")

        genre_columns = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        columns = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
        ] + genre_columns

        movies = pd.read_csv(movies_path, sep="|", names=columns, encoding="latin-1")

        print(f"Loaded {len(movies)} movies")
        return movies

    def load_users(self):
        """
        Load user demographic data.

        Returns:
            pd.DataFrame: DataFrame containing user information
        """
        users_path = os.path.join(self.dataset_path, "u.user")

        users = pd.read_csv(
            users_path,
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            encoding="latin-1",
        )

        print(f"Loaded {len(users)} users")
        return users

    def load_all_data(self):
        """
        Load all datasets.

        Returns:
            tuple:  (ratings, movies, users) DataFrames
        """
        self.download_dataset()
        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()

        return ratings, movies, users
