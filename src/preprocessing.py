"""
Data Preprocessing Module
Handles data cleaning, transformation, and preparation for model training
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Preprocesses the MovieLens data for recommendation models.
    
    This class handles: 
    - Data cleaning and validation
    - Train/test splitting
    - Feature engineering for content-based filtering
    - Creating user-item matrices
    """
    
    def __init__(self, ratings, movies, users):
        """
        Initialize the preprocessor with raw data.
        
        Args:
            ratings (pd.DataFrame): User ratings data
            movies (pd.DataFrame): Movie information
            users (pd.DataFrame): User demographic data
        """
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.users = users.copy()
        
    def clean_data(self):
        """
        Clean and validate the data.
        
        - Remove duplicates
        - Handle missing values
        - Validate rating ranges
        """
        print("Cleaning data...")
        
        # Remove duplicate ratings (keep the latest one)
        self.ratings = self.ratings.drop_duplicates(
            subset=['user_id', 'movie_id'],
            keep='last'
        )
        
        # Ensure ratings are in valid range (1-5)
        self.ratings = self.ratings[
            (self.ratings['rating'] >= 1) & 
            (self.ratings['rating'] <= 5)
        ]
        
        # Remove movies with missing titles
        self.movies = self.movies[self.movies['title'].notna()]
        
        print(f"After cleaning: {len(self.ratings)} ratings, {len(self.movies)} movies")
        
        return self
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the ratings data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns: 
            tuple: (train_data, test_data) DataFrames
        """
        print(f"Splitting data:  {1-test_size:.0%} train, {test_size:.0%} test")
        
        train_data, test_data = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # Could stratify by user or rating
        )
        
        print(f"Training set:  {len(train_data)} ratings")
        print(f"Testing set:  {len(test_data)} ratings")
        
        return train_data, test_data
    
    def create_user_item_matrix(self, data=None):
        """
        Create a user-item rating matrix (sparse representation).
        
        Args:
            data (pd.DataFrame): Ratings data (uses self.ratings if None)
            
        Returns:
            pd.DataFrame: User-item matrix with ratings
        """
        if data is None: 
            data = self.ratings
            
        user_item_matrix = data.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        print(f"User-Item matrix shape:  {user_item_matrix.shape}")
        return user_item_matrix
    
    def get_genre_features(self):
        """
        Extract genre features for content-based filtering.
        
        Returns: 
            pd.DataFrame: Movie-genre binary matrix
        """
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        genre_matrix = self.movies[['movie_id'] + genre_columns].copy()
        genre_matrix = genre_matrix.set_index('movie_id')
        
        return genre_matrix
    
    def get_statistics(self):
        """
        Calculate and display dataset statistics.
        
        Returns:
            dict: Dictionary containing various statistics
        """
        stats = {
            'n_users': self.ratings['user_id'].nunique(),
            'n_movies':  self.ratings['movie_id'].nunique(),
            'n_ratings':  len(self.ratings),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std(),
            'sparsity': 1 - (len(self.ratings) / 
                           (self.ratings['user_id'].nunique() * 
                            self.ratings['movie_id'].nunique())),
            'ratings_per_user':  self.ratings.groupby('user_id').size().mean(),
            'ratings_per_movie': self.ratings.groupby('movie_id').size().mean()
        }
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Number of users: {stats['n_users']}")
        print(f"Number of movies:  {stats['n_movies']}")
        print(f"Number of ratings: {stats['n_ratings']}")
        print(f"Average rating:  {stats['avg_rating']:.2f}")
        print(f"Rating std: {stats['rating_std']:.2f}")
        print(f"Matrix sparsity: {stats['sparsity']:.2%}")
        print(f"Avg ratings per user: {stats['ratings_per_user']:.1f}")
        print(f"Avg ratings per movie: {stats['ratings_per_movie']:.1f}")
        print("="*50 + "\n")
        
        return stats