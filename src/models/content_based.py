"""
Content-Based Filtering Model
Recommends movies based on item features (genres)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    """
    Content-Based Filtering using movie genre features.
    
    This approach recommends movies similar to those a user has liked
    based on movie attributes (genres in this case).
    
    Key Concepts: 
    - Feature Vectors: Represent each movie as a vector of features
    - Cosine Similarity:  Measure similarity between movie vectors
    - User Profile: Aggregate features of movies the user has liked
    """
    
    def __init__(self):
        """Initialize the content-based model."""
        self.genre_matrix = None
        self.similarity_matrix = None
        self.movies = None
        self.user_profiles = {}
        self.is_trained = False
        
    def train(self, movies_df, ratings_df, rating_threshold=3.5):
        """
        Train the content-based model.
        
        Args: 
            movies_df (pd.DataFrame): Movies with genre information
            ratings_df (pd.DataFrame): User ratings
            rating_threshold (float): Minimum rating to consider as "liked"
        """
        print("\nTraining Content-Based Model...")
        
        self.movies = movies_df.copy()
        self.ratings = ratings_df.copy()
        self.rating_threshold = rating_threshold
        
        # Extract genre columns
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Create genre matrix
        self.genre_matrix = movies_df.set_index('movie_id')[genre_columns].copy()
        
        # Calculate movie-movie similarity
        self.similarity_matrix = cosine_similarity(self.genre_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )
        
        # Build user profiles
        self._build_user_profiles()
        
        self.is_trained = True
        print("Training complete!")
        
    def _build_user_profiles(self):
        """Build user preference profiles based on their rating history."""
        for user_id in self.ratings['user_id'].unique():
            # Get movies this user liked (rated above threshold)
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            liked_movies = user_ratings[
                user_ratings['rating'] >= self.rating_threshold
            ]['movie_id'].values
            
            if len(liked_movies) > 0:
                # User profile = average of liked movie genre vectors
                liked_genres = self.genre_matrix.loc[
                    self.genre_matrix.index.isin(liked_movies)
                ]
                self.user_profiles[user_id] = liked_genres.mean().values
    
    def get_similar_movies(self, movie_id, n=10):
        """
        Find movies similar to a given movie.
        
        Args:
            movie_id (int): Movie ID
            n (int): Number of similar movies to return
            
        Returns:
            pd.DataFrame: Similar movies with similarity scores
        """
        if not self.is_trained: 
            raise ValueError("Model must be trained first")
            
        if movie_id not in self.similarity_df.index:
            raise ValueError(f"Movie {movie_id} not found in the dataset")
        
        # Get similarity scores for this movie
        similarities = self.similarity_df[movie_id].sort_values(ascending=False)
        
        # Exclude the movie itself and get top N
        similar_movies = similarities.drop(movie_id).head(n)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'movie_id': similar_movies.index,
            'similarity_score':  similar_movies.values
        })
        
        result = result.merge(self.movies[['movie_id', 'title']], on='movie_id')
        
        return result[['movie_id', 'title', 'similarity_score']]
    
    def get_top_n_recommendations(self, user_id, n=10, exclude_rated=True):
        """
        Get top N movie recommendations for a user.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            exclude_rated (bool): Whether to exclude already rated movies
            
        Returns: 
            pd.DataFrame: Top N recommended movies with scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if user_id not in self.user_profiles:
            print(f"User {user_id} has no profile. Using popularity-based fallback.")
            return self._get_popular_movies(n)
        
        # Get user profile
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity between user profile and all movies
        movie_scores = {}
        for movie_id in self.genre_matrix.index:
            movie_vector = self.genre_matrix.loc[movie_id].values
            # Cosine similarity
            dot_product = np.dot(user_profile, movie_vector)
            norm_product = np.linalg.norm(user_profile) * np.linalg.norm(movie_vector)
            if norm_product > 0:
                similarity = dot_product / norm_product
            else:
                similarity = 0
            movie_scores[movie_id] = similarity
        
        # Exclude rated movies if requested
        if exclude_rated:
            rated_movies = set(
                self.ratings[self.ratings['user_id'] == user_id]['movie_id']
            )
            movie_scores = {
                k: v for k, v in movie_scores.items() 
                if k not in rated_movies
            }
        
        # Sort and get top N
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_movies[:n]
        
        # Create result DataFrame
        result = pd.DataFrame(top_n, columns=['movie_id', 'content_score'])
        result = result.merge(self.movies[['movie_id', 'title']], on='movie_id')
        
        return result[['movie_id', 'title', 'content_score']]
    
    def _get_popular_movies(self, n):
        """Fallback:  Return most popular movies."""
        popular = self.ratings.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        popular.columns = ['movie_id', 'rating_count', 'avg_rating']
        popular = popular.sort_values(
            by=['rating_count', 'avg_rating'],
            ascending=False
        ).head(n)
        
        popular = popular.merge(self.movies[['movie_id', 'title']], on='movie_id')
        popular['content_score'] = popular['avg_rating'] / 5.0
        
        return popular[['movie_id', 'title', 'content_score']]
    
    def explain_recommendation(self, user_id, movie_id):
        """
        Explain why a movie was recommended to a user.
        
        Args: 
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns: 
            dict:  Explanation with matching genres
        """
        if user_id not in self.user_profiles:
            return {"error": "User profile not found"}
        
        user_profile = self.user_profiles[user_id]
        movie_genres = self.genre_matrix.loc[movie_id]
        
        genre_names = self.genre_matrix.columns
        
        # Find matching genres
        matching_genres = []
        for i, (genre, movie_has, user_pref) in enumerate(
            zip(genre_names, movie_genres.values, user_profile)
        ):
            if movie_has > 0 and user_pref > 0.3:  # User shows preference
                matching_genres.append((genre, user_pref))
        
        matching_genres.sort(key=lambda x:  x[1], reverse=True)
        
        return {
            "movie_id": movie_id,
            "movie_title": self.movies[self.movies['movie_id'] == movie_id]['title'].values[0],
            "matching_genres": [g[0] for g in matching_genres[: 5]],
            "explanation": f"Recommended because you like {', '.join([g[0] for g in matching_genres[:3]])} movies"
        }