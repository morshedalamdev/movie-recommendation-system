"""
Main Recommender Engine
Combines different recommendation approaches into a unified system
"""

import numpy as np
import pandas as pd
from .models import CollaborativeFilteringModel, ContentBasedModel


class HybridRecommender:
    """
    Hybrid Recommendation System combining multiple approaches.
    
    This class provides a unified interface for generating recommendations
    using both Collaborative Filtering and Content-Based methods.
    
    The hybrid approach can combine predictions from multiple models
    to provide more robust recommendations.
    """
    
    def __init__(self, cf_weight=0.7, cb_weight=0.3):
        """
        Initialize the hybrid recommender.
        
        Args:
            cf_weight (float): Weight for collaborative filtering predictions
            cb_weight (float): Weight for content-based predictions
        """
        self.cf_model = CollaborativeFilteringModel()
        self.cb_model = ContentBasedModel()
        
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        self.movies = None
        self.ratings = None
        self.is_trained = False
        
    def train(self, train_data, movies_df, ratings_df=None):
        """
        Train both recommendation models.
        
        Args:
            train_data (pd.DataFrame): Training ratings data
            movies_df (pd.DataFrame): Movie information with genres
            ratings_df (pd.DataFrame): Full ratings for content-based (optional)
        """
        print("\n" + "="*60)
        print("          TRAINING HYBRID RECOMMENDATION SYSTEM")
        print("="*60)
        
        self.movies = movies_df
        self.ratings = train_data
        
        # Train Collaborative Filtering
        self.cf_model.train(train_data)
        
        # Train Content-Based
        self.cb_model.train(
            movies_df, 
            ratings_df if ratings_df is not None else train_data
        )
        
        self.is_trained = True
        
        print("\n✅ Hybrid model training complete!")
        print(f"   CF Weight: {self.cf_weight}, CB Weight: {self.cb_weight}")
        print("="*60)
        
    def get_recommendations(self, user_id, n=10, method='hybrid'):
        """
        Get movie recommendations for a user.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            method (str): 'hybrid', 'collaborative', or 'content'
            
        Returns:
            pd.DataFrame: Recommended movies with scores
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making recommendations")
        
        if method == 'collaborative':
            return self.cf_model.get_top_n_recommendations(user_id, self.movies, n)
        elif method == 'content': 
            return self.cb_model.get_top_n_recommendations(user_id, n)
        else:
            return self._get_hybrid_recommendations(user_id, n)
    
    def _get_hybrid_recommendations(self, user_id, n):
        """
        Generate hybrid recommendations combining both models.
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Hybrid recommendations
        """
        # Get more candidates than needed
        n_candidates = min(n * 3, 100)
        
        # Get recommendations from both models
        cf_recs = self.cf_model.get_top_n_recommendations(
            user_id, self.movies, n_candidates
        )
        cb_recs = self.cb_model.get_top_n_recommendations(
            user_id, n_candidates
        )
        
        # Normalize scores
        cf_recs['cf_score_norm'] = (
            cf_recs['predicted_rating'] - cf_recs['predicted_rating'].min()
        ) / (cf_recs['predicted_rating'].max() - cf_recs['predicted_rating'].min() + 1e-6)
        
        cb_recs['cb_score_norm'] = (
            cb_recs['content_score'] - cb_recs['content_score'].min()
        ) / (cb_recs['content_score'].max() - cb_recs['content_score'].min() + 1e-6)
        
        # Merge recommendations
        merged = pd.merge(
            cf_recs[['movie_id', 'title', 'cf_score_norm']],
            cb_recs[['movie_id', 'cb_score_norm']],
            on='movie_id',
            how='outer'
        )
        
        # Fill missing scores with 0
        merged['cf_score_norm'] = merged['cf_score_norm'].fillna(0)
        merged['cb_score_norm'] = merged['cb_score_norm'].fillna(0)
        
        # Calculate hybrid score
        merged['hybrid_score'] = (
            self.cf_weight * merged['cf_score_norm'] + 
            self.cb_weight * merged['cb_score_norm']
        )
        
        # Fill missing titles
        merged = merged.merge(
            self.movies[['movie_id', 'title']],
            on='movie_id',
            how='left',
            suffixes=('', '_y')
        )
        merged['title'] = merged['title'].fillna(merged['title_y'])
        
        # Sort and return top N
        result = merged.nlargest(n, 'hybrid_score')
        
        return result[['movie_id', 'title', 'hybrid_score']]
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        return self.cf_model.predict(user_id, movie_id)
    
    def explain(self, user_id, movie_id):
        """
        Explain why a movie is recommended to a user.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            dict: Explanation
        """
        explanation = self.cb_model.explain_recommendation(user_id, movie_id)
        
        # Add predicted rating
        try:
            predicted_rating = self.cf_model.predict(user_id, movie_id)
            explanation['predicted_rating'] = round(predicted_rating, 2)
        except Exception:
            explanation['predicted_rating'] = None
            
        return explanation
    
    def get_similar_movies(self, movie_id, n=10):
        """
        Find movies similar to a given movie.
        
        Args:
            movie_id (int): Movie ID
            n (int): Number of similar movies
            
        Returns: 
            pd.DataFrame: Similar movies
        """
        return self.cb_model.get_similar_movies(movie_id, n)