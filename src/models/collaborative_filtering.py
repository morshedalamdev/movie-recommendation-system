"""
Collaborative Filtering Model
Implements SVD-based matrix factorization for recommendations
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate


class CollaborativeFilteringModel: 
    """
    Collaborative Filtering using Singular Value Decomposition (SVD).
    
    This approach learns latent factors for users and items from the
    rating matrix, then uses these factors to predict ratings for
    unseen user-item pairs.
    
    Key Concepts:
    - Matrix Factorization:  Decomposes the user-item matrix into lower-dimensional matrices
    - Latent Factors:  Hidden features that explain user preferences and item characteristics
    - SVD: Singular Value Decomposition algorithm for matrix factorization
    """
    
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Initialize the SVD model.
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of training iterations
            lr_all (float): Learning rate for all parameters
            reg_all (float): Regularization term for all parameters
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        
        self.trainset = None
        self.is_trained = False
        
    def prepare_data(self, ratings_df):
        """
        Prepare the data for the Surprise library.
        
        Args: 
            ratings_df (pd.DataFrame): DataFrame with user_id, movie_id, rating
            
        Returns: 
            Dataset: Surprise Dataset object
        """
        reader = Reader(rating_scale=(1, 5))
        
        data = Dataset.load_from_df(
            ratings_df[['user_id', 'movie_id', 'rating']],
            reader
        )
        
        return data
    
    def train(self, train_data):
        """
        Train the SVD model on the training data.
        
        Args:
            train_data (pd.DataFrame): Training ratings data
        """
        print("\nTraining Collaborative Filtering Model (SVD)...")
        print(f"Parameters: factors={self.n_factors}, epochs={self.n_epochs}")
        
        data = self.prepare_data(train_data)
        self.trainset = data.build_full_trainset()
        
        self.model.fit(self.trainset)
        self.is_trained = True
        
        print("Training complete!")
        
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a user-movie pair.
        
        Args: 
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns: 
            float:  Predicted rating
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
    
    def predict_batch(self, test_data):
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            test_data (pd.DataFrame): DataFrame with user_id, movie_id
            
        Returns: 
            np.array: Array of predicted ratings
        """
        predictions = []
        for _, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['movie_id'])
            predictions.append(pred)
            
        return np.array(predictions)
    
    def get_top_n_recommendations(self, user_id, movies_df, n=10, exclude_rated=True):
        """
        Get top N movie recommendations for a user.
        
        Args:
            user_id (int): User ID
            movies_df (pd.DataFrame): Movies data
            n (int): Number of recommendations
            exclude_rated (bool): Whether to exclude already rated movies
            
        Returns:
            pd.DataFrame: Top N recommended movies with predicted ratings
        """
        if not self.is_trained: 
            raise ValueError("Model must be trained before making recommendations")
        
        # Get all movie IDs
        all_movie_ids = movies_df['movie_id'].unique()
        
        # Get movies already rated by the user if excluding
        if exclude_rated:
            try:
                rated_movies = set(self.trainset.ur[self.trainset.to_inner_uid(user_id)])
                rated_movie_ids = {self.trainset.to_raw_iid(iid) for iid, _ in rated_movies}
            except ValueError:
                rated_movie_ids = set()
            
            candidate_movies = [m for m in all_movie_ids if m not in rated_movie_ids]
        else:
            candidate_movies = list(all_movie_ids)
        
        # Predict ratings for all candidate movies
        predictions = []
        for movie_id in candidate_movies:
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n]
        
        # Create result DataFrame
        result = pd.DataFrame(top_n, columns=['movie_id', 'predicted_rating'])
        result = result.merge(movies_df[['movie_id', 'title']], on='movie_id')
        
        return result[['movie_id', 'title', 'predicted_rating']]
    
    def cross_validate(self, data, cv=5):
        """
        Perform cross-validation to evaluate the model.
        
        Args:
            data (pd.DataFrame): Full ratings data
            cv (int): Number of folds
            
        Returns: 
            dict: Cross-validation results
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        dataset = self.prepare_data(data)
        
        results = cross_validate(
            self.model,
            dataset,
            measures=['RMSE', 'MAE'],
            cv=cv,
            verbose=True
        )
        
        return results