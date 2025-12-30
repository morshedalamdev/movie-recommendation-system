"""
Model Evaluation Module
Implements various metrics for evaluating recommendation systems
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RecommenderEvaluator: 
    """
    Evaluates recommendation system performance using various metrics.
    
    Metrics implemented:
    - RMSE (Root Mean Square Error): Measures prediction accuracy
    - MAE (Mean Absolute Error): Average absolute prediction error
    - Precision@K:  Fraction of relevant items in top-K recommendations
    - Recall@K:  Fraction of relevant items that appear in top-K
    - Coverage:  Percentage of items that can be recommended
    - Diversity: How diverse the recommendations are
    """
    
    def __init__(self, rating_threshold=3.5):
        """
        Initialize the evaluator.
        
        Args:
            rating_threshold (float): Threshold for considering an item as relevant
        """
        self.rating_threshold = rating_threshold
        self.results = {}
        
    def rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Square Error.
        
        RMSE = sqrt(mean((y_true - y_pred)^2))
        
        Args: 
            y_true (array): Actual ratings
            y_pred (array): Predicted ratings
            
        Returns:
            float: RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        MAE = mean(|y_true - y_pred|)
        
        Args:
            y_true (array): Actual ratings
            y_pred (array): Predicted ratings
            
        Returns: 
            float: MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Precision@K.
        
        Precision@K = |relevant ∩ recommended[: k]| / k
        
        Args:
            recommended_items (list): List of recommended item IDs (ranked)
            relevant_items (set): Set of relevant item IDs
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Precision@K value
        """
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / k if k > 0 else 0.0
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Recall@K.
        
        Recall@K = |relevant ∩ recommended[:k]| / |relevant|
        
        Args:
            recommended_items (list): List of recommended item IDs (ranked)
            relevant_items (set): Set of relevant item IDs
            k (int): Number of top recommendations to consider
            
        Returns:
            float:  Recall@K value
        """
        if len(relevant_items) == 0:
            return 0.0
            
        recommended_k = recommended_items[: k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def evaluate_predictions(self, test_data, predictions):
        """
        Evaluate rating predictions using RMSE and MAE.
        
        Args:
            test_data (pd.DataFrame): Test data with actual ratings
            predictions (array): Predicted ratings
            
        Returns:
            dict:  Dictionary with RMSE and MAE values
        """
        y_true = test_data['rating'].values
        y_pred = predictions
        
        results = {
            'RMSE': self.rmse(y_true, y_pred),
            'MAE': self.mae(y_true, y_pred)
        }
        
        print("\n" + "="*50)
        print("PREDICTION EVALUATION RESULTS")
        print("="*50)
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"MAE:   {results['MAE']:.4f}")
        print("="*50)
        
        self.results['prediction'] = results
        return results
    
    def evaluate_recommendations(self, model, test_data, k=10, sample_users=100):
        """
        Evaluate recommendation quality using Precision@K and Recall@K.
        
        Args: 
            model:  Trained recommendation model
            test_data (pd.DataFrame): Test data with actual ratings
            k (int): Number of recommendations to evaluate
            sample_users (int): Number of users to sample for evaluation
            
        Returns:
            dict:  Dictionary with Precision@K and Recall@K values
        """
        print(f"\nEvaluating recommendations (K={k})...")
        
        # Get unique users from test data
        test_users = test_data['user_id'].unique()
        sample_size = min(sample_users, len(test_users))
        sampled_users = np.random.choice(test_users, size=sample_size, replace=False)
        
        precision_scores = []
        recall_scores = []
        
        for user_id in sampled_users: 
            # Get relevant items (items user rated highly in test set)
            user_test = test_data[test_data['user_id'] == user_id]
            relevant_items = set(
                user_test[user_test['rating'] >= self.rating_threshold]['movie_id']
            )
            
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations from model
            try:
                if hasattr(model, 'get_top_n_recommendations'):
                    recs = model.get_top_n_recommendations(user_id, n=k)
                    recommended_items = recs['movie_id'].tolist()
                else:
                    continue
            except Exception: 
                continue
            
            # Calculate metrics
            precision = self.precision_at_k(recommended_items, relevant_items, k)
            recall = self.recall_at_k(recommended_items, relevant_items, k)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        results = {
            f'Precision@{k}': np.mean(precision_scores) if precision_scores else 0,
            f'Recall@{k}': np.mean(recall_scores) if recall_scores else 0,
            'users_evaluated': len(precision_scores)
        }
        
        print("\n" + "="*50)
        print("RECOMMENDATION EVALUATION RESULTS")
        print("="*50)
        print(f"Precision@{k}: {results[f'Precision@{k}']:.4f}")
        print(f"Recall@{k}:    {results[f'Recall@{k}']:.4f}")
        print(f"Users evaluated:  {results['users_evaluated']}")
        print("="*50)
        
        self.results['recommendation'] = results
        return results
    
    def calculate_coverage(self, model, all_items, sample_users, k=10):
        """
        Calculate catalog coverage - what fraction of items get recommended.
        
        Args:
            model: Trained recommendation model
            all_items (list): All possible item IDs
            sample_users (list): Users to generate recommendations for
            k (int): Number of recommendations per user
            
        Returns:
            float: Coverage percentage
        """
        recommended_items = set()
        
        for user_id in sample_users: 
            try:
                recs = model.get_top_n_recommendations(user_id, n=k)
                recommended_items.update(recs['movie_id'].tolist())
            except Exception:
                continue
        
        coverage = len(recommended_items) / len(all_items)
        
        print(f"Coverage:  {coverage:.2%} ({len(recommended_items)}/{len(all_items)} items)")
        
        return coverage
    
    def generate_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            str: Formatted evaluation report
        """
        report = []
        report.append("\n" + "="*60)
        report.append("         RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("="*60)
        
        if 'prediction' in self.results:
            report.append("\n📊 Prediction Accuracy:")
            report.append(f"   • RMSE: {self.results['prediction']['RMSE']:.4f}")
            report.append(f"   • MAE:  {self.results['prediction']['MAE']:.4f}")
        
        if 'recommendation' in self.results:
            report.append("\n🎯 Recommendation Quality:")
            for key, value in self.results['recommendation'].items():
                if isinstance(value, float):
                    report.append(f"   • {key}: {value:.4f}")
                else: 
                    report.append(f"   • {key}: {value}")
        
        report.append("\n" + "="*60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text