"""
Movie Recommendation System
Main entry point for the application

Introduction to Artificial Intelligence - Final Project
"""

import argparse
import numpy as np
import pandas as pd

from src.data_loader import MovieLensDataLoader
from src.preprocessing import DataPreprocessor
from src.recommender import HybridRecommender
from src.evaluation import RecommenderEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System - AI Course Project'
    )
    parser.add_argument(
        '--user_id', 
        type=int, 
        default=1,
        help='User ID to generate recommendations for'
    )
    parser.add_argument(
        '--num_recommendations', 
        type=int, 
        default=10,
        help='Number of recommendations to generate'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='hybrid',
        choices=['hybrid', 'collaborative', 'content'],
        help='Recommendation method to use'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the recommendation system."""
    
    print("\n" + "="*60)
    print("     🎬 MOVIE RECOMMENDATION SYSTEM 🎬")
    print("     Introduction to Artificial Intelligence")
    print("="*60)
    
    args = parse_arguments()
    
    # =========================================
    # Step 1: Load Data
    # =========================================
    print("\n📁 STEP 1: Loading Dataset")
    print("-" * 40)
    
    data_loader = MovieLensDataLoader()
    ratings, movies, users = data_loader.load_all_data()
    
    # =========================================
    # Step 2: Preprocess Data
    # =========================================
    print("\n🔧 STEP 2: Preprocessing Data")
    print("-" * 40)
    
    preprocessor = DataPreprocessor(ratings, movies, users)
    preprocessor.clean_data()
    
    # Display statistics
    stats = preprocessor.get_statistics()
    
    # Split into train/test
    train_data, test_data = preprocessor.create_train_test_split(
        test_size=0.2, 
        random_state=42
    )
    
    # =========================================
    # Step 3: Train Models
    # =========================================
    print("\n🧠 STEP 3: Training Models")
    print("-" * 40)
    
    recommender = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
    recommender.train(train_data, movies, ratings)
    
    # =========================================
    # Step 4: Evaluate Models
    # =========================================
    print("\n📊 STEP 4: Evaluating Models")
    print("-" * 40)
    
    evaluator = RecommenderEvaluator(rating_threshold=3.5)
    
    # Evaluate prediction accuracy
    print("\nEvaluating prediction accuracy on test set...")
    predictions = recommender.cf_model.predict_batch(test_data)
    evaluator.evaluate_predictions(test_data, predictions)
    
    # Evaluate recommendation quality
    evaluator.evaluate_recommendations(
        recommender.cf_model, 
        test_data, 
        k=10, 
        sample_users=50
    )
    
    # =========================================
    # Step 5: Generate Recommendations
    # =========================================
    print("\n🎯 STEP 5: Generating Recommendations")
    print("-" * 40)
    
    user_id = args.user_id
    n_recs = args.num_recommendations
    method = args.method
    
    print(f"\nGenerating {n_recs} recommendations for User {user_id}")
    print(f"Method:  {method.upper()}")
    
    recommendations = recommender.get_recommendations(
        user_id=user_id,
        n=n_recs,
        method=method
    )
    
    print(f"\n📽️  TOP {n_recs} MOVIE RECOMMENDATIONS FOR USER {user_id}:")
    print("-" * 60)
    
    for i, row in recommendations.iterrows():
        rank = list(recommendations.index).index(i) + 1
        print(f"{rank: 2d}.{row['title'][: 50]: <50}")
    
    # =========================================
    # Step 6: Show Similar Movies Example
    # =========================================
    print("\n🔍 STEP 6: Finding Similar Movies (Example)")
    print("-" * 40)
    
    # Use Toy Story (movie_id=1) as an example
    example_movie_id = 1
    example_movie_title = movies[movies['movie_id'] == example_movie_id]['title'].values[0]
    
    print(f"\nMovies similar to '{example_movie_title}':")
    similar_movies = recommender.get_similar_movies(example_movie_id, n=5)
    
    for i, row in similar_movies.iterrows():
        print(f"  • {row['title'][:45]: <45} (similarity: {row['similarity_score']:.2f})")
    
    # =========================================
    # Step 7: Generate Final Report
    # =========================================
    print("\n📋 STEP 7: Final Evaluation Report")
    print("-" * 40)
    
    evaluator.generate_report()
    
    print("\n✅ Recommendation system execution complete!")
    print("="*60 + "\n")
    
    return recommender, evaluator


if __name__ == "__main__":
    recommender, evaluator = main()