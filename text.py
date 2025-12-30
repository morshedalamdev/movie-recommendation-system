def _get_hybrid_recommendations(self, user_id, n):
    """
    Generate hybrid recommendations combining both models.
    Args:
        user_id (int): User ID
        n (int): Number of recommendations
    Returns: pd.DataFrame: Hybrid recommendations
    """
    # Get more candidates than needed
    n_candidates = min(n * 3, 100)
    # Get recommendations from both models
    cf_recs = self.cf_model.get_top_n_recommendations(
        user_id, self.movies, n_candidates
    )
    cb_recs = self.cb_model.get_top_n_recommendations(user_id, n_candidates)
    # Normalize scores
    cf_recs["cf_score_norm"] = (
        cf_recs["predicted_rating"] - cf_recs["predicted_rating"].min()
    ) / (cf_recs["predicted_rating"].max() - cf_recs["predicted_rating"].min() + 1e-6)

    cb_recs["cb_score_norm"] = (
        cb_recs["content_score"] - cb_recs["content_score"].min()
    ) / (cb_recs["content_score"].max() - cb_recs["content_score"].min() + 1e-6)
    # Merge recommendations
    merged = pd.merge(
        cf_recs[["movie_id", "title", "cf_score_norm"]],
        cb_recs[["movie_id", "cb_score_norm"]],
        on="movie_id",
        how="outer",
    )
    # Fill missing scores with 0
    merged["cf_score_norm"] = merged["cf_score_norm"].fillna(0)
    merged["cb_score_norm"] = merged["cb_score_norm"].fillna(0)
    # Calculate hybrid score
    merged["hybrid_score"] = (
        self.cf_weight * merged["cf_score_norm"]
        + self.cb_weight * merged["cb_score_norm"]
    )
    # Fill missing titles
    merged = merged.merge(
        self.movies[["movie_id", "title"]],
        on="movie_id",
        how="left",
        suffixes=("", "_y"),
    )
    merged["title"] = merged["title"].fillna(merged["title_y"])
    # Sort and return top N
    result = merged.nlargest(n, "hybrid_score")

    return result[["movie_id", "title", "hybrid_score"]]
