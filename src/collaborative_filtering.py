import numpy as np


def recommend_movies(user_id, predicted_ratings, user_index_map, movie_index_map, n_recommendations):
    """
    Given a user's ID (in original user indexing),
    return top-N recommended movie IDs.
    """
    if user_id not in user_index_map:
        return []

    # Map original user_id to internal index
    user_index = user_index_map[user_id]
    predicted_ratings_user = predicted_ratings[user_index, :]

    # Sort movie indices by descending rating
    indices = np.argsort(predicted_ratings_user)[::-1]

    # Convert movie indices back to original movie IDs
    inv_movie_index_map = {v: k for k, v in movie_index_map.items()}
    top_n_indices = [inv_movie_index_map[i] for i in indices[:n_recommendations]]
    return top_n_indices
