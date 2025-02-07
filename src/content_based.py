import numpy as np

from src.data_preprocessing import build_movie_dict


def build_user_profile(user_id, user_index_map, data_matrix, movie_genre_matrix, rating_threshold=4.0):
    """
    Build a user's profile vector by averaging the genre vectors
    of movies they rated >= 'rating_threshold'.
    """

    user_index = user_index_map[user_id]
    # Extract the user's ratings row
    user_ratings = data_matrix[user_index, :]
    liked_movies = np.argwhere(user_ratings >= rating_threshold).flatten()

    if len(liked_movies) == 0:
        return np.zeros(movie_genre_matrix.shape[1])

    # Average the genre vectors of the liked movies
    # the values in the final user_profile are between 0 and 1, 1 means the user likes that genre and 0 means the user doesn't like that genre
    user_profile = np.mean(movie_genre_matrix[liked_movies, :], axis=0)
    # Normalize the user profile vector
    norm = np.linalg.norm(user_profile)
    if norm > 0:
        user_profile /= norm

    return user_profile


def normalize_movie_genres(movie_genre_matrix):
    """
    Normalize the movie genre matrix by dividing each row by its norm.
    """

    for i in range(movie_genre_matrix.shape[0]):
        norm = np.linalg.norm(movie_genre_matrix[i, :])
        if norm > 0:
            movie_genre_matrix[i, :] /= norm
    return movie_genre_matrix


def recommend_content(user_id, data_matrix, normalized_movie_genres, movie_index_map, user_index_map, movie_genre_matrix, number_recommendations):
    """
    Recommend the top-N movies for a user based on content filtering.
    """

    user_profile = build_user_profile(user_id, user_index_map, data_matrix, movie_genre_matrix)

    # Dot product with each movie's vector to obtain the scores measuring the relevance of each movie to the user's preferences
    scores = normalized_movie_genres @ user_profile

    # Exclude movies the user has already rated
    user_index = user_index_map[user_id]
    user_ratings = data_matrix[user_index, :]
    rated_indices = np.argwhere(user_ratings > 0).flatten()

    assert len(rated_indices) == len(user_ratings[~np.isnan(user_ratings)])

    # We only want to consider movies not rated by this user
    mask = np.ones(len(scores), dtype=bool)
    mask[rated_indices] = False

    # Sort by descending score and return the top N indices
    indices = np.argsort(scores)[::-1]
    indices = indices[mask[indices]]
    top_n_indices = indices[:number_recommendations]

    # Convert back to movie IDs
    inv_movie_index_map = {v: k for k, v in movie_index_map.items()}

    top_n_movies = []
    for i in top_n_indices:
        if i in inv_movie_index_map:
            top_n_movies.append(inv_movie_index_map[i])

    return top_n_movies


def recommend_for_user(user_id, data_matrix, movie_genres, movies, movie_index_map, user_index_map, number_recommendations):
    genre_index_map, movie_genre_matrix = build_movie_dict(movies, movie_genres, movie_index_map)
    normalized_movie_genres = normalize_movie_genres(movie_genre_matrix)
    recommended_movies = recommend_content(user_id, data_matrix.values, normalized_movie_genres, movie_index_map, user_index_map, movie_genre_matrix, number_recommendations)
    return recommended_movies
