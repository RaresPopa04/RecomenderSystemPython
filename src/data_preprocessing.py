import pandas as pd
import numpy as np


def load_data():
    # Load the datasets from CSV files
    ratings = pd.read_csv('../data/ratings.csv')
    movies = pd.read_csv('../data/movies.csv')

    # Pivot the ratings DataFrame to create a matrix with userIds as rows, movieIds as columns, and ratings as values
    data_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

    # Add all the movieIds to the columns of the data_matrix
    all_movie_ids = movies['movieId'].values
    data_matrix = data_matrix.reindex(columns=all_movie_ids)

    # Initialize dictionaries to map user IDs and movie IDs to their corresponding matrix indices
    user_index_map = {}
    movie_index_map = {}

    movie_genres = {}

    for row in movies.itertuples():
        index = row.movieId
        title = row.title
        genres = row.genres
        movie_genres[index] = (title, genres.split('|'))

    for i, user_id in enumerate(data_matrix.index):
        user_index_map[user_id] = i

    for i, movie_id in enumerate(data_matrix.columns):
        movie_index_map[movie_id] = i

    return data_matrix, movie_genres, movie_index_map, user_index_map, movies


def build_movie_dict(movies, movie_genres, movie_index_map):
    """
    Create a dictionary mapping movie IDs to movie titles.
    """

    # Create a dictionary mapping movie IDs to movie titles
    all_genres = set()
    for movie in movie_genres:
        genres = movie_genres[movie][1]
        for g in genres:
            if g != '(no genres listed)':
                all_genres.add(g)

    all_genres = sorted(list(all_genres))
    genre_index_map = {genre: i for i, genre in enumerate(all_genres)}

    movie_genre_matrix = np.zeros((len(movies), len(all_genres)), dtype=np.float32)

    # Now create a matrix (num_movies, num_genres)
    for movie in movie_genres:

        genres = movie_genres[movie][1]
        movie_index = movie_index_map[movie]
        if '(no genres listed)' in genres:
            continue
        if movie_index is not None:
            for genre in genres:
                if genre in genre_index_map:
                    genre_index = genre_index_map[genre]
                    movie_genre_matrix[movie_index, genre_index] = 1.0

    return genre_index_map, movie_genre_matrix
