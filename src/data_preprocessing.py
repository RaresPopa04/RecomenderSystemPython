import pandas as pd
import numpy as np

def load_data():
    ratings_file = '../data/ratings.csv'
    movies_file = '../data/movies.csv'

    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)

    movie_info = {}
    for row in movies_df.itertuples():
        movieId = row.movieId
        title = row.title
        genres = row.genres
        movie_info[movieId] = (title, genres)


    # Create a matrix of users and movies
    # users as rows and movies as columns
    data_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

    data_matrix.fillna(0, inplace=True)

    V = data_matrix.values.astype(np.float32)

    return V, movie_info
