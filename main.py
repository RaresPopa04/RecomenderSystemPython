from src.collaborative_filtering import recommend_movies
from src.content_based import recommend_for_user
from src.data_preprocessing import load_data
from src.nmf import calculate_prediction_matrix


def main():
    data_matrix, movie_genres, movie_index_map, user_index_map, movies = load_data()
    prediction_matrix = calculate_prediction_matrix(data_matrix)
    print(recommend_movies(1, prediction_matrix, user_index_map, movie_index_map))

    print(recommend_for_user(1, data_matrix, movie_genres, movies, movie_index_map, user_index_map))


if __name__ == "__main__":
    main()
