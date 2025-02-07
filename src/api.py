from wsgiref.util import application_uri

from flask import Flask, request, jsonify

from src.collaborative_filtering import recommend_movies
from src.content_based import recommend_for_user
from src.data_preprocessing import load_data
from src.nmf import calculate_prediction_matrix

app = Flask(__name__)

data_matrix = None
prediction_matrix = None
movie_genres = None
movie_index_map = None
user_index_map = None
movies_df = None

def load_model_artifacts():
    """
    This function runs once before the first request is handled.
    We load (or train) the model so that we can handle recommendation queries quickly.
    """
    global data_matrix, movie_genres, movie_index_map, user_index_map, movies_df
    data_matrix, movie_genres, movie_index_map, user_index_map, movies_df = load_data()

    # Load the pre-trained collaborative filtering model
    global prediction_matrix
    prediction_matrix = calculate_prediction_matrix(data_matrix)


@app.route('/recommend/collaborative', methods=['GET'])
def recommend_collaborative():
    """
    Recommend movies for a user using collaborative filtering.
    """
    user_id = int(request.args.get('user_id', '0'))
    n_recommendations = int(request.args.get('n_recommendations', 10))

    if user_id not in user_index_map:
        return jsonify({"recommendations": [], "message": "User ID not found."})

    recommended_movies = recommend_movies(user_id, prediction_matrix, user_index_map, movie_index_map, n_recommendations)

    return jsonify({"recommendations": recommended_movies})

@app.route('/recommend/content', methods=['GET'])
def recommend_content():
    """
    Recommend movies for a user using content-based filtering.
    """
    user_id = int(request.args.get('user_id'))
    n_recommendations = int(request.args.get('n_recommendations', 10))

    if user_id not in user_index_map:
        return jsonify({"recommendations": [], "message": "User ID not found."})

    recommended_movies = recommend_for_user(user_id, data_matrix, movie_genres, movies_df, movie_index_map, user_index_map, n_recommendations)

    return jsonify(recommended_movies)

def main():
    app.run(port=5000, debug=True)

if __name__ == "__main__":
    load_model_artifacts()
    main()



