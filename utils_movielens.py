import numpy as np
from tqdm import tqdm
from utils import get_dict, write_dict_to_file, write_to_file

"""
File description: MovieID, UserID, Rating
util file contain helper function.
"""

def process_raw_data(training_path, output_folder):
    user_dict = dict()
    movie_dict = dict()

    with open(training_path, 'r') as file:
        output_str, movie_dict, user_dict, movie_count, user_count, train_count = get_dict(file, movie_dict, user_dict)
        write_to_file(output_folder, output_str)

    # write_dict_to_file(output_folder + '/user_dict.txt', user_dict)
    # write_dict_to_file(output_folder + '/movie_dict.txt', movie_dict)
    count_file_content = str(user_count) + "," + str(movie_count) + '\n' + str(train_count)
    # write_to_file(output_folder + '/count.txt', count_file_content)

    print("Number of movie: " + str(movie_count))
    print("Number of user: " + str(user_count))
    print("Number of train: " + str(train_count))

    return movie_dict, user_dict, movie_count, user_count


if __name__ == "__main__":
    process_raw_data("data/movielens/ratings.csv", "data/movielens/format_ratings.csv")