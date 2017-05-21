import numpy as np
from tqdm import tqdm
"""
File description: MovieID, UserID, Rating
util file contain helper function.
"""

def get_dict(file, movie_dict, user_dict, movie_count=0, user_count=0):
    count = 0
    output_str = ''
    for line in tqdm(file):
        count += 1
        movie_id, user_id, rating = line.split(',')
        # get movie id:
        if movie_id not in movie_dict.keys():
            movie_dict[movie_id] = movie_count
            movie_id = movie_count
            movie_count += 1
        else:
            movie_id = movie_dict[movie_id]
        # get user id:
        if user_id not in user_dict.keys():
            user_dict[user_id] = user_count
            user_id = user_count
            user_count += 1
        else:
            user_id = user_dict[user_id]
        output_str += ','.join([str(movie_id), str(user_id), str(rating)])
    print(len(movie_dict))
    return output_str, movie_dict, user_dict, movie_count, user_count, count


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def write_dict_to_file(file_path, content_dict):
    output_str = ''
    for key in content_dict.keys():
        output_str += ','.join([str(key), str(content_dict[key])]) + '\n'
    write_to_file(file_path, output_str)


def process_raw_data(training_path, testing_path, output_folder):
    user_dict = dict()
    movie_dict = dict()

    with open(training_path, 'r') as file:
        output_str, movie_dict, user_dict, movie_count, user_count, train_count = get_dict(file, movie_dict, user_dict)
        write_to_file(output_folder + '/train.txt', output_str)    
    
    with open(testing_path, 'r') as file:
        output_str, movie_dict, user_dict, movie_count, user_count, test_count = get_dict(file, movie_dict, user_dict, movie_count, user_count)
        write_to_file(output_folder + '/test.txt', output_str)
    
    write_dict_to_file(output_folder + '/user_dict.txt', user_dict)
    write_dict_to_file(output_folder + '/movie_dict.txt', movie_dict)
    count_file_content = str(user_count) + "," + str(movie_count) + '\n' + str(train_count) + ',' + str(test_count)
    write_to_file(output_folder + '/count.txt', count_file_content)

    print("Number of movie: " + str(movie_count))
    print("Number of user: " + str(user_count))
    print("Number of train: " + str(train_count))
    print("Number of test: " + str(test_count))

    return movie_dict, user_dict, movie_count, user_count


def create_input_matrix(file_path, file_type="train"):
    n_user, n_movie, n_train, n_test = get_count(file_path)
    print(n_movie, n_user, n_train, n_test)
    if file_type == "train":
        file_path += "/train.txt"
    elif file_type == "test":
        file_path += "/test.txt"
    # Create input matrix
    matrix = np.zeros([n_user, n_movie, 5], dtype=np.float32)
    with open(file_path, 'r') as file:
        for line in file:
            movie_id, user_id, rating = line.split(',')
            movie_id, user_id, rating = int(movie_id), int(user_id), int(float(rating))
            matrix[user_id, movie_id, rating-1] = 1
    return matrix


def get_count(file_path):
    count_path = file_path + "/count.txt"
    with open(count_path, 'r') as file:
        n_movie, n_user = file.readline().split(',')
        n_train, n_test = file.readline().split(',')
    return int(n_movie), int(n_user), int(n_train), int(n_test)


def gen_batches(data, batch_size):
    """
    Generate batches from the data with batch_size
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


if __name__ == "__main__":
    training_path = './data/raw_data/TrainingRatings.txt'
    testing_path = './data/raw_data/TestingRatings.txt'
    output_folder = './data/converted_data'
    process_raw_data(training_path, testing_path, output_folder)
