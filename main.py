from rbm import RBM
import numpy as np
import os


def main_netflix():
    import utils
    data_path = 'data/converted_data'
    train_path = 'data/rbm/train.npy'
    val_path = 'data/rbm/test.npy'
    if os.path.isfile(train_path):
        trX = np.load(train_path)
    else:
        trX = utils.create_input_matrix(data_path)
    if os.path.isfile(val_path):
        valX = np.load(val_path)
    else:
        valX = utils.create_input_matrix(data_path)
    trX = np.reshape(trX, [trX.shape[0], -1])
    valX = np.reshape(valX, [valX.shape[0], -1])

    # Train the model
    data_shape = trX.shape
    rbm = RBM(data_shape)
    rbm.train(trX, valX)

def main_movielens():
    data_path = 'data/movielens/format_ratings.csv'
    input_matrix = np.zeros([671, 9125, 10], dtype=np.float32)
    with open(data_path, 'r') as file:
        for line in file:
            movie_id, user_id,  rating = line.split(",")
            movie_id, user_id, rating = int(movie_id), int(user_id), float(rating)*2
            input_matrix[user_id - 1 , movie_id - 1, rating - 1] = 1
    input_matrix = np.reshape(input_matrix, [input_matrix.shape[0], -1])
    data_shape = input_matrix.shape
    rbm = RBM(data_shape, n_rating=10)
    rbm.train(trX= input_matrix, valX=None)

if __name__ == '__main__':
    main_netflix()