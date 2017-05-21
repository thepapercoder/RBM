from rbm import RBM
import numpy as np
import utils, os


def main():
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


if __name__ == '__main__':
    main()