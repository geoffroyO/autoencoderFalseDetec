import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.python.keras import Input, Model

import blurredVae as b


def enumMatrix(N, M, block_size):
    enum = np.zeros((N, M))
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            enum[i:(i+block_size), j:(j+block_size)] += 1
    return enum


def pred(model, img, block_size):
    N, M, _ = img.shape
    mask = np.zeros((N, M))
    blocks = []
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])
    blocks = np.array(blocks)
    pred = model.predict(blocks)
    count = 0
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            mask_pred = pred[count]
            mask[i:(i+block_size), j:(j+block_size)] += mask_pred[0]
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask /= enum
    return mask

def predendVae(model, img, block_size):
    N, M, C = img.shape
    reconstuction_img, features_img, mask_error = np.zeros((N, M, C)), np.zeros((N, M, C)), np.zeros((N, M))

    blocks = []
    print("... Creating blocks")
    for i in tqdm(range(N-block_size+1)):
        for j in range(M-block_size+1):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])

    blocks = np.array(blocks)
    features, reconstruction, error = model.predict(blocks)
    count = 0
    print("... Prediction for each blocks")
    for i in tqdm(range(N-block_size+1)):
        for j in range(M-block_size+1):
            mask_error_pred = error[count]
            mask_error[i:(i+block_size), j:(j+block_size)] += mask_error_pred

            block_reconstruction = reconstruction[count]
            reconstuction_img[i:(i+block_size), j:(j+block_size)] += block_reconstruction

            block_features = features[count]
            features_img[i:(i+block_size), j:(j+block_size)] += block_features
            count += 1
    enum = enumMatrix(N, M, block_size)
    mask_error /= enum
    enum_3D = np.dstack((enum, enum))
    enum_3D = np.dstack((enum_3D, enum))
    reconstuction_img /= enum_3D
    features_img /= enum_3D
    return reconstuction_img, features_img, mask_error

def test_endVae():
    pathModel = "../models/blurredVae_250.hdf5"

    encoder = b.encoder()
    decoder = b.decoder()
    model = b.srmAno(encoder, decoder)
    path = "./easy_test/{}.jpg".format(1)
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.
    model.predict(np.array([img[0:32, 0:32]]))

    model.load_weights(pathModel)

    for k in range(1, 4):
        path = "./easy_test/{}.jpg".format(k)

        img = cv2.imread(path, 1)
        img = img[..., ::-1]
        img = img.astype('float32') / 255.

        reconstruction, features, error = predendVae(model, img, 32)
        np.save("./easy_test/{}_reconstruction.npy".format(k), reconstruction)
        np.save("./easy_test/{}_features.npy".format(k), features)
        np.save("./easy_test/{}_error.npy".format(k), error)


if __name__ == '__main__':
    test_endVae()