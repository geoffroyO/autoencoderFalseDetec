import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.python.keras import Input, Model

import blurredVae as b
import vae as v

def enumMatrix(N, M, block_size):
    enum = np.zeros((N, M))
    for i in range(N-block_size+1):
        for j in range(M-block_size+1):
            enum[i:(i+block_size), j:(j+block_size)] += 1
    return enum


def predendVae(model, img, block_size):
    N, M, C = img.shape
    reconstuction_img, features_img, mask_error = np.zeros((N, M, 30)), np.zeros((N, M, 30)), np.zeros((N, M)) #modif

    blocks = []
    print("... Creating blocks")
    for i in tqdm(range(N-block_size+1, 32)):
        for j in range(M-block_size+1, 32):
            blocks.append(img[i:(i+block_size), j:(j+block_size)])

    blocks = np.array(blocks)
    print("******{}******".format(blocks.shape))
    features, reconstruction, error = model.predict(blocks)
    count = 0
    print("... Prediction for each blocks")
    for i in tqdm(range(N-block_size+1, 32)):
        for j in range(M-block_size+1, 32):
            mask_error_pred = error[count]
            mask_error[i:(i+block_size), j:(j+block_size)] += mask_error_pred

            block_reconstruction = reconstruction[count]
            reconstuction_img[i:(i+block_size), j:(j+block_size)] += block_reconstruction

            block_features = features[count]
            features_img[i:(i+block_size), j:(j+block_size)] += block_features
            count += 1

    return reconstuction_img, features_img, mask_error


def test_endVae():
    pathModel = "../models/vae_250.hdf5"

    encoder = v.encoder()
    decoder = v.decoder()
    model = v.vae(encoder, decoder)
    path = "./easy_test/{}.jpg".format(1)
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.
    model.predict(np.array([img[0:32, 0:32]]))

    model.load_weights(pathModel)
    model.predict(np.array([img[0:32, 0:32]]))


    for k in range(1, 4):
        path = "./easy_test/{}.jpg".format(k)

        img = cv2.imread(path, 1)
        img = img[..., ::-1]
        img = img.astype('float32') / 255.

        reconstruction, features, error = predendVae(model, img, 32)
        np.save("./easy_test/{}_reconstruction_v.npy".format(k), reconstruction)
        np.save("./easy_test/{}_features_v.npy".format(k), features)
        np.save("./easy_test/{}_error_v.npy".format(k), error)


if __name__ == '__main__':
    test_endVae()