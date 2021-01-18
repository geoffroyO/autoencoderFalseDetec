import numpy as np
import cv2
from tqdm import tqdm
import sys

sys.path.append('../src/')
import vae as vae


def predendVae4K(model, img, block_size):
    N, M, C = img.shape
    mask_error = np.zeros((N, M))

    blocks = []
    print("... Creating blocks")
    for i in tqdm(range(0, N - block_size + 1, block_size)):
        for j in range(0, M - block_size + 1, block_size):
            blocks.append(img[i:(i + block_size), j:(j + block_size)])

    error = model.predict(np.array(blocks))

    count = 0
    print("... Prediction for each blocks")
    for i in tqdm(range(0, N - block_size + 1, block_size)):
        for j in range(0, M - block_size + 1, block_size):
            mask_error_pred = error[count]
            mask_error[i:(i + block_size), j:(j + block_size)] += mask_error_pred

            count += 1

    return mask_error


def test_endVae4K(pathModel):
    encoder = vae.encoder()
    decoder = vae.decoder()

    model = vae.vae(encoder, decoder)

    path = "./lnoise/1/0.png"
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.
    model.predict(np.array([img[0:32, 0:32]]))

    model.load_weights(pathModel)
    for noise in [0, 20, 40, 60, 80, 100]:
        for file in range(1, 49):
            path = "./lnoise/{}/".format(file) + "{}.png".format(noise)

            img = cv2.imread(path, 1)
            img = img[..., ::-1]
            img = img.astype('float32') / 255.

            error = predendVae4K(model, img, 32)
            np.save("./lnoise/{}/".format(file) + "{}.npy".format(noise), error)


if __name__ == '__main__':
    pathModel = "../models/vae_76.h5"

    test_endVae4K(pathModel)
