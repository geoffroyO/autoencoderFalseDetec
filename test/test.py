import numpy as np
import cv2
from tqdm import tqdm

import vae as vae
import blurredVae as b


def predendVae4K(model, img, block_size, size_feat):
    N, M, C = img.shape
    reconstuction_img, features_img, mask_error = np.zeros((N, M, size_feat)), np.zeros((N, M, size_feat)), np.zeros((N, M))

    blocks = []
    print("... Creating blocks")
    for i in tqdm(range(0, N - block_size + 1, block_size)):
        for j in range(0, M - block_size + 1, block_size):
            blocks.append(img[i:(i + block_size), j:(j + block_size)])

    features, reconstruction, error = model.predict(np.array(blocks))

    count = 0
    print("... Prediction for each blocks")
    for i in tqdm(range(0, N - block_size + 1, block_size)):
        for j in range(0, M - block_size + 1, block_size):
            mask_error_pred = error[count]
            mask_error[i:(i + block_size), j:(j + block_size)] += mask_error_pred

            block_reconstruction = reconstruction[count]
            reconstuction_img[i:(i + block_size), j:(j + block_size)] += block_reconstruction

            block_features = features[count]
            features_img[i:(i + block_size), j:(j + block_size)] += block_features

            count += 1

    return mask_error, features_img, reconstuction_img


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

            error, features, reconstruction = predendVae4K(model, img, 32, 30)

            np.save("./lnoise/{}/".format(file) + "v_err_{}.npy".format(noise), error)
            np.save("./lnoise/{}/".format(file) + "v_features_{}.npy".format(noise), features)
            np.save("./lnoise/{}/".format(file) + "v_reconstruction_{}.npy".format(noise), reconstruction)


if __name__ == '__main__':
    pathModel = "../models/vae_250.hdf5"

    test_endVae4K(pathModel)
