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
    encoder = b.encoder()
    decoder = b.decoder()

    model = b.srmAno(encoder, decoder)

    path = "./rot_test/1/2.png"
    img = cv2.imread(path, 1)
    img = img[..., ::-1]
    img = img.astype('float32') / 255.
    model.predict(np.array([img[0:32, 0:32]]))

    model.load_weights(pathModel)
    for rot in [2, 4, 6, 8, 10, 20, 60, 180]:
        for file in range(1, 11):
            path = "./rot_test/{}/".format(file) + "{}.png".format(rot)

            img = cv2.imread(path, 1)
            img = img[..., ::-1]
            img = img.astype('float32') / 255.

            error, features, reconstruction = predendVae4K(model, img, 32, 30)

            np.save("./rot_test/{}/".format(file) + "b_err_{}.npy".format(rot), error)
            np.save("./rot_test/{}/".format(file) + "b_features_{}.npy".format(rot), features)
            np.save("./rot_test/{}/".format(file) + "b_reconstruction_{}.npy".format(rot), reconstruction)


if __name__ == '__main__':
    pathModel = "../models/srmBlurred_4K_250.hdf5"

    test_endVae4K(pathModel)
