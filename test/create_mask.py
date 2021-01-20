import numpy as np
from tqdm import tqdm
import matplolib.pyplot as plt

from sklearn.cluster import KMeans
import skimage.morphology as morph
from scipy.ndimage.measurements import label

def gen_msk():
    for noise in tqdm([20, 40, 60, 80, 100]):
        for k in tqdm(range(1, 49)):
            features = np.load("./lnoise/{}/".format(k)+"b_features_{}.npy".format(noise))
            reconstruction = np.load("./lnoise/{}/".format(k)+"b_features_{}.npy".format(noise))

            error = np.abs(features - reconstruction)
            error = np.sum(error, axis=-1)

            """ K-Means """
            error_list = []
            n, m = error.shape
            for i in range(n):
                for j in range(m):
                    error_list.append([error[i, j]])

            kmeans = KMeans(n_clusters=5, random_state=0).fit(error_list)

            counts = [0, 0, 0, 0, 0]
            for e in kmeans.labels_:
                counts[e] += 1

            # Keeping the less numerous classes
            error_2 = np.zeros((n, m))
            count = 0
            for i in range(n):
                for j in range(m):
                    classe = kmeans.labels_[count]
                    if classe != np.argmax(counts):
                        error_2[i, j] = 1
                    count += 1

            """ Morphologie """
            closing = morph.binary_closing(error_2, morph.square(4))
            opening = morph.binary_opening(closing, morph.square(4))

            """ CC """
            labeled_array, num_features = label(opening)

            count = [0 for _ in range(num_features)]
            for i in range(n):
                for j in range(m):
                    lab = labeled_array[i, j]
                    if lab != 0:
                        count[lab - 1] += 1

            error_final = np.zeros(error.shape)
            for i in range(n):
                for j in range(m):
                    if labeled_array[i, j] == np.argmax(count) + 1:
                        error_final[i, j] = 1

            """ Morphologie """
            closing_2 = morph.binary_closing(error_final, morph.square(15))
            dilatation = morph.binary_dilation(closing_2)

            plt.imsave("./lnoise/{}/".format(k) + "{}_gt.png".format(noise), format='png', cmap='gray')
    return None

if __name__ == '__main__':
    gen_msk()