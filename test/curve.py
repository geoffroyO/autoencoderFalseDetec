import cv2
import matplotlib.pyplot as plt
import seaborn as sns; 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from pylab import savefig
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import skimage.morphology as morph
from scipy.ndimage.measurements import label
from tqdm import tqdm
from skimage import io


def dice(pred_mask, mask):
    n, m = mask.shape
    count_pred, count_ori = 0, 0
    inter = 0
    for i in range(n):
        for j in range(m):
            pred = pred_mask[i, j]
            ori = mask[i, j]
            
            count_pred += pred
            count_ori += ori
            
            if pred and ori:
                inter += 1
    return 2*inter/(count_pred+count_ori)


if __name__ == '__main__':
    recall, precision = [0 for _ in range(5)], [0 for _ in range(5)]
    for count, noise in enumerate([20, 40, 60, 80, 100]):
        for k in tqdm(range(1, 49)):
            pred_msk, msk = io.imread("./lnoise/{}/".format(k) + "{}_pred_final_gt.png".format(noise), as_gray=True), io.imread("./lnoise/{}/gt.png".format(k), as_gray=True)
            n, m = msk.shape
            tp, fp, fn = 0, 0, 0
            for i in range(n):
                for j in range(m):
                    if pred_msk and msk:
                        tp += 1
                    elif pred_msk and not msk:
                        fp += 1
                    elif not pred_msk and msk:
                        fn += 1
            if tp != 0:
                recall[count] += tp/(tp+fp)
                precision[count] += tp/(tp+fn)
        recall[count] /= 49
        precision[count] /= 49

    np.save("./precision_final.npy", np.array(precision))

    np.save("./recall_final.npy", np.array(recall))
