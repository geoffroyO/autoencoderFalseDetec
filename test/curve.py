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

precision, recall = [0 for _ in range(5)], [0 for _ in range(5)]

for count, noise in tqdm(enumerate([20, 40, 60, 80, 100])):
    for k in tqdm(range(1, 49)):
        pred_msk, msk = io.imread("./lnoise/{}/".format(k) + "{}_gt.png".format(noise), as_gray=True), io.imread("./lnoise/{}/gt.png".format(k), as_gray=True)
        n, m = np.shape(pred_msk)
        fp, tp, fn = 0, 0, 0
        for i in range(n):
            for j in range(m):
                pred, true = pred_msk[i, j], msk[i, j]/255
                if pred and not true:
                    fp += 1
                elif pred and true:
                    tp += 1
                elif not pred and true:
                    fn += 1
        if tp+fp != 0:
            precision[count] += tp/(tp+fp) 
        else:
            precision[count] += 0
        if tp+fn != 0:
                recall[count] += tp/(tp+fn)
        recall[count] += 0
        
    precision[count] /= 49
    recall[count] /= 49   

np.save("./precision.npy", np.array(precision))
np.save("./recall.npy", np.array(recall))