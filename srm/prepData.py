import cv2
import os
import skimage
import numpy as np
from tqdm import tqdm


def load_images(path_img, path_msk):
    spliced, copy_moved = [], []
    names = os.listdir(path_img)
    names.sort()
    spliced_msk, copy_moved_msk = [], []
    for name in tqdm(names):
        mask_name = name[:-4] + "_gt.png"
        msk = cv2.imread(path_msk + mask_name, 0)
        img = cv2.imread(path_img + name, 1)
        if msk is not None:
            msk = cv2.flip(msk, 1)
            Nx_msk, Ny_msk = msk.shape
            Nx_img, Ny_img, _ = img.shape
            if Nx_msk != Nx_img:
                img = cv2.transpose(img)
            Nx_msk, Ny_msk = msk.shape
            Nx_img, Ny_img, _ = img.shape
            if Nx_msk == Nx_img and Ny_msk == Ny_img:
                if name.split(".")[-1].lower() in {"jpeg", "jpg", "png", 'tif'}:
                    if name[3] == 'D':
                        spliced.append(img[..., ::-1])
                        spliced_msk.append(msk[..., ::-1])
                    else:
                        copy_moved.append(img[..., ::-1])
                        copy_moved_msk.append(msk[..., ::-1])
    return spliced, copy_moved, spliced_msk, copy_moved_msk


def patch_images(images, masks):
    data, labels = [], []
    for n, image in enumerate(tqdm(images)):
        patchs_img = extractPatches(image, (32, 32, 3), 8)
        patchs_msk = extractPatchesMask(masks[n], (32, 32), 8)
        for k, patch_img in enumerate(patchs_img):
            patch_msk = patchs_msk[k]
            data.append(patch_img)
            labels.append(patch_msk)
    return data, labels


def extractPatches(im, window_shape, stride):
    patches = skimage.util.view_as_windows(im, window_shape, stride)
    nR, nC, t, H, W, C = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W, C))

    return patches


def extractPatchesMask(msk, window_shape, stride):
    patches = skimage.util.view_as_windows(msk, window_shape, stride)
    nR, nC, H, W = patches.shape
    nWindow = nR * nC
    patches = np.reshape(patches, (nWindow, H, W))

    return patches


def applySRM(img, srmkernel):
    lst_srm = []
    for k in range(30):
        filter_srm = srmkernel[:,:,0,k]
        filtered = cv2.filter2D(img,-1,filter_srm)
        lst_srm.append(filtered)
    return np.concatenate(lst_srm, axis=-1)


def filter_data():
    srm = np.loadtxt('rich_model.txt')
    srmkernel = np.float32(srm)
    srmkernel = np.reshape(srmkernel, [30, 1, 5, 5])
    srmkernel = np.transpose(srmkernel, (2, 3, 1, 0))

    img = np.load('PATH')
    n, _, _, _ = img.shape
    img_srm = []
    for k in range(n):
        img_srm.append(applySRM(img[k], srmkernel))
    np.save('PATH', img_srm)


def save_data():
    path_img, path_msk = "../data/CASIA2/Tp/", "../data/CASIA2/gt/"

    spliced, copy_moved, spliced_msk, copy_moved_msk = load_images(path_img, path_msk)

    data_spliced, labels_spliced = patch_images(spliced, spliced_msk)
    data_cp, labels_cp = patch_images(copy_moved, copy_moved_msk)

    np.save("../data/CASIA.numpy/spliced.npy", data_spliced)
    np.save("../data/CASIA.numpy/spliced_msk.npy", labels_spliced)

    np.save("../data/CASIA.numpy/copymoved.npy", data_cp)
    np.save("../data/CASIA.numpy/copymoved_msk.npy", labels_cp)

    np.save("../data/CASIA.numpy/all.npy", np.concatenate((data_spliced, data_cp)))
    np.save("../data/CASIA.numpy/all_msk.npy", np.concatenate((labels_spliced, labels_cp)))


def extract_no_border():
    data = np.load("../data/CASIA.numpy/all.npy")
    labels = np.load("../data/CASIA.numpy/all_msk.npy")

    n, _, _ = labels.shape
    inside = []

    for k in range(n):
        s = np.sum(labels[k])
        if s == 0 or s == 32*32:
            inside.append(data[k])

    np.svae("../data/CASIA.numpy/inside.npy", inside)


if __name__ == '__main__':
    save_data()