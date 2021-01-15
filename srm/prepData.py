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
        patchs_img = extractPatches(image, (32, 32, 3), 16)
        patchs_msk = extractPatchesMask(masks[n], (32, 32), 16)
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


def extract_no_border(type):
    data = np.load("../data/CASIA.numpy/{}.npy".format(type))
    labels = np.load("../data/CASIA.numpy/{}_msk.npy".format(type))

    n, _, _ = labels.shape
    inside = []

    for k in range(n):
        s = np.sum(labels[k])
        if s == 0 or s == 32*32:
            inside.append(data[k])

    np.svae("../data/CASIA.numpy/{}_inside.npy".format(type).format(type), inside)


if __name__ == '__main__':
    save_data()