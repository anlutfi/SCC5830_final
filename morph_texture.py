# Morphology and Haralick tools from assignment 5

import numpy as np
import cv2



def normalize(img, value_range=(0, 1), type=float, by_channel=False):
    """returns an image normalized in a given range and type
    if :param by_channel: is True, normalizes each color channel separately"""
    result = img.astype(float)
    if len(img.shape) == 3 and by_channel:
        for i in range(result.shape[2]):
            result[:,:,i] = normalize(result[:, :, i], value_range)
    else:
        result -= np.min(img)
        imax = np.max(result)
        result /= imax if imax != 0 else 1
        result = (result * (value_range[1] - value_range[0]) + value_range[0])
    
    return result.astype(type)


def get_masks(img, morph_f):
    """performs morphology operation over :param img:
    :param bin_thresh: is the binarization threshold
    :param morph_f: is the morphology transformation function to use(open or close)

    it converts :param img: to grayscale (gray_img); binarizes it
    then performs the morphology (lmorph)
    returns a tuple of 2 images, one containing the pixels in gray_img
    corresponding to 0 valued pixels in lmorph; and the other
    the remaining pixels
    """
    gray_img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bin_img = normalize(cv2.threshold(gray_img, 0, 1, cv2.THRESH_OTSU)[1])
    lmorph = morph_f(bin_img)
    
    mask_0 = np.zeros(gray_img.shape).astype(gray_img.dtype)
    idxs = np.where(lmorph == 0)
    mask_0[idxs] = gray_img[idxs]

    mask_1 = np.zeros(gray_img.shape).astype(gray_img.dtype)
    idxs = np.where(lmorph == 1)
    mask_1[idxs] = gray_img[idxs]

    return mask_0, mask_1


euclidean_distance = lambda a, b: np.sqrt(np.sum(np.square(a - b)))


def get_haralick(img, neighbor):
    """calculate haralick descriptor for a given :param img:
    :param neighbor: is the neighborhood relationship used to calculate
    the image's co-occurrence matrix
    """
    # since all chosen Haralick descriptors don't change when a factor
    # inside the sum is zero, there's no need to keep the zero probabilities.
    # so the co-occurrence matrix is a dictionary containing only the pairs
    # that have positive occurence counts
    cooc_mat = {}
    for h in range(1, img.shape[0] - 1):
        for w in range(1, img.shape[1] - 1):
            pair = (float(img[h, w]), float(img[h + neighbor[0], w + neighbor[1]]))
            # if it's the first time a pair of values appear, start its count
            if pair not in cooc_mat:
                cooc_mat[pair] = 0
            cooc_mat[pair] += 1

    # normalizing co-occurence matrix into probability distribution
    denom = sum([cooc_mat[(i, j)] for (i, j) in cooc_mat])
    cooc_mat = {key: cooc_mat[key] / denom for key in cooc_mat}

    # calculating haralick desriptors
    # since p(i, j) is never zero, there is no need to worry about divisions
    # or log(0)
    auto_corr = 0.0
    contrast = 0.0
    dissimil = 0.0
    energy = 0.0
    entropy = 0.0
    homog = 0.0
    invdiff = 0.0
    maxp = 0.0

    for (i, j) in cooc_mat:
        pij = cooc_mat[(i, j)]
        
        auto_corr += i * j * pij
        contrast  += ((i - j) ** 2) * pij
        dissimil  += abs(i - j) * pij
        energy    += pij ** 2
        entropy   -= pij * np.log2(pij)
        homog     += pij / (1 + (i - j) ** 2)
        invdiff   += pij / (1 + abs(i - j))
        maxp       = max(maxp, pij)

    return np.array([auto_corr,
                     contrast,
                     dissimil,
                     energy,
                     entropy,
                     homog,
                     invdiff,
                     maxp])