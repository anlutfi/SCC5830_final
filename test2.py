import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageEnhance
import mahotas as mt


def get_descriptors(img):
    result = mt.features.haralick(img)
    return np.mean(result, axis=0)


def apply_mask(img, mask):
    result = np.zeros_like(img)
    idxs = np.where(mask == 1)
    result[idxs] = img[idxs]
    return result