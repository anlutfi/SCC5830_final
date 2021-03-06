# Morphology and Haralick tools from assignment 5

import numpy as np
import matplotlib.pyplot as plt



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


def show(img, plt_size=(16, 10), cmap=None, title=None):
    """displays an image in jupyter"""
    fig = plt.figure(figsize=plt_size)
    if title is not None:
        plt.title(title)
    plt.imshow(img, cmap=(cmap
                          if cmap is not None
                          else (None if len(img.shape) == 3 else 'gray')))