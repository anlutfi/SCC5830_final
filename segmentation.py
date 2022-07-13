import cv2
import numpy as np
from skimage.morphology import disk
from skimage.segmentation import watershed


def segment_by_depth(img,
                     depth_map,
                     morph_operation=cv2.MORPH_DILATE,
                     structuring_el=disk(15),
                     return_intermediate=False):
    """Segment an image :param img: unsing watershed segmentation.
    For its initialization, :param depth_map: undergoes a morphology operation given by :param morph_operator: using 
    :param structuring_el: as structuring element. The resulting image's local maxima are set as region seeds.
    
    if :param_return intermediate: is set to True, all intermediate images are returned in a dictionary
    """

    morph = cv2.morphologyEx(depth_map, morph_operation, structuring_el)
    maxima_ixs = np.where(morph == depth_map)
    maxima = np.zeros_like(depth_map)
    maxima[maxima_ixs] = 255
    maxima_open = cv2.morphologyEx(maxima,
                                   cv2.MORPH_CLOSE,
                                   disk(5),
                                   iterations=2)
    seeds = cv2.connectedComponents(maxima_open)[1]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    segmented = watershed(np.max(img) - img, seeds)

    if return_intermediate:
        inter = {'morph': morph,
                 'maxima': maxima,
                 'maxima_open': maxima_open,
                 'seeds': seeds,
                 'edges': cv2.Canny(segmented.astype(np.uint8), 0, 0)
                }
        return segmented, inter
    return segmented, None

