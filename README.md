# Individual Tree Identification in Native Forest Images

## Main Objective and Image Source
The main objective of this work is to identify and differentiate individual trees in drone-captured images of native forests. The specific images for this experiment are of a green area within the UFRRJ campus. There is a single stitched image of the whole flight and the tests and processing will be done in segments, such as the one below:
![sample image](test.png)

This application is intended to make forestry upkeep efforts less labor intensive. Today, making inventory of a forest requires frequent visits by foot. If there was a reliable method to observe previously catalogued trees with a drone survey, it would make the job of forest engineers much easier.

Images are provided by Professor [Bruno Mendonça](https://institutos.ufrrj.br/if/bruno-araujo-furtado-de-mendonca/) and Fernando Canto. They were obtained with support from FAPERJ and UFFRJ.

## Metodology
The main processing tasks involved will be image segmentation, to divide the images into individuals. Also, image description will be used to identify the segmented individuals' species. More specifically, the intended steps for the process are detailed below:

1. Get a dataset with many examples of each tree species present in these woods, as each one is a texture. As of now, I did this by eye and with one example per texture. This depends on the availability of the forest engineers involved and is scheduled to happen soon.

2. Compute the centroid of each sepecies' texture by its descriptors. As this domain does not have very strong edges or transitions, enhancements will have to be experimented with in order to differentiate the classes.

3. Segment the image into the textures. As of now, this is done by sliding a window through the main image and assigning it to its closest class. A better method is expected to be applied in the final approach.



## Initial Experiments and Results
As of now, two experiments were conducted. In the first, segmentation is being tried via haralick descriptors of different tree textures. In the second, a mean shift approach is tried to segment individual trees before texture identification.

### First Experiment

#### Overview

Different textures were extracted from a test image(shown a little further). Their haralick descriptors were computed in similar fashion from the fifith course assignment. Then, the input image was divided into 120x120 chunks, and each was assigned to its closest texture according to euclidean distance between descriptors.

The results are far from complete, but even in this crude initial approach, there is some consistency within them.

#### Next Steps

There are varying shades of green within the same species. So, if a classifier is sensible to the hue, it could make mistakes, as it is observed. The next thing to solve is make the descriptors invariant to small changes in green. The first approach to that, of course, is getting a larger dataset for each texture. Experiments with different color spaces also can help.

Other methods of segmentation shall also be tried. As there is no clear background beyond the gaps between trees, common algorithms such as watershed don't seem applicable so far, at least to my knowledge of it. Fractal Dimension(FD) segmentation is also a possible candidate, but there is a chance that the variation within the FD of trees could be to small to be reliable.

#### Code

##### Imports


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
```


```python
from morph_texture import euclidean_distance, get_haralick, get_masks, normalize
```


```python
R, G, B = 0, 1, 2
```

##### Displaying Tools


```python
def show(img, plt_size=(16, 10), cmap=None):
    fig = plt.figure(figsize=plt_size)
    plt.imshow(img, cmap=cmap if cmap is not None else (None if len(img.shape) == 3 else 'gray'))
```

##### Loading Test Image and Textures


```python
img = imageio.imread('./test_img.png')
textures = [imageio.imread('../texons/{}.png'.format(i)) for i in range(1, 9)]
```


```python
fig = plt.figure(figsize=(32, 20))
for i in range(len(textures)):
    plt.subplot(2, 4, i + 1); plt.imshow(textures[i])
```


    
![png](test_files/test_16_0.png)
    



```python
show(img)
```


    
![png](test_files/test_17_0.png)
    


##### Get Haralick Descriptors for Textures to use as Clusters' Centroids


```python
def morph_f(img):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
```


```python
descriptors = []
for texture in textures:
    descriptors.append(np.concatenate([get_haralick(m, (0, 1)) for m in get_masks(texture, morph_f)]))
```


```python
descriptors[0]
```




    array([1.69474708e+03, 8.32185397e+02, 1.52495475e+01, 2.11911591e-01,
           7.36031198e+00, 4.91814577e-01, 5.22124481e-01, 4.60070563e-01,
           6.20761954e+03, 1.42260825e+03, 1.89394079e+01, 1.77622211e-01,
           7.62654529e+00, 4.53159947e-01, 4.86216098e-01, 4.21168891e-01])



##### Segment Image
Run a moving window and classify it accosrding to its closest centroid (texture)


```python
def segment_by_sliding_window(img, window):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.zeros((img.shape[0] - (img.shape[0] % window[0]),
                       img.shape[1] - (img.shape[1] % window[1])))
    for h in range(0, img.shape[0], window[0]):
        for w in range(0, img.shape[1], window[1]):
            delta_h = min(h + window[0], img.shape[0] - 1)
            delta_w = min(w + window[1], img.shape[1] - 1)
            sub_img = img[h : delta_h, w : delta_w]
            sub_img_descr = np.concatenate([get_haralick(m, (0, 1)) for m in get_masks(sub_img, morph_f)])
            distances = [euclidean_distance(sub_img_descr, d) for d in descriptors]
            color = distances.index(min(distances)) * 1 / len(descriptors)
            result[h : delta_h, w : delta_w] = color
    return result
```


```python
segmented = segment_by_sliding_window(img, (120, 120))
```


```python
show(segmented)
```


    
![png](test_files/test_25_0.png)
    


##### Show Each Segment Separately
It's far from impressive, but there is some consistency. By sampling more of the textures and getting a more general representation, results could improve. Other segmentation algorithms will be tried as well.


```python
clusters = np.unique(segmented)
masked = []
for cluster in clusters:
    mask = np.zeros_like(img)
    ids = np.where(segmented == cluster)
    mask[ids] = img[ids]
    masked.append(mask)
    
```


```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[0])
```




    <matplotlib.image.AxesImage at 0x7f3d0b75e7c0>




    
![png](test_files/test_28_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[1])
```




    <matplotlib.image.AxesImage at 0x7f3d0bfccd30>




    
![png](test_files/test_29_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[2])
```




    <matplotlib.image.AxesImage at 0x7f3d0c0424c0>




    
![png](test_files/test_30_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[3])
```




    <matplotlib.image.AxesImage at 0x7f3d0b7b0940>




    
![png](test_files/test_31_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[4])
```




    <matplotlib.image.AxesImage at 0x7f3d0b6db280>




    
![png](test_files/test_32_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[5])
```




    <matplotlib.image.AxesImage at 0x7f3d0c0f03d0>




    
![png](test_files/test_33_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[6])
```




    <matplotlib.image.AxesImage at 0x7f3d0d999f10>




    
![png](test_files/test_34_1.png)
    



```python
fig = plt.figure(figsize=(32, 20))
plt.imshow(masked[7])
```




    <matplotlib.image.AxesImage at 0x7f3d0da66a90>




    
![png](test_files/test_35_1.png)
    


### Second experiment

#### Overview

Edge detection in this domain is very hard, so a more convoluted method was devised and tried. Firstly, we binarize the image using Otsu thresholding. Then we erode the binary image to exaggerate the gaps between trees. We then use this eroded binary as a mask over the grayscale version of the original image.

The idea is to use this masked image to perform mean shift segmentation. Since the top of the trees are more illuminated than the rest, we can use the pixel values as probabilities for the mean shift. The results were surprisingly good for a first approach.

#### Next Steps

Since I am not familiar with mean shift, there are some theoretical studies that could benefit the experiment. Performance is also a concern.

#### Code

##### Imports


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
```


```python
H, W, CH = 0, 1, 2
```


```python
from morph_texture import normalize
```

##### Image Loading


```python
img = imageio.imread('./test_img.png')
show(img)
```


    
![png](test_files/test_45_0.png)
    


##### Get grayscale and binarize


```python
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, bin_img = cv2.threshold(gray_img, 0, 1, cv2.THRESH_OTSU)
show(bin_img)
```


    
![png](test_files/test_47_0.png)
    


##### Erode binary image and use as mask over grayscale original


```python
tst = bin_img
for i in range(2):
    tst = cv2.morphologyEx(tst, cv2.MORPH_ERODE, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    #tst = cv2.morphologyEx(tst, cv2.MORPH_CLOSE, np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))

probs = np.zeros_like(gray_img)
idxs = np.where(tst == 1)
probs[idxs] = gray_img[idxs]

show(probs)


```


    
![png](test_files/test_49_0.png)
    


##### Segmentation function

Since it's impossible to run it for every pixel, the function uses a sample rate, and assigns surrounding pixels to the same cluster.
I am also ignoring borders for now.


```python
def segment_by_meanshft(img, smpl_rate, window_sz, criterion):
    result = np.zeros((img.shape[0], img.shape[1], 3))
    for h in range(window_sz[H] // 2, probs.shape[H] - (window_sz[H] // 2), smpl_rate[H]):
        for w in range(window_sz[W] // 2, probs.shape[W] - (window_sz[W] // 2), smpl_rate[W]):
            window = (w - window_sz[W] // 2, h - window_sz[H] // 2, window_sz[W], window_sz[H])
            _, window = cv2.meanShift(probs, window, criterion)
            center = window[H] + window[2] // 2, window[W] + window[3] // 2
            color = (center[W] / probs.shape[H], 0.25, center[H] / probs.shape[W])
            result[h - smpl_rate[H] // 2: h + smpl_rate[H] // 2,
                   w - smpl_rate[W] // 2: w + smpl_rate[W] // 2] = color
    
    return normalize(result, (0, 255), np.uint8)
```

##### Set parameters and run


```python
smpl_rate = (100, 100)
window_sz = (400, 400)
criterion = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
```


```python
segmented = segment_by_meanshft(probs, smpl_rate, window_sz, criterion)
```

##### Very similar clusters need to be joined, so requantising is necessary


```python
def re_quantise(img, bin_mask=0b11100000):
    return img & bin_mask
```


```python
seg2 = re_quantise(segmented)
```

##### Visualization Technique 1


```python
edges = cv2.Canny(seg2, 0, 200)
edges = cv2.morphologyEx(edges,
                         cv2.MORPH_DILATE,
                         np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                         iterations=3)
idxs = np.where(edges == 255)
seg2[idxs] = (255, 255, 255)
show(seg2)
```


    
![png](test_files/test_59_0.png)
    



```python
alpha = 0.4
show(cv2.addWeighted(seg2, alpha, img, 1 - alpha, 0))
```


    
![png](test_files/test_60_0.png)
    


##### Visualization Technique 2


```python
clusters = np.unique(seg2.reshape(-1, seg2.shape[2]), axis=0)
```


```python
def visualize_cluster(segmented, cluster, img):
    idxs = np.where(np.all(segmented == cluster, axis=-1))
    img = normalize(img)
    result = img * 0.5
    result[idxs] = img[idxs]
    result = normalize(result, (0, 255), np.uint8)
    show(result)
```


```python
show_cluster = lambda i: visualize_cluster(seg2, clusters[i], img)
```


```python
show_cluster(40)
```


    
![png](test_files/test_65_0.png)
    



```python
show_cluster(16)
```


    
![png](test_files/test_66_0.png)
    

