import laspy
import numpy as np
from os import listdir, path
from morph_texture import normalize

X, Y, Z, R, G, B = 0, 1, 2, 3, 4, 5

def get_depth_and_rgb(point_cloud_dir,
                      output_resolution,
                      output_range=(0, 255),
                      output_type=np.uint8):
    points = None
    for fname in listdir(point_cloud_dir):
        pc = laspy.read(path.sep.join([point_cloud_dir, fname]))
        point_data = np.stack([pc.X,
                               pc.Y,
                               pc.Z,
                               pc.red,
                               pc.green,
                               pc.blue], axis=0).transpose((1, 0))
        if points is None:
            points = point_data
        else:
            points = np.concatenate([points, point_data])
    
    points = points.astype(np.float32)
    
    H, W = output_resolution
    
    points[:, X] = normalize(points[:, X], (0, W - 1), points.dtype)
    points[:, X] = np.floor(points[:, X])

    points[:, Y] = normalize(points[:, Y], (0, H - 1), points.dtype)
    points[:, Y] = np.floor(points[:, Y])

    points[:, Z] = normalize(points[:, Z], type=points.dtype)

    points[:, R] = normalize(points[:, R], type=points.dtype)
    points[:, G] = normalize(points[:, G], type=points.dtype)
    points[:, B] = normalize(points[:, B], type=points.dtype)

    depth = [[[] for w in range(W)] for h in range(H)]
    rgb = [[[] for w in range(W)] for h in range(H)]
    # TODO these fors need desperate improvement
    for point in points:
        depth[int(point[Y])][int(point[X])].append(point[Z])
        rgb[int(point[Y])][int(point[X])].append((point[R], point[G], point[B]))
    for h in range(H):
        for w in range(W):
            depth[h][w] = np.mean(depth[h][w]) if len(depth[h][w]) > 0 else 0
            rgb[h][w] = np.mean(rgb[h][w], axis=0) if len(rgb[h][w]) > 0 else (-1 , -1, -1)
    
    depth = normalize(np.array(depth), output_range, output_type)
    rgb = normalize(np.array(rgb), output_range, np.int32)
    
    return np.flip(depth, axis=0), np.flip(rgb, axis=0)
    
    
    