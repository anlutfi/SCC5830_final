import laspy
import numpy as np
from os import listdir, path
from morph_texture import normalize

X, Y, Z, R, G, B = 0, 1, 2, 3, 4, 5

def process_point_cloud(point_cloud_dir,
                        output_resolution,
                        output_range=(0, 255),
                        depth_map_type=np.uint8,
                        generate_rgb=False):
    """Loads and processes LAS/LAZ files in a given directory into
    a depth map. if :param generate_rgb: is True, it also
    returns an RGB image made from the projected points.
    This RGB image will have pixels set to (-1, -1, -1) if no points
    were associated with its particular region."""    
    points = None
    for fname in listdir(point_cloud_dir):
        pc = laspy.read(path.sep.join([point_cloud_dir, fname]))
        if generate_rgb:
            point_data = np.stack([pc.X,
                                   pc.Y,
                                   pc.Z,
                                   pc.red,
                                   pc.green,
                                   pc.blue], axis=0).transpose((1, 0))
        else:
            point_data = np.stack([pc.X,
                                   pc.Y,
                                   pc.Z], axis=0).transpose((1, 0))
        
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

    if generate_rgb:
        points[:, R] = normalize(points[:, R], type=points.dtype)
        points[:, G] = normalize(points[:, G], type=points.dtype)
        points[:, B] = normalize(points[:, B], type=points.dtype)
        rgb = [[[] for w in range(W)] for h in range(H)]

    depth = [[[] for w in range(W)] for h in range(H)]
    
    # TODO these fors desperately need improvement
    for point in points:
        depth[int(point[Y])][int(point[X])].append(point[Z])
        if generate_rgb:
            rgb[int(point[Y])][int(point[X])].append((point[R], point[G], point[B]))
    for h in range(H):
        for w in range(W):
            depth[h][w] = np.mean(depth[h][w]) if len(depth[h][w]) > 0 else 0
            if generate_rgb:
                rgb[h][w] = np.mean(rgb[h][w], axis=0) if len(rgb[h][w]) > 0 else (-1 , -1, -1)
    
    depth = normalize(np.array(depth), output_range, depth_map_type)
    
    if generate_rgb:
        rgb = np.array(rgb)
        idxs = np.where(np.all(rgb == (-1, -1, -1), axis=-1))
        rgb[idxs] = (0, 0, 0)
        rgb = normalize(np.array(rgb), output_range, np.float)
        rgb[idxs] = (-1, -1, -1)
    
    return (np.flip(depth, axis=0),
            np.flip(rgb, axis=0) if generate_rgb else None)
    
    
    