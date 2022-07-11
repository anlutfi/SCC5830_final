import laspy
import numpy as np
from os import listdir, path
from morph_texture import normalize

def get_depth_map(point_cloud_dir,
                  output_resolution,
                  output_range=(0, 255),
                  output_type=np.uint8):
    points = None
    for fname in listdir(point_cloud_dir):
        pc = laspy.read(path.sep.join([point_cloud_dir, fname]))
        point_data = np.stack([pc.X, pc.Y, pc.Z], axis=0).transpose((1, 0))
        if points is None:
            points = point_data
        else:
            points = np.concatenate([points, point_data])
    
    points = points.astype(np.float32)
    
    H, W = output_resolution
    
    points[:, 0] = normalize(points[:, 0], (0, W - 1), points.dtype)
    points[:, 0] = np.floor(points[:, 0])

    points[:, 1] = normalize(points[:, 1], (0, H - 1), points.dtype)
    points[:, 1] = np.floor(points[:, 1])

    points[:, 2] = normalize(points[:, 2], type=points.dtype)

    result = [[[] for w in range(W)] for h in range(H)]
    for point in points:
        result[int(point[1])][int(point[0])].append(point[2])
    for h in range(H):
        for w in range(W):
            result[h][w] = np.mean(result[h][w]) if len(result[h][w]) > 0 else 0
    
    result = normalize(np.array(result), output_range, output_type)
    return np.flip(result, axis=0)
    
    
    