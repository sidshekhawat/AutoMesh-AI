import cv2
import numpy as np
import os


DEPTH_MAP_PATH = "outputs/depth/depth_map.png"


def load_depth_map(path):
    """
    Load a grayscale depth map.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth map not found: {path}")

    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if depth is None:
        raise ValueError("Failed to load depth map.")

    return depth


def inspect_depth_map(depth):
    """
    Print useful information about the depth map.
    """

    print("\n===== Depth Map Information =====")

    print(f"Shape      : {depth.shape}")
    print(f"Data Type  : {depth.dtype}")
    print(f"Min Value  : {depth.min()}")
    print(f"Max Value  : {depth.max()}")

    print("=================================\n")

def depth_to_pointcloud(depth):
    """
    Convert a depth image into a list of 3D points.
    """

    height, width = depth.shape

    # Camera intrinsics (temporary approximation)
    fx = width
    fy = width

    cx = width / 2
    cy = height / 2

    points = []

    for v in range(height):
        for u in range(width):

            z = float(depth[v, u]) / 255.0

            # Ignore invalid points
            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])

    return np.array(points, dtype=np.float32)

def main():

    depth = load_depth_map(DEPTH_MAP_PATH)

    inspect_depth_map(depth)

    points = depth_to_pointcloud(depth)

    print("Point Cloud Shape :", points.shape)
    print("First Five Points:\n")
    print(points[:5])


if __name__ == "__main__":
    main()