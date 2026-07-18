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


def main():

    depth = load_depth_map(DEPTH_MAP_PATH)

    inspect_depth_map(depth)


if __name__ == "__main__":
    main()