import cv2
import numpy as np
import os


DEPTH_MAP_PATH = "outputs/depth/depth_map.png"
MASK_PATH = "outputs/masks/vehicle_mask.png"

POINTCLOUD_OUTPUT = "outputs/pointcloud"
PLY_OUTPUT = os.path.join(
    POINTCLOUD_OUTPUT,
    "vehicle_pointcloud.ply"
)

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

def normalize_depth(depth):
    """
    Normalize depth values to the range [0, 1].
    """

    depth = depth.astype(np.float32)

    depth /= 255.0

    return depth

def apply_vehicle_mask(depth, mask):
    """
    Keep only vehicle pixels in the depth map.
    Background pixels become zero.
    """

    if depth.shape != mask.shape:
        raise ValueError(
            "Depth map and mask dimensions do not match."
        )

    masked_depth = depth.copy()

    masked_depth[mask == 0] = 0

    return masked_depth

def load_vehicle_mask(path):
    """
    Load the binary vehicle mask.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mask not found: {path}"
        )

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            "Failed to load vehicle mask."
        )

    return mask

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

            z = float(depth[v, u])

            # Ignore invalid points
            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])

    return np.array(points, dtype=np.float32)

def save_pointcloud(points, output_path):
    """
    Save the point cloud as an ASCII PLY file.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(output_path, "w") as file:

        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")

        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        file.write("end_header\n")

        for point in points:
            file.write(
                f"{point[0]} {point[1]} {point[2]}\n"
            )

def generate_pointcloud(depth_path, mask_path):
    """
    Generate a point cloud from a depth map and
    vehicle mask.

    Returns:
        np.ndarray
    """

    depth = load_depth_map(depth_path)

    depth = normalize_depth(depth)

    mask = load_vehicle_mask(mask_path)

    depth = apply_vehicle_mask(depth, mask)

    return depth_to_pointcloud(depth)

def main():

    points = generate_pointcloud(
        DEPTH_MAP_PATH,
        MASK_PATH
    )

    print("Point Cloud Shape:", points.shape)
    print(points[:5])

    save_pointcloud(
        points,
        PLY_OUTPUT
    )

    print(f"Point cloud saved to: {PLY_OUTPUT}")


if __name__ == "__main__":
    main()