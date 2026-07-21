from src.segmentation.segment import (
    segment_vehicle,
    save_segmentation
)

from src.depth.estimate import (
    estimate_depth,
    save_depth_map
)

from src.pointcloud.generate import (
    generate_pointcloud,
    save_pointcloud,
    PLY_OUTPUT
)

from src.pointcloud.visualize import (
    visualize_pointcloud
)

MASK_OUTPUT = "outputs/masks/vehicle_mask.png"
CUTOUT_OUTPUT = "outputs/cutouts/vehicle_cutout.png"

DEPTH_OUTPUT = "outputs/depth/depth_map.png"

def reconstruct_vehicle(image_path):
    """
    Complete reconstruction pipeline.

    Returns:
        str: Path to generated PLY file.
    """

    # Vehicle Segmentation
    mask, cutout = segment_vehicle(image_path)

    save_segmentation(
        mask,
        cutout
    )

    # Depth Estimation
    depth = estimate_depth(image_path)

    save_depth_map(depth)

    # Point Cloud Generation
    points = generate_pointcloud(
        DEPTH_OUTPUT,
        MASK_OUTPUT
    )

    save_pointcloud(
        points,
        PLY_OUTPUT
    )

    return PLY_OUTPUT

def main():

    image_path = "data/photos/car.jpg"

    ply_path = reconstruct_vehicle(image_path)

    visualize_pointcloud(ply_path)


if __name__ == "__main__":
    main()