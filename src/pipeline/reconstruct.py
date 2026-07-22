import os

from src.dataset.loader import (
    load_vehicle_dataset
)

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
)

from src.pointcloud.visualize import (
    visualize_pointcloud
)

MASK_OUTPUT = "outputs/masks/vehicle_mask.png"
CUTOUT_OUTPUT = "outputs/cutouts/vehicle_cutout.png"

DEPTH_OUTPUT = "outputs/depth/depth_map.png"

def reconstruct_vehicle(
    image_path,
    output_name="vehicle_pointcloud"
):
    """
    Complete reconstruction pipeline.

    Args:
        image_path (str)
        output_name (str)

    Returns:
        str
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

    output_path = os.path.join(
        "outputs",
        "pointcloud",
        f"{output_name}.ply"
    )

    save_pointcloud(
        points,
        output_path
    )
    return output_path

def reconstruct_dataset(dataset):
    """
    Reconstruct every vehicle view contained in
    a multi-view dataset.

    Args:
        dataset (dict)

    Returns:
        dict
    """
    reconstructed_views = {}

    for view, image_path in dataset.items():
        print(f"\nReconstructing {view} view...")

        ply_path = reconstruct_vehicle(
            image_path,
            output_name=view
        )

        reconstructed_views[view] = ply_path

    return reconstructed_views
    
def main():

    dataset = load_vehicle_dataset(
        "data/photos/vehicle_01"
    )

    reconstructed = reconstruct_dataset(dataset)

    print("\n===== Reconstruction Complete =====\n")

    for view, ply_path in reconstructed.items():

        print(f"{view:>5} -> {ply_path}")

if __name__ == "__main__":
    main()