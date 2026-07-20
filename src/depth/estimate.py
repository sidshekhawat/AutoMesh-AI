from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import os

OUTPUT_PATH = "outputs/depth/depth_map.png"

print("Loading Depth Anything V2...")

depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

def estimate_depth(image_path):
    """
    Generate a normalized depth map.

    Args:
        image_path (str)

    Returns:
        np.ndarray
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    result = depth_pipe(image)

    depth = result["depth"]

    depth_np = np.array(depth)

    depth_np = cv2.normalize(
        depth_np,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return depth_np.astype(np.uint8)

def save_depth_map(
    depth_map,
    output_path=OUTPUT_PATH
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        depth_map
    )
    

def main():

    image_path = "data/photos/car.jpg"

    print("Generating depth map...")

    depth = estimate_depth(image_path)

    output_path = OUTPUT_PATH

    save_depth_map(depth, output_path)

    print(f"Depth map saved to: {output_path}")


if __name__ == "__main__":
    main()