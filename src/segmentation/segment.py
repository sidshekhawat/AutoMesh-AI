# src/segmentation/segment.py

from ultralytics import YOLO
import cv2
import numpy as np
import os

MODEL_PATH = "yolov8x-seg.pt"

MASK_OUTPUT = "outputs/masks/vehicle_mask.png"
CUTOUT_OUTPUT = "outputs/cutouts/vehicle_cutout.png"

model = YOLO(MODEL_PATH)


def segment_vehicle(image_path):
    """
    Segment a vehicle from an image.

    Args:
        image_path (str)

    Returns:
        tuple:
            (binary_mask, cutout)
    """

    results = model(image_path)

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    for result in results:

        if result.masks is None:
            continue

        for mask in result.masks.data:

            mask = mask.cpu().numpy()

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0])
            )

            binary_mask = (
                (mask > 0.5)
                .astype(np.uint8)
                * 255
            )

            cutout = cv2.bitwise_and(
                image,
                image,
                mask=binary_mask
            )

            return binary_mask, cutout

    return None, None


def save_segmentation(binary_mask, cutout):
    """
    Save segmentation outputs.
    """

    os.makedirs(
        os.path.dirname(MASK_OUTPUT),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(CUTOUT_OUTPUT),
        exist_ok=True
    )

    cv2.imwrite(
        MASK_OUTPUT,
        binary_mask
    )

    cv2.imwrite(
        CUTOUT_OUTPUT,
        cutout
    )


def main():

    image_path = "data/photos/car.jpg"

    binary_mask, cutout = segment_vehicle(image_path)

    if binary_mask is None:
        print("No vehicle detected.")
        return

    save_segmentation(
        binary_mask,
        cutout
    )

    print("Vehicle extracted.")


if __name__ == "__main__":
    main()