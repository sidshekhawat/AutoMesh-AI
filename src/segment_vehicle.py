# src/segment_vehicle.py

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8x-seg.pt")

image_path = "data/photos/car.jpg"

results = model(image_path)

image = cv2.imread(image_path)

for result in results:

    if result.masks is None:
        continue

    for mask in result.masks.data:

        mask = mask.cpu().numpy()

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0])
        )

        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        cutout = cv2.bitwise_and(
            image,
            image,
            mask=binary_mask
        )

        cv2.imwrite(
            "outputs/masks/vehicle_mask.png",
            binary_mask
        )

        cv2.imwrite(
            "outputs/cutouts/vehicle_cutout.png",
            cutout
        )

        print("Vehicle extracted")
        break