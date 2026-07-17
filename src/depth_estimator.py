from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import os

# Create output folder
os.makedirs("outputs/depth", exist_ok=True)

# Load image
image_path = "data/photos/car.jpg"
image = Image.open(image_path).convert("RGB")

print("Loading Depth Anything V2...")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("Generating depth map...")
result = depth_pipe(image)

depth = result["depth"]

# Convert to numpy
depth_np = np.array(depth)

# Normalize to 0-255
depth_np = cv2.normalize(
    depth_np,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_np = depth_np.astype(np.uint8)

# Save result
output_path = "outputs/depth/depth_map.png"

cv2.imwrite(output_path, depth_np)

print(f"Depth map saved to: {output_path}")