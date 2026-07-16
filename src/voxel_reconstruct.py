import cv2
import os
import numpy as np

BLUEPRINT_PATH = "data/blueprints/car_blueprint.jpeg"


def load_blueprint(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image

def create_silhouette(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    binary = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        mask = np.zeros_like(binary)

        cv2.drawContours(
            mask,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

        binary = mask

    return binary

def create_voxel_grid():

    length = 128
    width = 64
    height = 64

    voxels = np.ones(
        (length, width, height),
        dtype=bool
    )

    return voxels

def load_silhouette(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load {path}"
        )

    return image

def resize_silhouettes(side, front, top):

    side = cv2.resize(
        side,
        (128, 64)
    )

    front = cv2.resize(
        front,
        (64, 64)
    )

    top = cv2.resize(
        top,
        (128, 64)
    )

    return side, front, top

def carve_top_view(voxels, top):

    for x in range(128):

        for y in range(64):

            if top[y, x] == 0:

                voxels[x, y, :] = False

    return voxels

def carve_side_view(voxels, side):

    for x in range(128):

        for z in range(64):

            if side[z, x] == 0:

                voxels[x, :, z] = False

    return voxels

def carve_front_view(voxels, front):

    for y in range(64):

        for z in range(64):

            if front[z, y] == 0:

                voxels[:, y, z] = False

    return voxels

def analyze_band(band_path):

    band_name = os.path.basename(
        band_path
    ).replace(".png", "")

    os.makedirs(
        "outputs/views",
        exist_ok=True
    )

    band = cv2.imread(band_path)

    gray = cv2.cvtColor(
        band,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    column_sums = binary.sum(axis=0)

    cols_with_content = []

    for i, value in enumerate(column_sums):

        if value > 1000:
            cols_with_content.append(i)

    view_ranges = []

    start = cols_with_content[0]
    prev = cols_with_content[0]

    for col in cols_with_content[1:]:

        if col - prev > 10:
            view_ranges.append((start, prev))
            start = col

        prev = col

    view_ranges.append((start, prev))

    print("\nDetected Views:\n")

    for i, (start_col, end_col) in enumerate(view_ranges, start=1):

        print((start_col, end_col))

        crop = band[:, start_col:end_col]

        silhouette = create_silhouette(crop)

        band_name = band_path.split("/")[-1].replace(".png", "")

        view_filename = (
            f"outputs/views/"
            f"{band_name}_view_{i}.png"
        )

        silhouette_filename = (
            f"outputs/views/"
            f"{band_name}_view_{i}_silhouette.png"
        )

        cv2.imwrite(
            view_filename,
            crop
        )

        cv2.imwrite(
            silhouette_filename,
            silhouette
        )

        print(f"Saved {view_filename}")
        print(f"Saved {silhouette_filename}")

    cv2.imshow("Band Binary", binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def main():
    blueprint = load_blueprint(BLUEPRINT_PATH)

    gray = cv2.cvtColor(blueprint, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Count white pixels in each row
    row_sums = binary.sum(axis=1)

    rows_with_content = []

    for i, value in enumerate(row_sums):
        if value > 1000:
            rows_with_content.append(i)

    bands = []

    start = rows_with_content[0]
    prev = rows_with_content[0]

    for row in rows_with_content[1:]:

        if row - prev > 5:
            bands.append((start, prev))
            start = row

        prev = row

    bands.append((start, prev))

    print("\nDetected Bands:\n")

    output = blueprint.copy()

    for start_row, end_row in bands:

        print("Drawing band:", start_row, end_row)

        cv2.rectangle(
            output,
            (0, start_row),
            (blueprint.shape[1] - 1, end_row),
            (0, 0, 255),
            6
        )

    cv2.imwrite("outputs/bands_debug.png", output)
    print("Saved bands image")

    cv2.imshow("Bands", output)
    #cv2.imshow("Binary", binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    import os

    os.makedirs("outputs/bands", exist_ok=True)

    for i, (start_row, end_row) in enumerate(bands, start=1):

        band = blueprint[start_row:end_row, :]

        filename = f"outputs/bands/band_{i}.png"

        cv2.imwrite(filename, band)

        print(f"Saved {filename}")

    side = load_silhouette(
        "outputs/views/band_1_view_1_silhouette.png"
    )

    front = load_silhouette(
        "outputs/views/band_3_view_2_silhouette.png"
    )

    top = load_silhouette(
        "outputs/views/band_4_view_1_silhouette.png"
    )

    side, front, top = resize_silhouettes(
        side,
        front,
        top
    )

    print(
        f"Resized Side: {side.shape}"
    )

    print(
        f"Resized Front: {front.shape}"
    )

    print(
        f"Resized Top: {top.shape}"
    )

    print(
        f"Front Silhouette Shape: {front.shape}"
    )

    print(
        f"Top Silhouette Shape: {top.shape}"
    )

    print(
        f"Side Silhouette Shape: {side.shape}"
    )

    voxels = create_voxel_grid()
    print(
        f"Voxel Grid Shape: {voxels.shape}"
    )

    voxels = carve_top_view(
        voxels,
        top
    )

    voxels = carve_side_view(
        voxels,
        side
    )

    voxels = carve_front_view(
        voxels,
        front
    )

    print(
        f"Remaining Voxels After Front Carving: "
        f"{np.sum(voxels)}"
    )

    np.save(
        "outputs/vehicle_voxels.npy",
        voxels
    )

    print(
        "Voxel volume saved."
    )


if __name__ == "__main__":
    main()

    for i in range(1, 5):
        analyze_band(
            f"outputs/bands/band_{i}.png"
        )


