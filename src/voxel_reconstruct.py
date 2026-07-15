import cv2
import os


BLUEPRINT_PATH = "data/blueprints/car_blueprint.jpeg"


def load_blueprint(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image

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

        band_name = band_path.split("/")[-1].replace(".png", "")

        filename = (
            f"outputs/views/"
            f"{band_name}_view_{i}.png"
        )

        cv2.imwrite(filename, crop)

        print(f"Saved {filename}")

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


if __name__ == "__main__":
    main()

    for i in range(1, 5):
        analyze_band(
            f"outputs/bands/band_{i}.png"
        )


