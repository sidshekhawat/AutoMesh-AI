import os

REQUIRED_VIEWS = [
    "front",
    "rear",
    "left",
    "right"
]

def load_vehicle_dataset(dataset_path):
    """
    Load a multi-view vehicle dataset.

    Args:
        dataset_path (str)

    Returns:
        dict
    """

    views = {}

    for view in REQUIRED_VIEWS:

        image_path = os.path.join(
            dataset_path,
            f"{view}.jpg"
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Missing required view: {view}.jpg"
            )
        
        views[view] = image_path

    return views

def main():

    dataset = load_vehicle_dataset(
        "data/photos/vehicle_01"
    )

    print(dataset)


if __name__ == "__main__":
    main()