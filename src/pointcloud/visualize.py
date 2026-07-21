import pyvista as pv
import os

PLY_PATH = "outputs/pointcloud/vehicle_pointcloud.ply"


def load_pointcloud(path):
    """
    Load a PLY point cloud.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Point cloud not found: {path}"
        )

    reader = pv.get_reader(path)

    return reader.read()


def visualize_pointcloud(ply_path):
    """
    Visualize a PLY point cloud.
    """

    pointcloud = load_pointcloud(ply_path)

    print(pointcloud)

    plotter = pv.Plotter()

    plotter.add_mesh(
        pointcloud,
        point_size=3,
        render_points_as_spheres=True
    )

    plotter.add_axes()

    plotter.show_grid()

    plotter.show()


def main():

    visualize_pointcloud(PLY_PATH)


if __name__ == "__main__":
    main()