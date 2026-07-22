import copy
import open3d as o3d
import numpy as np



def load_pointcloud(path):
    """
    Load a point cloud from a PLY file.

    Args:
        path (str): Path to the point cloud.

    Returns:
        open3d.geometry.PointCloud
    """

    pointcloud = o3d.io.read_point_cloud(path)

    return pointcloud

def save_pointcloud(pointcloud, output_path):
    """
    Save a point cloud to a PLY file.

    Args:
        pointcloud (open3d.geometry.PointCloud)
        output_path (str)
    """

    o3d.io.write_point_cloud(
        output_path,
        pointcloud
    )

def rotate_pointcloud(
    pointcloud,
    angle,
    axis="y"
):
    """
    Rotate a point cloud using NumPy.

    Args:
        pointcloud (PointCloud)
        angle (float): Degrees
        axis (str): x, y or z

    Returns:
        PointCloud
    """

    angle = np.radians(angle)

    if axis == "x":

        rotation = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])

    elif axis == "y":

        rotation = np.array([
            [ np.cos(angle), 0, np.sin(angle)],
            [0,              1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    elif axis == "z":

        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])

    else:
        raise ValueError("Axis must be x, y or z.")

    points = np.array(pointcloud.points, dtype=np.float64, copy=True)

    rotated_points = np.einsum("ij,kj->ki", rotation, points)

    rotated_cloud = o3d.geometry.PointCloud()

    rotated_cloud.points = o3d.utility.Vector3dVector(
        rotated_points
    )

    if pointcloud.has_colors():
        rotated_cloud.colors = pointcloud.colors

    if pointcloud.has_normals():
        rotated_cloud.normals = pointcloud.normals

    return rotated_cloud

def translate_pointcloud(
    pointcloud,
    tx,
    ty,
    tz
):
    """
    Translate a point cloud.

    Args:
        pointcloud (PointCloud)
        tx (float)
        ty (float)
        tz (float)

    Returns:
        PointCloud
    """
    points = np.array(
        pointcloud.points,
        dtype=np.float64,
        copy=True
    )

    translation = np.array([
        tx,
        ty,
        tz
    ], dtype=np.float64)

    translated_points = points + translation

    translated_cloud = o3d.geometry.PointCloud()

    translated_cloud.points = o3d.utility.Vector3dVector(
        translated_points
    )

    if pointcloud.has_colors():
        translated_cloud.colors = pointcloud.colors

    if pointcloud.has_normals():
        translated_cloud.normals = pointcloud.normals

    return translated_cloud


def main():

    pointcloud = load_pointcloud(
        "outputs/pointcloud/front.ply"
    )

    translated = translate_pointcloud(
        pointcloud,
        tx=1.0,
        ty=0.0,
        tz=0.0
    )

    save_pointcloud(
        translated,
        "outputs/registration/front_translated.ply"
    )

    print("Translation complete.")


if __name__ == "__main__":
    main()

