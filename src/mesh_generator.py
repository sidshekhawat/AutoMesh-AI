import numpy as np
from skimage import measure

voxels = np.load(
    "outputs/vehicle_voxels.npy"
)

vertices, faces, normals, values = (
    measure.marching_cubes(
        voxels.astype(float),
        level=0.5
    )
)

print(
    f"Vertices: {len(vertices)}"
)

print(
    f"Faces: {len(faces)}"
)

with open(
    "outputs/vehicle.obj",
    "w"
) as f:

    for vertex in vertices:
        f.write(
            f"v {vertex[0]} "
            f"{vertex[1]} "
            f"{vertex[2]}\n"
        )

    for face in faces:
        f.write(
            f"f {face[0]+1} "
            f"{face[1]+1} "
            f"{face[2]+1}\n"
        )

print(
    "OBJ file saved."
)