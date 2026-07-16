import numpy as np
import matplotlib.pyplot as plt

voxels = np.load(
    "outputs/vehicle_voxels.npy"
)

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.voxels(
    voxels,
    edgecolor="k"
)

plt.show()