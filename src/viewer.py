import pyvista as pv

reader = pv.get_reader(
    "outputs/vehicle.obj"
)

mesh = reader.read()

print(mesh)

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show()