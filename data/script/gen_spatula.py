import trimesh

path = 'data/spatula/'

handle = trimesh.creation.capsule(radius=0.015, height=0.10)
handle.export(path + 'handle.obj')

pad = trimesh.creation.box(extents=[0.15, 0.05, 0.0005])
pad.export(path + 'pad.obj')
