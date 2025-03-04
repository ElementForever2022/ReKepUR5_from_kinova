import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(
        # file='/home/jiachengxu/Desktop/workspace/robots/urdf_files_dataset/urdf_files/robotics-toolbox/abb_irb140/urdf/irb140.urdf',
        file='/home/cyt/genesis/ur_description/urdf/ur5.urdf',
        fixed=True,
    ),
    # gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()

for i in range(1000):
    scene.step()