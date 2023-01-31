import open3d as o3d
import numpy as np
import h5py


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black



def get_object_nodes(sample_data):
    object_nodes = []
    for idx in range(len(sample_data['object_nodes'])):
        object_node = {}
        node_data = sample_data['object_nodes'][str(idx)]
        for key in node_data.keys():
            if node_data[key].shape is None:
                continue
            object_node[key] = node_data[key][:]
        object_nodes.append(object_node)
    return object_nodes


def mesh_penetration_loss(mesh_path, nearest_k_frames, weight=0.8):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)


    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

    total_loss = 0.0
    
    for frame in nearest_k_frames:
        frame = frame.astype('float32')
        signed_distances = scene.compute_signed_distance(frame).numpy()
        negative_indices = signed_distances > 0
        signed_distances = signed_distances**2
        signed_distances[negative_indices] *= weight
        total_loss += signed_distances.sum()

    return total_loss
# o3d.visualization.draw_geometries_with_key_callbacks([mesh, bb1, bb2, original_bb], key_to_callback)
# sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
# sphere_mesh.translate(mesh.get_center() - [0, 0.05, 0])
# sphere_mesh.paint_uniform_color([1, 0.706, 0])
# o3d.visualization.draw_geometries([mesh, sphere_mesh])

# Create a visualization object and window
# vis = o3d.visualization.Visualizer()
# vis.create_window()


# Display the bounding boxes:
# vis.add_geometry(mesh)
# vis.add_geometry(bb)
# vis.run()
# vis.draw_geometries_with_key_callbacks(
# vis.destroy_window()



mesh_path = '../pointnet_pytorch/data/adl_shapenet/watertight/bed/bed0.obj'
scene_path = './datasets/virtualhome_22_classes/samples/4_2_202_Female2_0.hdf5'
loss = mesh_penetration_loss(mesh_path, scene_path)
print(loss)