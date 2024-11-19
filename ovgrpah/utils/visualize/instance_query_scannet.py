import sys
import glob
import importlib
import copy
import pickle
import argparse
from functools import partial
from pathlib import Path

import git
import numpy as np
import open3d as o3d
# import clip
import torch
import torch.nn.functional as F
from scipy.spatial import distance


def vis_pcd(pcd, cam_pose=None, coord_frame_size=0.2):
    if not isinstance(pcd, list):
        pcd = [pcd]
    pcd_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
    if cam_pose is not None:
        cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
        cam_frame.transform(cam_pose)
        o3d.visualization.draw_geometries([*pcd, pcd_frame, cam_frame])
    else:
        o3d.visualization.draw_geometries([*pcd, pcd_frame])


def visualize_scene_graph(scene_pcd, scene_graph, show_center=False):
    geometries = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = (0, 0, 0)
    colors = np.random.random((len(scene_graph.nodes), 3))

    # naive visualization (sort by segment size and visualize large instances first)
    node_ids = list(scene_graph.nodes)
    print(f"{len(node_ids) = }")
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    for node_id in node_ids:
        node = scene_graph.nodes[node_id]
        pt_indices = node['pt_indices']
        vis_colors[pt_indices] = colors[node_id]
        if show_center:
            mesh_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color((1, 0, 0))
            mesh_sphere.translate(node['center'])
            geometries.append(mesh_sphere)
    geometries.append(pcd_vis)
    vis_pcd(geometries)


def visualize_instances(scene_pcd, scene_graph):
    print("Press J to show next instance, K to show previous instance")
    pcd_vis = copy.deepcopy(scene_pcd)
    ptr = [0]

    def show_next_instance_callback(visualizer, action, mod, pressed_key):
        if action == 0:  # key is pressed
            if pressed_key == 'J':
                ptr[0] += 1
            elif pressed_key == 'K':
                ptr[0] -= 1
        elif action == 2:
            if pressed_key == 'J':
                ptr[0] += 1
            elif pressed_key == 'K':
                ptr[0] -= 1
        instance_id = ptr[0] % len(scene_graph.nodes)

        print(f"instance: {instance_id}")
        colors = np.asarray(pcd_vis.colors)
        colors[:] = np.asarray(scene_pcd.colors)
        pt_indices = scene_graph.nodes[instance_id]['pt_indices']
        colors[pt_indices] = (1, 1, 0)

        return True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    for key in ['J', 'K']:
        vis.register_key_action_callback(ord(key), partial(show_next_instance_callback, pressed_key=key))
    vis.create_window()
    vis.add_geometry(pcd_vis)
    vis.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="./results")
    parser.add_argument("-v", "--video", type=str, default="scene0011_00")
    parser.add_argument("--prediction_file", default="proposed_fusion_cropformer-none-0.5-ovseg_tap_interval-100.pkl")
    parser.add_argument("--vocabs", default="lvis")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Using {args.vocabs} vocabulary.")
    dataset = Path(args.dataset).expanduser()
    video_path = dataset / args.video
    scene_pcd_path = video_path / f"{args.video}_vh_clean_2.ply"
    print(f"{str(scene_pcd_path) = }")
    scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))

    prediction_path = video_path / args.prediction_file
    with open(prediction_path, 'rb') as fp:
        scene_graph = pickle.load(fp)

    # print("Visualizing scene point cloud")
    # vis_pcd(scene_pcd)

    print(f'Visualizing all instances in {args.video}')
    visualize_scene_graph(scene_pcd, scene_graph, show_center=False)

    print(f"Visualizing instances in {args.video} one by one")
    visualize_instances(scene_pcd, scene_graph)


if __name__ == "__main__":
    main()
