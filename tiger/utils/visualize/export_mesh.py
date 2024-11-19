import copy
import json
from collections import defaultdict
import pickle
import sys

import pandas as pd
from sklearn.neighbors import BallTree

import plyfile
import torch
from plyfile import *
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import numpy as np
import networkx as nx

from tiger.utils.utils import vis_pcd
from datasets.constants.replica.replica_constants import REPLICA_CLASSES, REPLICA_CLASSES_VALIDATION, REPLICA_COLOR_MAP_101
from datasets.constants.scannet.scannet200_constants import VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_200, CLASS_LABELS_200, SCANNET_COLOR_MAP_200


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


def visualize_scene_graph_replica_gt(file_path, semantic_info_path):
    # Read PLY file
    plydata = plyfile.PlyData.read(file_path)

    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)
    object_class_mapping = {obj["id"]: obj["class_id"] for obj in semantic_info["objects"]}
    # all instance with annotated class in the current scene
    unique_class_ids = np.unique(list(object_class_mapping.values()))

    # extract vertex data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    # extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]

    # 1. instance point cloud; not all annotated;
    object_ids1 = np.zeros(vertices.shape[0], dtype=np.int32)
    for i, face in enumerate(face_vertices):
        object_ids1[face] = object_ids[i]

    # 2. semantic point cloud;
    class_ids = np.zeros(vertices.shape[0], dtype=np.int32)
    for i, face in enumerate(face_vertices):
        class_ids[face] = object_class_mapping[object_ids[i]] if object_ids[i] in object_class_mapping else 0

    # 1. instance colors
    # unique_object_ids = np.unique(object_ids)
    # instance_colors = np.zeros((len(object_ids1), 3))
    # unique_colors = np.random.rand(len(unique_object_ids), 3)
    # for i, object_id in enumerate(unique_object_ids):
    #     instance_colors[object_ids1 == object_id] = unique_colors[i]

    # 2. semantic colors
    with open(f"class_id_colors.json", 'r') as f:
        unique_class_colors = json.load(f)
    class_colors = np.zeros((len(object_ids1), 3), dtype=np.uint8)
    for i, class_id in enumerate(unique_class_ids):
        class_colors[class_ids == class_id] = (np.array(unique_class_colors[f"{class_id}"]) * 255).astype(np.uint8)

    # write plyfile
    points = [(
        plydata.elements[0].data[i][0],
        plydata.elements[0].data[i][1],
        plydata.elements[0].data[i][2],
        plydata.elements[0].data[i][3],
        plydata.elements[0].data[i][4],
        plydata.elements[0].data[i][5],
        class_colors[i][0],
        class_colors[i][1],
        class_colors[i][2],
    ) for i in range(plydata.elements[0].data.shape[0])]
    vertex = np.array(points, dtype=[('x', 'float'), ('y', 'float'), ('z', 'float'), ('nx', 'float'), ('ny', 'float'), ('nz', 'float'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    vertex_el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([vertex_el, plydata.elements[1]]).write(f'results/{scene}/{scene}_gt_vis.ply')


def export_mesh_scannet(name, v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)


def visualize_scene_graph_scannet(scene_pcd, scene_mesh, scene_graph, scene, show_center=False):
    geometries = []
    segment_position_xy = []
    segment_idx = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = vis_colors[:] * 0.6
    colors = np.random.random((len(scene_graph.nodes), 3))
    # colors = np.array([np.array(c) for c in SCANNET_COLOR_MAP_20.values()]) / 255.0

    # naive visualization (sort by segment size and visualize large instances first)
    node_ids = list(scene_graph.nodes)
    print(f"{len(node_ids) = }")
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    for idx, node_id in enumerate(node_ids):
        if node_id in [88, 43]:
            continue
        node = scene_graph.nodes[node_id]
        pt_indices = node['pt_indices']

        # vis_colors[pt_indices] = 0.0
        vis_colors[pt_indices] = colors[node_id] * 0.95
        # vis_colors[pt_indices] = colors[CLASS_LABELS_200.index(node['top1_cls'])] * 0.95

        if show_center:
            mesh_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color((1, 0, 0))
            mesh_sphere.translate(node['center'])
            geometries.append(mesh_sphere)

        segment_idx.append(idx)
        segment_position_xy.append(np.asarray(pcd_vis.points)[pt_indices].mean(0))

    export_mesh_scannet(f'results/{scene}/{scene}_vis.ply', np.array(scene_mesh.vertices), np.array(scene_mesh.triangles), vis_colors)
    geometries.append(pcd_vis)
    vis_pcd(geometries)


def visualize_scene_graph_replica(scene_pcd, scene_graph, scene, show_center=False):
    geometries = []
    segment_position_xy = []
    segment_idx = []

    faces_el = scene_pcd.elements[1]

    # naive visualization (sort by segment size and visualize large instances first)
    vis_colors = np.stack([scene_pcd["vertex"]["red"], scene_pcd["vertex"]["green"], scene_pcd["vertex"]["blue"]]).T
    vis_colors[:] = vis_colors[:] * 0.5
    colors = (np.random.random((len(scene_graph.nodes), 3)) * 255).astype(np.uint8)

    node_ids = list(scene_graph.nodes)
    print(f"{len(node_ids) = }")
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    for idx, node_id in enumerate(node_ids):
        node = scene_graph.nodes[node_id]
        pt_indices = node['pt_indices']
        vis_colors[pt_indices] = colors[node_id] * 0.9

    # write plyfile
    vis_colors = vis_colors.astype(np.uint8)
    points = [(
        scene_pcd.elements[0].data[i][0],
        scene_pcd.elements[0].data[i][1],
        scene_pcd.elements[0].data[i][2],
        scene_pcd.elements[0].data[i][3],
        scene_pcd.elements[0].data[i][4],
        scene_pcd.elements[0].data[i][5],
        # scene_pcd.elements[0].data[i][6],
        # scene_pcd.elements[0].data[i][7],
        # scene_pcd.elements[0].data[i][8],
        vis_colors[i][0],
        vis_colors[i][1],
        vis_colors[i][2],
    ) for i in range(scene_pcd.elements[0].data.shape[0])]
    vertex = np.array(points, dtype=[('x', 'float'), ('y', 'float'), ('z', 'float'), ('nx', 'float'), ('ny', 'float'), ('nz', 'float'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    vertex_el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([vertex_el, faces_el]).write(f'vis/{scene}_vis.ply')

def knn_interpolation(cumulated_pc: np.ndarray, full_sized_data: np.ndarray, k):
    """
    Using k-nn interpolation to find labels of points of the full sized pointcloud
    :param cumulated_pc: cumulated pointcloud results after running the network
    :param full_sized_data: full sized point cloud
    :param k: k for k nearest neighbor interpolation
    :return: pointcloud with predicted labels in last column and ground truth labels in last but one column
    """

    labeled = cumulated_pc[cumulated_pc[:, -1] != -1]
    to_be_predicted = full_sized_data.copy()

    ball_tree = BallTree(labeled[:, :3], metric="minkowski")

    knn_classes = labeled[ball_tree.query(to_be_predicted[:, :3], k=k)[1]][:, :, -1].astype(int)
    print("knn_classes: ", knn_classes.shape)

    interpolated = np.zeros(knn_classes.shape[0])

    for i in range(knn_classes.shape[0]):
        interpolated[i] = np.bincount(knn_classes[i]).argmax()

    output = np.zeros((to_be_predicted.shape[0], to_be_predicted.shape[1] + 1))
    output[:, :-1] = to_be_predicted

    output[:, -1] = interpolated

    assert output.shape[0] == full_sized_data.shape[0]

    return output

def visualize_scene_graph_replica_fixed(scene_pcd, scene_graph, scene, show_center=False):
    geometries = []
    segment_position_xy = []
    segment_idx = []

    faces_el = scene_pcd.elements[1]

    # naive visualization (sort by segment size and visualize large instances first)
    gt_xyz = np.stack([scene_pcd["vertex"]["x"], scene_pcd["vertex"]["y"], scene_pcd["vertex"]["z"]]).T
    vis_colors = np.stack([scene_pcd["vertex"]["red"], scene_pcd["vertex"]["green"], scene_pcd["vertex"]["blue"]]).T
    vis_colors[:] = vis_colors[:] * 0.5
    with open(f"class_id_colors.json", 'r') as f:
        colors = json.load(f)

    node_ids = list(scene_graph.nodes)
    pred_labels = np.ones([vis_colors.shape[0]], dtype=np.uint8)*-1
    print(f"{len(node_ids) = }")
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    for idx, node_id in enumerate(node_ids):
        node = scene_graph.nodes[node_id]
        pt_indices = node['pt_indices']
        print(node['top1_cls'])
        pred_labels[pt_indices] = REPLICA_CLASSES.index(node['top1_cls'])

    # write plyfile
    # concat coords and labels for predicied pcd
    coords_labels = np.zeros((vis_colors.shape[0], 4))
    coords_labels[:, :3] = gt_xyz
    coords_labels[:, -1] = pred_labels
    # concat coords and labels for gt pcd
    coords_gt = np.zeros((vis_colors.shape[0], 4))
    coords_gt[:, :3] = gt_xyz
    coords_gt[:, -1] = pred_labels
    # knn interpolation
    match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
    pred_labels = match_pc[:, -1].reshape(-1, 1)[:, 0].astype(np.uint8)

    vis_colors = vis_colors.astype(np.uint8)
    uni_cls = np.unique(pred_labels)
    for ii in uni_cls:
        vis_colors[pred_labels==ii] = (np.array(colors[str(ii)]) * 255).astype(np.uint8)

    points = [(
        scene_pcd.elements[0].data[i][0],
        scene_pcd.elements[0].data[i][1],
        scene_pcd.elements[0].data[i][2],
        scene_pcd.elements[0].data[i][3],
        scene_pcd.elements[0].data[i][4],
        scene_pcd.elements[0].data[i][5],
        vis_colors[i][0],
        vis_colors[i][1],
        vis_colors[i][2],
    ) for i in range(scene_pcd.elements[0].data.shape[0])]
    vertex = np.array(points, dtype=[('x', 'float'), ('y', 'float'), ('z', 'float'), ('nx', 'float'), ('ny', 'float'), ('nz', 'float'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    vertex_el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])

    PlyData([vertex_el, faces_el]).write(f'vis/{scene}_vis.ply')


IGNORE_INDEX = -1


def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if len(label_id20) > 0 else 0
    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if len(label_id200) > 0 else 0

    # Only store for the valid categories
    if label_id20 in VALID_CLASS_IDS_20:
        label_id20 = VALID_CLASS_IDS_20.index(label_id20)
    else:
        label_id20 = IGNORE_INDEX

    if label_id200 in VALID_CLASS_IDS_200:
        label_id200 = VALID_CLASS_IDS_200.index(label_id200)
    else:
        label_id200 = IGNORE_INDEX

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_idx = np.isin(seg_indices, group_segments)
    return point_idx, label_id20, label_id200


def get_gt_semantic(seg_groups, seg_indices, labels_pd, gt_xyz, ignore_index=(0,)):
    # parse each instance
    gt_cat_ids, gt_masks = [], []
    semantic_gt200 = np.ones((gt_xyz.shape[0]), dtype=np.int16) * IGNORE_INDEX
    semantic_gt20 = np.ones((gt_xyz.shape[0]), dtype=np.int16) * IGNORE_INDEX
    for group in seg_groups:
        point_idx, label_id20, label_id200 = point_indices_from_group(seg_indices, group, labels_pd)
        semantic_gt200[point_idx] = label_id200
        semantic_gt20[point_idx] = label_id20

        # filter out the ignore class
        if label_id200 not in ignore_index and label_id200 != IGNORE_INDEX:
            gt_cat_ids.append(label_id200)
            gt_masks.append(point_idx)

    # each point's cai_id, [Np, ], the index is already in scannet200 set
    semantic_gt200 = semantic_gt200.astype(int)
    semantic_gt20 = semantic_gt20.astype(int)

    # each instance's id, [Ni, ]
    gt_cat_ids = np.array(gt_cat_ids)
    # each instance idx in point_clouds, [Ni, Np]
    gt_masks = np.array(gt_masks)

    return semantic_gt200, semantic_gt20, gt_cat_ids, gt_masks


def visualize_scene_graph_scannet_gt_in(scene_pcd, scene_mesh, cls_gt, scene, show_center=False):
    geometries = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = vis_colors[:] * 0.6

    colors = np.random.random((cls_gt.shape[0], 3))

    for uuid in range(cls_gt.shape[0]):
        vis_colors[cls_gt[uuid]] = colors[uuid] * 0.9

    export_mesh_scannet(f'results/{scene}/{scene}_gt_vis.ply', np.array(scene_mesh.vertices), np.array(scene_mesh.triangles), vis_colors)
    geometries.append(pcd_vis)
    vis_pcd(geometries)


def visualize_scene_graph_scannet_pt(scene_pcd, scene_mesh, instan_indice, scene, show_center=False):
    geometries = []
    segment_position_xy = []
    segment_idx = []
    pcd_vis = copy.deepcopy(scene_pcd)
    vis_colors = np.asarray(pcd_vis.colors)
    vis_colors[:] = vis_colors[:] * 0.4
    instan_indice = instan_indice.T
    colors = np.random.random((instan_indice.shape[0], 3))
    # colors = np.array([np.array(c) for c in SCANNET_COLOR_MAP_20.values()]) / 255.0

    # naive visualization (sort by segment size and visualize large instances first)
    print(f"{instan_indice.shape[0] = }")
    for idx, pt_indices in enumerate(instan_indice):
        # vis_colors[pt_indices] = 0.0
        vis_colors[pt_indices > 0] = colors[idx]
        # vis_colors[pt_indices] = colors[CLASS_LABELS_200.index(node['top1_cls'])] * 0.95

        segment_idx.append(idx)
        segment_position_xy.append(np.asarray(pcd_vis.points)[pt_indices > 0].mean(0))

    export_mesh_scannet(f'results/{scene}/{scene}_om_vis.ply', np.array(scene_mesh.vertices), np.array(scene_mesh.triangles), vis_colors)
    geometries.append(pcd_vis)
    vis_pcd(geometries)


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------- #
    # replica
    # ----------------------------------------------------------------------------------------- #
    # scene = 'office1'
    # 1. vis gt color
    # with open(f"class_id_colors.json", 'r') as f:
    #     replica_colors = json.load(f)
    # visualize_scene_graph_replica_gt(f'results/{scene}/mesh_semantic.ply', f'results/{scene}/info_semantic.json')
    # scene_pcd = PlyData.read(f'results/{scene}/{scene}_mesh.ply')

    # 2. vis pred random color
    # with open(f'results/{scene}/proposed_fusion_cropformer-none-0.5-ovseg_tap_iou-0.25_recall-0.75_feature-0.75_interval-50-mine.pkl', 'rb') as f:
    #     scene_graph = pickle.load(f)
    # visualize_scene_graph_replica(scene_pcd, scene_graph, scene)
    # 3. vis pred fixed color
    # with open(f'results/{scene}/proposed_fusion_cropformer-none-0.5-ovseg_tap_iou-0.25_recall-0.75_feature-0.75_interval-50-mine-backprj-top1.pkl', 'rb') as f:
    #     scene_graph = pickle.load(f)
    # visualize_scene_graph_replica_fixed(scene_pcd, scene_graph, scene)

    # ----------------------------------------------------------------------------------------- #
    # scannet
    # ----------------------------------------------------------------------------------------- #
    scene = 'scene0565_00'
    ply_path = f'results/{scene}/{scene}_vh_clean_2.ply'
    gt_segs_path = f'results/{scene}/{scene}_vh_clean_2.0.010000.segs.json'
    gt_aggr_path = f'results/{scene}/{scene}.aggregation.json'

    scene_pcd = o3d.io.read_point_cloud(ply_path)
    scene_mesh = o3d.io.read_triangle_mesh(ply_path)

    # 1. vis gt color
    # 2.1 load segments file, over-segmentation id
    # with open(gt_segs_path) as f:
    #     segments = json.load(f)
    #     seg_indices = np.array(segments["segIndices"])
    # # 2.2 load aggregations file, instance with cat_id
    # with open(gt_aggr_path) as f:
    #     aggregation = json.load(f)
    #     seg_groups = np.array(aggregation["segGroups"])
    # labels_pd = pd.read_csv("datasets/constants/scannet/scannetv2-labels.combined.tsv", sep="\t", header=0)
    # semantic_gt200, semantic_gt20, gt_cat_ids, gt_masks = get_gt_semantic(seg_groups, seg_indices, labels_pd, np.array(scene_pcd.points))
    # visualize_scene_graph_scannet_gt_in(scene_pcd, scene_mesh, gt_masks, scene)

    result_path = f'results/{scene}/proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl'
    with open(result_path, 'rb') as f:
        scene_graph = pickle.load(f)
    visualize_scene_graph_scannet(scene_pcd, scene_mesh, scene_graph, scene)

    segment_idx = torch.load(f'results/{scene}/{scene}_vh_clean_2_masks.pt')
    visualize_scene_graph_scannet_pt(scene_pcd, scene_mesh, segment_idx, scene)
