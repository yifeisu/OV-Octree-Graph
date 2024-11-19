import pickle
import pickletools

import torch

import numpy as np
from plyfile import PlyData
from sklearn.neighbors import BallTree


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


if __name__ == '__main__':
    scene = 'office4'

    with open(f'results/{scene}/proposed_fusion_cropformer-none-0.5-ovseg_tap_iou-0.25_recall-0.75_feature-0.75_interval-50-mine-backprj-top1.pkl', 'rb') as f:
        scene_graph = pickle.load(f)

    scene_pcd = PlyData.read(f'results/{scene}/{scene}_mesh.ply')
    gt_xyz = np.stack([scene_pcd["vertex"]["x"], scene_pcd["vertex"]["y"], scene_pcd["vertex"]["z"]]).T

    point_instance_id = np.ones([gt_xyz.shape[0]]) * -1

    node_ids = list(scene_graph.nodes)
    node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    print(f"{len(node_ids) = }")

    for idx, node_id in enumerate(node_ids):
        node = scene_graph.nodes[node_id]
        pt_indices = node['pt_indices']
        point_instance_id[pt_indices] = node_id

    # knn
    coords_labels = np.zeros((gt_xyz.shape[0], 4))
    coords_labels[:, :3] = gt_xyz
    coords_labels[:, -1] = point_instance_id
    # concat coords and labels for gt pcd
    coords_gt = np.zeros((gt_xyz.shape[0], 4))
    coords_gt[:, :3] = gt_xyz
    coords_gt[:, -1] = point_instance_id
    # knn interpolation
    match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
    pred_instance = match_pc[:, -1]

    #
    for idx, node_id in enumerate(node_ids):
        scene_graph.nodes[node_id]['pt_indices'] = torch.nonzero(torch.from_numpy(pred_instance == node_id)).cpu().numpy().astype(np.int32)

    with open(f'results/{scene}/proposed_fusion_cropformer-none-0.5-ovseg_tap_iou-0.25_recall-0.75_feature-0.75_interval-50-mine-backprj-top1-knn.pkl', 'wb') as f:
        pickle.dump(scene_graph, f)
