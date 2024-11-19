import open3d as o3d

import itertools
import operator
from collections import deque, Counter
from functools import reduce
from time import perf_counter

import numpy as np
import faiss
import torch

from numba import njit
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


def resolve_overlapping_masks(pred_masks, pred_scores, score_thresh=0.5, device="cuda:0"):
    M, H, W = pred_masks.shape
    pred_masks = torch.from_numpy(pred_masks).to(device)
    panoptic_masks = torch.clone(pred_masks)
    scores = torch.from_numpy(pred_scores)[:, None, None].repeat(1, H, W).to(device)
    scores[~pred_masks] = 0
    indices = ((scores == torch.max(scores, dim=0, keepdim=True).values) & pred_masks).nonzero()
    panoptic_masks = torch.zeros((M, H, W), dtype=torch.bool, device=device)
    panoptic_masks[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    panoptic_masks[scores > score_thresh] = True  # if prediction score is high enough, keep the mask anyway
    return panoptic_masks.detach().cpu().numpy()


# -------------------------------------------------------------------------------------- #
# project and back-project
# -------------------------------------------------------------------------------------- #
def create_pcd_from_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_pcd_from_depth(
        depth_im: np.ndarray,
        cam_intr: np.ndarray,
        color_im: np.ndarray = None,
        cam_extr: np.ndarray = np.eye(4)
):
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr

    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(color_im_o3d, depth_im_o3d, depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic=cam_extr)
    else:
        pcd = o3d.geometry.PointCloud().create_from_depth_image(depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr, depth_scale=1)
    return pcd


@njit
def compute_projected_pts(pts, cam_intr):
    N = pts.shape[0]
    projected_pts = np.empty((N, 2), dtype=np.int64)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(pts.shape[0]):
        z = pts[i, 2]
        x = int(np.round(fx * pts[i, 0] / z + cx))
        y = int(np.round(fy * pts[i, 1] / z + cy))
        projected_pts[i, 0] = x
        projected_pts[i, 1] = y
    return projected_pts


@njit
def compute_visible_indices(pts, projected_pts, depth_im, depth_thresh=0.005):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    visible_indices = visibility_mask.nonzero()[0]
    return visible_indices


@njit
def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.005):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    return visibility_mask


@njit
def compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks):
    N = scene_pts.shape[0]
    M, h, w = pred_masks.shape  # (M, H, W)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visible_indices = np.nonzero(visibility_mask)[0]
    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x] and pred_masks[m, max(0, y - 1), x] and pred_masks[m, min(y + 1, h), x] and pred_masks[m, y, max(0, x - 1)] and pred_masks[m, y, min(x + 1, w)]:
                masked_pts[m, i] = True
    return masked_pts


# -------------------------------------------------------------------------------------- #
# similarity, match and merge
# -------------------------------------------------------------------------------------- #
def compute_relation_matrix(masked_pts, instance_pt_count, visibility_mask):
    visibility_mask = torch.from_numpy(visibility_mask).to(instance_pt_count.device)
    instance_pt_mask = instance_pt_count.to(torch.bool)
    instance_pt_mask[:, ~visibility_mask] = False
    instance_pt_mask = instance_pt_mask.to(torch.float32)
    masked_pts = masked_pts.to(torch.float32)

    intersection = masked_pts @ instance_pt_mask.T  # (M, num_instances)
    masked_pts_sum = masked_pts.sum(1, keepdims=True)  # (M, 1)
    instance_pt_mask_sum = instance_pt_mask.sum(1, keepdims=True)  # (num_instances, 1)

    union = masked_pts_sum + instance_pt_mask_sum.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    recall_matrix = intersection / (instance_pt_mask_sum.T + 1e-6)
    precision_matrix = intersection / (masked_pts_sum + 1e-6)

    return iou_matrix, precision_matrix, recall_matrix


def compute_relation_matrix2(detect_segments, anchor_objects, visibility_mask):
    visibility_mask = torch.from_numpy(visibility_mask)

    masked_pts = detect_segments.get_stacked_values_torch('mask_ids').to(torch.float32)
    instance_pt_mask = anchor_objects.get_stacked_values_torch('mask_ids')
    instance_pt_mask[:, ~visibility_mask] = False
    instance_pt_mask = instance_pt_mask.to(torch.float32)

    masked_pts_sum = masked_pts.sum(1, keepdims=True)  # (M, 1)
    instance_pt_mask_sum = instance_pt_mask.sum(1, keepdims=True)  # (N, 1)

    intersection = masked_pts @ instance_pt_mask.T  # (M, N)
    union = masked_pts_sum + instance_pt_mask_sum.T - intersection  # (M, N)
    iou_matrix = intersection / (union + 1e-6)
    recall_matrix = intersection / (instance_pt_mask_sum.T + 1e-6)
    precision_matrix = intersection / (masked_pts_sum + 1e-6)

    return iou_matrix, precision_matrix, recall_matrix


def compute_relation_matrix3(detect_segments, anchor_objects):
    masked_pts = detect_segments.get_stacked_values_torch('mask_ids').to(torch.float32)
    instance_pt_mask = anchor_objects.get_stacked_values_torch('mask_ids').to(torch.float32)

    masked_pts_sum = masked_pts.sum(1, keepdims=True)  # (M, 1)
    instance_pt_mask_sum = instance_pt_mask.sum(1, keepdims=True)  # (N, 1)

    intersection = masked_pts @ instance_pt_mask.T  # (M, N)
    union = masked_pts_sum + instance_pt_mask_sum.T - intersection  # (M, N)
    iou_matrix = intersection / (union + 1e-6)
    recall_matrix = intersection / (instance_pt_mask_sum.T + 1e-6)
    precision_matrix = intersection / (masked_pts_sum + 1e-6)

    return iou_matrix, precision_matrix, recall_matrix


def compute_relation_matrix_w_overlap(masked_pts, instance_pt_count, visibility_mask, scene_pts):
    visibility_mask = torch.from_numpy(visibility_mask).to(instance_pt_count.device)
    instance_pt_mask = instance_pt_count.to(torch.bool)
    instance_pt_mask[:, ~visibility_mask] = False
    instance_pt_mask = instance_pt_mask.to(torch.float32)
    masked_pts_f = masked_pts.to(torch.float32)

    intersection = masked_pts_f @ instance_pt_mask.T  # (M, num_instances)
    masked_pts_sum = masked_pts_f.sum(1, keepdims=True)  # (M, 1)
    instance_pt_mask_sum = instance_pt_mask.sum(1, keepdims=True)  # (num_instances, 1)

    union = masked_pts_sum + instance_pt_mask_sum.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    recall_matrix = intersection / (instance_pt_mask_sum.T + 1e-6)
    precision_matrix = intersection / (masked_pts_sum + 1e-6)

    # * overlap matrix here * #
    m = instance_pt_count.shape[0]  # n existing segments
    n = masked_pts.shape[0]  # m newly detected segments
    overlap_matrix = np.zeros((m, n))
    if m != 0:
        # 1. convert the point clouds into numpy arrays and then into FAISS indices for efficient search;
        points_map = [scene_pts[instance_pt_mask] for instance_pt_mask in instance_pt_count.to(torch.bool).cpu().numpy()]  # m arrays
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map]  # m indices
        # add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, points_map):
            index.add(arr)
        points_new = [scene_pts[instance_pt_mask] for instance_pt_mask in masked_pts.cpu().numpy()]  # n arrays

        # segment pcd
        objects_map = [create_pcd_from_points(points_seg) for points_seg in points_map]
        objects_new = [create_pcd_from_points(points_seg) for points_seg in points_new]

        try:
            bbox_map = torch.tensor([get_bounding_box(points_seg) for points_seg in points_map])
            bbox_new = torch.tensor([get_bounding_box(points_seg) for points_seg in points_new])
            iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new)  # (m, n)
        except ValueError:
            print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
            bbox_map = []
            bbox_new = []
            for pcd in objects_map:
                bbox_map.append(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))
            for pcd in objects_new:
                bbox_new.append(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))
            bbox_map = torch.from_numpy(np.stack(bbox_map))
            bbox_new = torch.from_numpy(np.stack(bbox_new))

            iou = compute_iou_batch(bbox_map, bbox_new)  # (m, n)

        # Compute the pairwise overlaps
        for i in range(m):
            for j in range(n):
                if iou[i, j] < 1e-6:
                    continue

                D, I = indices[i].search(points_new[j], 1)  # search new object j in map object i

                overlap = (D < 0.025 ** 2).sum()  # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(points_new[j])

    return iou_matrix, precision_matrix, recall_matrix, overlap_matrix.T


def compute_relation_matrix_self(instance_pt_count):
    if not torch.is_tensor(instance_pt_count):
        instance_pt_count = torch.from_numpy(instance_pt_count)
    instance_pt_mask = instance_pt_count.to(torch.bool).to(torch.float32)

    intersection = instance_pt_mask @ instance_pt_mask.T  # (M, num_instances)
    inliers = instance_pt_mask.sum(1, keepdims=True)
    union = inliers + inliers.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    precision_matrix = intersection / (inliers.T + 1e-6)
    recall_matrix = intersection / (inliers + 1e-6)
    return iou_matrix, precision_matrix, recall_matrix


def compute_relation_matrix_self2(detect_segments):
    instance_pt_count = detect_segments.get_stacked_values_torch('mask_ids')
    instance_pt_mask = instance_pt_count.to(torch.bool).to(torch.float32)

    inliers = instance_pt_mask.sum(1, keepdims=True)
    intersection = instance_pt_mask @ instance_pt_mask.T  # (M, M)
    union = inliers + inliers.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    precision_matrix = intersection / (inliers.T + 1e-6)
    recall_matrix = intersection / (inliers + 1e-6)
    return iou_matrix, precision_matrix, recall_matrix


def compute_relation_normals_matrix_self(detect_segments, gt_points):
    normals = []
    for seg in detect_segments:
        pcd_segment = o3d.geometry.PointCloud()
        pcd_segment.points = o3d.utility.Vector3dVector(gt_points[seg.mask_ids.cpu().numpy()])
        pcd_segment.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals_segment = np.asarray(pcd_segment.normals)
        avg_normal = np.mean(normals_segment, axis=0)
        normals.append(avg_normal)
    normals = np.stack(normals)

    return torch.from_numpy(normals @ normals.T)


def compute_recall_3d_matrix(boxes_o3d, intersection_3d_matrix):
    N = len(boxes_o3d)
    assert intersection_3d_matrix.shape[0] == N
    volume = torch.tensor([box.volume() for box in boxes_o3d], device=intersection_3d_matrix.device)  # (N,)
    scale = 100  # magic number, intersection_3d_matrix is scaled to prevent numerical issues
    recall_3d_matrix = intersection_3d_matrix / (volume.unsqueeze(0) * scale ** 3)
    return recall_3d_matrix


@njit
def iou_matching_greedy(iou_matrix, area_counter, iou_thresh=0.5, unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    matched_keys = np.zeros((N,), dtype=np.bool_)
    sorted_query = np.argsort(-area_counter)  # sort query segments by area (descending order)
    for i in sorted_query:
        for j in np.argsort(-iou_matrix[i]):  # sort key segments by iou (descending order)
            if matched_keys[j]:
                continue
            iou = iou_matrix[i, j]
            if iou >= iou_thresh:
                correspondences[i] = j
                matched_keys[j] = True
            break
    return correspondences


@njit
def iou_matching_greedy_2(iou_matrix, iou_thresh=0.5, unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    for i in range(M):
        for j in np.argsort(-iou_matrix[i]):  # sort key segments by iou (descending order)
            iou = iou_matrix[i, j]
            if iou >= iou_thresh:
                correspondences[i] = j
            break
    return correspondences


def iou_matching_hungarian(iou_matrix, iou_thresh=0.5, unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    mask = iou_matrix[row_indices, col_indices] > iou_thresh
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    correspondences[row_indices] = col_indices
    return correspondences


@njit
def matching_with_feature_rejection(iou_matrix, overlap_matrix, recall_matrix, feature_similarity_matrix,
                                    iou_thresh=0.3, precision_recall_thresh=0.8, feature_similarity_thresh=0.8,
                                    unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    for i in range(M):
        for j in np.argsort(-iou_matrix[i]):  # sort key segments by iou (descending order)
            iou = iou_matrix[i, j]
            feature_similarity = feature_similarity_matrix[i, j]
            if (iou >= iou_thresh) and (feature_similarity >= feature_similarity_thresh):
                correspondences[i] = j
                break
    return correspondences


def find_connected_components(adj_matrix):
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency matrix should be a square matrix"

    N = adj_matrix.shape[0]
    clusters = []
    visited = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])
        visited[i] = True
        while queue:
            j = queue.popleft()
            cluster.append(j)
            for k in np.nonzero(adj_matrix[j])[0]:
                if not visited[k]:
                    queue.append(k)
                    visited[k] = True
        clusters.append(cluster)
    return clusters


def merge_instances(instance_pt_count, instance_features, instance_detections, clusters, inplace=True):
    _, N = instance_pt_count.shape
    _, C = instance_features.shape
    M = len(clusters)
    device = instance_pt_count.device
    merged_instance_pt_count = torch.zeros((M, N), dtype=instance_pt_count.dtype, device=device)
    merged_features = torch.zeros((M, C), dtype=instance_features.dtype, device=device)
    merged_detections = dict()

    for i, cluster in enumerate(clusters):
        merged_instance_pt_count[i] = instance_pt_count[cluster].sum(0)
        merged_features[i] = instance_features[cluster].sum(0)
        merged_detections[i] = reduce(operator.add, (instance_detections[j] for j in cluster))
        assert len(merged_detections[i]) == sum([len(instance_detections[j]) for j in cluster])
    if inplace:
        instance_pt_count[:M] = merged_instance_pt_count
        instance_features[:M] = merged_features
        instance_pt_count[M:] = 0
        instance_features[M:] = 0

    return merged_instance_pt_count, merged_features, merged_detections


def integrate_instances(instance_pt_count, num_instances, masked_pts, correspondences, instance_features, pred_features, instance_detections, frame_id, valid_mask_indices, unmatched_indicator=-1):
    assert valid_mask_indices.shape[0] == masked_pts.shape[0]
    for m in range(masked_pts.shape[0]):
        c = correspondences[m]
        if c == unmatched_indicator:
            c = num_instances
            instance_detections[c] = []
            num_instances += 1
        # add the segment to the pointer
        instance_pt_count[c] += masked_pts[m]
        instance_features[c] += pred_features[m]
        mask_size = torch.sum(masked_pts[m]).detach().cpu().numpy()
        instance_detections[c].append((frame_id, valid_mask_indices[m], mask_size))
    return num_instances


# -------------------------------------------------------------------------------------- #
# bbox and iou calculation
# -------------------------------------------------------------------------------------- #
def get_bounding_box(pcd_xyz, accurate=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

    if accurate and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True).get_box_points()
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box().get_box_points()
    else:
        return pcd.get_axis_aligned_bounding_box().get_box_points()


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of axis-aligned 3D bounding boxes.

    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1)  # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1)  # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1)  # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1)  # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou


def compute_3d_iou_accuracte_batch(bbox1, bbox2):
    """
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.

    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]

    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())

    return iou


def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    """
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention.

    bbox: (N, 8, D)

    returns: (N, 8, D)
    """
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)

    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)

    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)

    new_bbox = torch.stack([
        center - va / 2.0 - vb / 2.0 - vc / 2.0,
        center + va / 2.0 - vb / 2.0 - vc / 2.0,
        center - va / 2.0 + vb / 2.0 - vc / 2.0,
        center - va / 2.0 - vb / 2.0 + vc / 2.0,
        center + va / 2.0 + vb / 2.0 + vc / 2.0,
        center - va / 2.0 + vb / 2.0 + vc / 2.0,
        center + va / 2.0 - vb / 2.0 + vc / 2.0,
        center + va / 2.0 + vb / 2.0 - vc / 2.0,
    ], dim=1)  # shape: (N, 8, D)

    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)

    return new_bbox


# -------------------------------------------------------------------------------------- #
# post process and fusion
# -------------------------------------------------------------------------------------- #
def filter_pt_by_visibility_count(instance_pt_count, visibility_count, visibility_thresh=0.2, inplace=False):
    if isinstance(visibility_count, np.ndarray):
        visibility_count = torch.from_numpy(visibility_count).to(instance_pt_count.device)
    instance_pt_visibility = instance_pt_count / visibility_count.clip(min=1e-6).unsqueeze(0)
    if not inplace:
        instance_pt_count = torch.clone(instance_pt_count)
    instance_pt_count[instance_pt_visibility < visibility_thresh] = 0
    return instance_pt_count


def filter_pt_by_visibility_count2(detect_segments, visibility_count, visibility_thresh=0.1):
    if isinstance(visibility_count, np.ndarray):
        visibility_count = torch.from_numpy(visibility_count)

    for segment in detect_segments:
        instance_pt_count = segment.instance_pt_count
        instance_pt_visibility = instance_pt_count / visibility_count.clip(min=1e-6)
        instance_pt_count[instance_pt_visibility < visibility_thresh] = 0

        segment.mask_ids = segment.mask_ids * instance_pt_count.bool()
        segment.instance_pt_count = instance_pt_count

    return detect_segments


def filter_pt_by_visibility_count3(detect_segments, visibility_thresh=0.05):
    for segment in detect_segments:
        instance_pt_count = segment.instance_pt_count

        instance_pt_visibility = instance_pt_count / max(instance_pt_count.max(), 1)
        instance_pt_count[instance_pt_visibility < visibility_thresh] = 0

        segment.mask_ids = segment.mask_ids * instance_pt_count.bool()
        segment.instance_pt_count = segment.instance_pt_count * instance_pt_count.bool()

    return detect_segments


def filter_by_instance_size(instance_pt_count, instance_detections, size_thresh=50, median=True):
    # filter by size threshold
    sizes = instance_pt_count.to(torch.bool).sum(1)
    mask = sizes > size_thresh

    # filter by median mask size, IMPORTANT!
    median_mask_sizes = []
    for i in range(len(instance_detections)):
        instance_detection = instance_detections[i]
        median_mask_sizes.append(np.median([detection[2] for detection in instance_detection]))
    median_mask_sizes = torch.tensor(median_mask_sizes, device=mask.device)
    mask2 = sizes > median_mask_sizes

    if median:
        final_mask = mask & mask2
    else:
        final_mask = mask

    keep_indices = torch.nonzero(final_mask)[:, 0]
    return keep_indices


def filter_by_instance_size_no_media2(detect_segments, size_thresh=50, median=True):
    # filter by size threshold
    sizes = detect_segments.get_stacked_values_numpy('mask_ids').sum(1)
    mask = sizes > size_thresh

    # filter by median mask size, IMPORTANT!
    if median:
        median_mask_sizes = []
        for i in range(len(detect_segments)):
            instance_detection = detect_segments[i]
            median_mask_sizes.append(np.median(instance_detection.mask_size_list))
        median_mask_sizes = np.array(median_mask_sizes)
        mask2 = sizes > median_mask_sizes
        mask = mask & mask2

    detect_segments = detect_segments.slice_by_mask(mask.tolist())
    return detect_segments


def filter_by_instance_size_no_media(instance_pt_count, instance_detections, size_thresh=50):
    # filter by size threshold
    sizes = instance_pt_count.to(torch.bool).sum(1)
    mask = sizes > size_thresh

    # filter by median mask size, IMPORTANT!
    final_mask = mask

    keep_indices = torch.nonzero(final_mask)[:, 0]
    return keep_indices


def filter_pt_by_dbscan_noise(gt_point, instance_pt_count, eps=0.1, min_points=10):
    for i in range(instance_pt_count.shape[0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_point[instance_pt_count[i].to(torch.bool).cpu().numpy()])

        # Remove noise via clustering
        pcd_clusters = pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
        )

        # Convert to numpy arrays
        pcd_clusters = np.array(pcd_clusters)
        # Count all labels in the cluster
        counter = Counter(pcd_clusters)

        # Remove the noise label
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            # Create mask for points in the largest cluster
            largest_mask = pcd_clusters == most_common_label

            if np.sum(largest_mask) < 25:
                continue

            instance_pt_count[i][instance_pt_count[i] > 0] = instance_pt_count[i][instance_pt_count[i] > 0] * torch.from_numpy(largest_mask.astype(np.int32))


def filter_pt_by_dbscan_noise2(gt_point, detect_segments, eps=0.1, min_points=10):
    for idx, seg in enumerate(detect_segments):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_point[seg.mask_ids])

        # denoise via clustering
        pcd_clusters = pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
        )

        # Convert to numpy arrays
        pcd_clusters = np.array(pcd_clusters)

        # Remove the noise label
        counter = Counter(pcd_clusters)
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]

            # Create mask for points in the largest cluster
            largest_mask = pcd_clusters == most_common_label

            if np.sum(largest_mask) < 5:
                continue

            seg.instance_pt_count[seg.mask_ids > 0] = seg.instance_pt_count[seg.mask_ids > 0] * torch.from_numpy(largest_mask.astype(np.int32))
            seg.mask_ids[seg.mask_ids > 0] = seg.mask_ids[seg.mask_ids > 0] * torch.from_numpy(largest_mask.astype(np.int32)).bool()
            assert torch.sum(seg.instance_pt_count > 0) == torch.sum(seg.mask_ids > 0)

    return detect_segments


def post_processing(scene_pcd, instance_pt_count, instance_features, instance_detections, size_thresh=50, iou_thresh=0.3, recall_thresh=0.5, feature_similarity_thresh=0.7):
    num_instances = instance_pt_count.shape[0]

    # ===================== filter by segment connectivity =====================
    tic = perf_counter()
    mean_stats = np.empty((num_instances, 3), dtype=np.float32)
    cov_stats = np.empty((num_instances, 3, 3), dtype=np.float32)

    pcd_pts = []
    keep_indices = []
    for i in range(num_instances):
        pt_indices = instance_pt_count[i].nonzero()[:, 0].detach().cpu().numpy()
        segment_pcd = scene_pcd.select_by_index(pt_indices)
        mean, cov = segment_pcd.compute_mean_and_covariance()
        mean_stats[i] = mean
        cov_stats[i] = cov
        pts = np.array(segment_pcd.points)
        pcd_pts.append(pts)

        dbscan = DBSCAN(eps=0.1, min_samples=1)
        dbscan.fit(pts)

        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        inlier_indices = pt_indices[dbscan.labels_ == largest_cluster_label]
        outlier_mask = np.ones(instance_pt_count.shape[1], dtype=bool)
        outlier_mask[inlier_indices] = False
        instance_pt_count[i, outlier_mask] = 0
        if len(inlier_indices) > 10:
            keep_indices.append(i)

    keep_indices = torch.LongTensor(keep_indices)
    instance_pt_count = instance_pt_count[keep_indices]
    instance_features = instance_features[keep_indices]
    instance_detections = {new_idx: instance_detections[idx.item()] for new_idx, idx in enumerate(keep_indices)}

    return instance_pt_count, instance_features, instance_detections


def post_processing2(scene_pcd, anchor_objects, size_thresh=50, iou_thresh=0.3, recall_thresh=0.5, feature_similarity_thresh=0.7):
    num_instances = len(anchor_objects)

    # ===================== filter by segment connectivity =====================
    keep_indices = []
    for i in range(num_instances):
        pt_indices = anchor_objects[i].mask_ids.nonzero()[:, 0].detach().cpu().numpy()
        segment_pcd = scene_pcd.select_by_index(pt_indices)
        pts = np.array(segment_pcd.points)

        dbscan = DBSCAN(eps=0.1, min_samples=1)
        dbscan.fit(pts)

        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        inlier_indices = pt_indices[dbscan.labels_ == largest_cluster_label]
        outlier_mask = np.ones(anchor_objects[i].instance_pt_count.shape[0], dtype=bool)
        outlier_mask[inlier_indices] = False

        anchor_objects[i].mask_ids[torch.from_numpy(outlier_mask)] = 0
        anchor_objects[i].instance_pt_count[torch.from_numpy(outlier_mask)] = 0
        if len(inlier_indices) > 10:
            keep_indices.append(i)

    anchor_objects = anchor_objects.slice_by_indices(keep_indices)

    return anchor_objects
