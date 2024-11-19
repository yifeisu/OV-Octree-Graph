import numpy as np
import torch
from sklearn.neighbors import BallTree


# -------------------------------------------------------------------------------------- #
# evaluation utils
# -------------------------------------------------------------------------------------- #
def get_gt_instances(gt_instance_ids, valid_ids=None):
    unique_instance_ids = np.unique(gt_instance_ids)
    N = unique_instance_ids.shape[0]
    masks = []
    cat_ids = []
    for i in range(N):
        unique_id = unique_instance_ids[i]
        cat_id = unique_id // 1000
        if valid_ids and cat_id not in valid_ids:
            continue
        mask = gt_instance_ids == unique_id
        cat_ids.append(cat_id)
        masks.append(mask)
    cat_ids = np.array(cat_ids)
    masks = np.stack(masks)
    return cat_ids, masks


def get_predicted_instances(scene_graph, feature_name="feature"):
    node_visual_features = []
    node_caption_features = []
    node_masks = []
    n_pts = scene_graph.graph['n_pts']

    node_visual_features_list = []
    node_caption_features_list = []

    for node in scene_graph.nodes:
        if feature_name == "feature":
            node_visual_features.append(scene_graph.nodes[node]['feature'])
            node_caption_features.append(scene_graph.nodes[node]['feature'])
            node_visual_features_list.append(None)
            node_caption_features_list.append(None)

        else:
            node_visual_features.append(scene_graph.nodes[node]['back_prj_feat_mean'])
            node_caption_features.append(scene_graph.nodes[node]['back_prj_cap_feat_mean'] if "back_prj_cap_feat_mean" in scene_graph.nodes[node] else scene_graph.nodes[node]['back_prj_feat_mean'])  # back_prj_cap_feat_mean
            node_visual_features_list.append(scene_graph.nodes[node]['back_prj_feat_list'])
            node_caption_features_list.append(scene_graph.nodes[node]['back_prj_cap_feat_list'])

        node_mask = np.zeros(n_pts, dtype=np.bool_)
        node_mask[scene_graph.nodes[node]["pt_indices"]] = True
        node_masks.append(node_mask)

    node_visual_features = np.vstack(node_visual_features)  # (N, n_dims)
    node_caption_features = np.vstack(node_caption_features)  # (N, n_dims)
    node_masks = np.stack(node_masks)  # (N, n_pts)

    return node_visual_features, node_masks, node_caption_features, node_visual_features_list, node_caption_features_list


# -------------------------------------------------------------------------------------- #
# ap calculation
# ------------------------------------------------------------------------------------- #
def CalculateAveragePrecision(rec, prec):
    """
    https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py#L295
    """
    mrec = [0]
    for e in rec:
        mrec.append(e)
    mrec.append(1)

    mpre = [0]
    for e in prec:
        mpre.append(e)
    mpre.append(0)

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1 + i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


# -------------------------------------------------------------------------------------- #
# ovir3d's metric below
# -------------------------------------------------------------------------------------- #
def compute_ap_for_each_scan(pred_features, pred_caption_features, pred_masks, gt_cat_ids, gt_masks, cat_id_to_feature, lamb):
    """
    ref: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
    """
    iou_thresholds = np.zeros(11)
    iou_thresholds[0] = 0.25
    iou_thresholds[1:] = np.arange(0.5, 1, 0.05)

    ap_dict = dict()
    unique_cat_ids = np.unique(gt_cat_ids)
    for cat_id in unique_cat_ids:
        if cat_id not in cat_id_to_feature:
            continue

        ap_dict[cat_id] = dict()
        cat_feature = cat_id_to_feature[cat_id]
        similarities1 = pred_features @ cat_feature
        similarities2 = pred_caption_features @ cat_feature

        similarities = torch.softmax(torch.from_numpy(similarities1), dim=-1) * lamb + torch.softmax(torch.from_numpy(similarities2), dim=-1) * (1 - lamb)

        # if using K representative features, take the max similarity
        if pred_features.shape[0] != pred_masks.shape[0]:
            K = pred_features.shape[0] // pred_masks.shape[0]
            similarities = similarities.reshape(-1, K).max(1)
            assert similarities.shape[0] == pred_masks.shape[0]

        # sort pred mask based on similarity (descending order)
        sorted_indices = np.argsort(-similarities)
        sorted_node_masks = pred_masks[sorted_indices]  # (N, n_pts)
        cat_masks = gt_masks[gt_cat_ids == cat_id]  # (M, n_pts)

        # compute IoU matrix between all predictions and gt masks
        intersection = sorted_node_masks.astype(np.float32) @ cat_masks.astype(np.float32).T  # (N, M)
        union = sorted_node_masks.sum(1, keepdims=True) + cat_masks.sum(1, keepdims=True).T - intersection
        iou_matrix = intersection / (union + 1e-6)
        N, M = iou_matrix.shape  # (num_pred, num_gt)

        # ref: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py#L102
        for iou_threshold in iou_thresholds:
            visited = np.zeros((M,), dtype=np.bool_)
            TP = np.zeros((N,), dtype=np.bool_)
            FP = np.zeros((N,), dtype=np.bool_)
            for n in range(N):  # go through all sorted predictions based on similarity
                iou_max = -np.inf
                m_match = -1
                for m in range(M):  # find the best gt match for this prediction
                    iou = iou_matrix[n, m]
                    if iou > iou_max:
                        iou_max = iou
                        m_match = m
                if iou_max >= iou_threshold:
                    if not visited[m_match]:
                        TP[n] = True
                        visited[m_match] = True
                    else:  # gt mask has been matched with other predictions
                        FP[n] = True
                else:
                    FP[n] = True
            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum(FP)
            precisions = acc_TP / (acc_FP + acc_TP)
            recalls = acc_TP / M
            ap, _, _, _ = CalculateAveragePrecision(recalls, precisions)
            ap_name = f"ap_{int(iou_threshold * 100)}"
            ap_dict[cat_id][ap_name] = ap
    return ap_dict


def confusion_matrix(pred_ids, gt_ids, num_classes, IGNORE_INDEX=-1):
    """calculate the confusion matrix."""
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)

    idxs = gt_ids != IGNORE_INDEX

    if IGNORE_INDEX in pred_ids:  # some points have no feature assigned for prediction
        pred_ids[pred_ids == IGNORE_INDEX] = num_classes

    confmatrix = np.zeros([num_classes, num_classes])
    for class_gt_int in range(num_classes):
        tensor_gt_class = np.equal(gt_ids[idxs], class_gt_int)
        for class_pred_int in range(num_classes):
            tensor_pred_class = np.equal(pred_ids[idxs], class_pred_int)
            tensor_pred_class = tensor_gt_class * tensor_pred_class
            count = np.sum(tensor_pred_class)
            confmatrix[class_gt_int, class_pred_int] += count

    return confmatrix


# ----------------------------------------------------------------------------------------- #
# hovsg's metric below
# ----------------------------------------------------------------------------------------- #

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


def pixel_accuracy(eval_segm, gt_segm, ignore=[]):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: pixel accuracy
    """

    # print("unique classes: ", np.unique(gt_segm))
    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

        # per_class_acc =  np.sum(np.logical_and(curr_eval_mask, curr_gt_mask)) / np.sum(curr_gt_mask)
        # print(i, per_class_acc)

    if sum_t_i == 0:
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm, ignore=[]):
    """
    (1/n_cl) sum_i(n_ii/t_i)
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: mean accuracy
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    overlap, n_overlap = get_ignore_classes_num(cl, ignore)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl
    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if t_i != 0:
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.sum(accuracy) / float(n_cl - n_overlap)
    return mean_accuracy_


def per_class_IU(eval_segm, gt_segm, ignore=[]):
    """
    for each class, compute
    n_ii / (t_i + sum_j(n_ji) - n_ii)
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: per class IU
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    gt_cl, n_cl_gt = extract_classes(gt_segm)
    overlap, n_overlap = get_ignore_classes_num(gt_cl, ignore)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        # if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
        #     continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    # mean_IU_ = np.sum(IU) / (n_cl_gt - n_overlap)
    return IU


def mean_IU(eval_segm, gt_segm, ignore=[], class_id_name=None):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: mean IU
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    gt_cl, n_cl_gt = extract_classes(gt_segm)
    overlap, n_overlap = get_ignore_classes_num(gt_cl, ignore)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
        print(c, class_id_name[c], IU[i], np.sum(curr_eval_mask), np.sum(curr_gt_mask))

    mean_IU_ = np.sum(IU) / (n_cl_gt - n_overlap)
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm, ignore=[]):
    """
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: frequency weighted IU
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    n_ignore = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if c in ignore:
            n_ignore += np.sum(curr_gt_mask)
            continue

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm) - n_ignore

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


"""
Auxiliary functions used during evaluation.
"""


def get_ignore_classes_num(cl, ignore):
    """
    Returns the number of classes to ignore
    :param cl: list of classes
    :param ignore: list of classes to ignore
    :return: number of classes to ignore
    """
    overlap = [c for c in cl if c in ignore]
    return overlap, len(overlap)


def get_pixel_area(segm):
    """
    Returns the area of the segmentation
    :param segm: 2D array, segmentation
    :return: area of the segmentation
    """
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    """"
    Extracts the masks of the segmentation
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param cl: list of classes
    :param n_cl: number of classes
    :return: masks of the segmentation
    """
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    """
    Extracts the classes from the segmentation
    :param segm: 2D array, segmentation
    :return: classes and number of classes
    """
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    """
    Returns the union of the classes
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :return: union of the classes
    """
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    """
    Extracts the masks of the segmentation
    :param segm: 2D array, segmentation
    :param cl: list of classes
    :param n_cl: number of classes
    :return: masks of the segmentation
    """
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    """
    Returns the size of the segmentation
    :param segm: 2D array, segmentation
    :return: size of the segmentation
    """
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    """
    Checks the size of the segmentation
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    """
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


"""
Exceptions
"""


class EvalSegErr(Exception):
    """
    Custom exception for errors during evaluation
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
