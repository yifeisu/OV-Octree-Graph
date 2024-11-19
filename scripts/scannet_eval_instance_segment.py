import sys

import open3d as o3d

import os
import pickle
import argparse
import tqdm

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from omegaconf import OmegaConf
from collections import deque, Counter, defaultdict

import clip
import torch

from openmask3d_tools.eval_semantic_instance import evaluate
from datasets.constants.scannet.scannet200_constants import SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200
from ovgrpah.utils.scene_path import parse_scene_path


class InstSegEvaluator:
    def __init__(self, dataset_type, clip_model_type, sentence_structure):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type
        self.clip_model_type = clip_model_type
        self.clip_model = self.get_clip_model(clip_model_type)
        self.query_sentences = self.get_query_sentences(dataset_type, sentence_structure)
        self.feature_size = self.get_feature_size(clip_model_type)
        self.text_query_embeddings = self.get_text_query_embeddings().numpy()  # torch.Size([20, 768])
        self.set_label_and_color_mapper(dataset_type)

    def get_query_sentences(self, dataset_type, sentence_structure="a {} in a scene"):
        if dataset_type == 'scannet':
            label_list = list(CLASS_LABELS_20)
            label_list[-1] = 'other'  # replace otherfurniture with other, following OpenScene
        elif dataset_type == 'scannet200':
            label_list = list(CLASS_LABELS_200)
        else:
            raise NotImplementedError
        return [sentence_structure.format(label) for label in label_list]

    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model

    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            # print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized = (sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings

    def set_label_and_color_mapper(self, dataset_type):
        if dataset_type == 'scannet':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_20)}.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_20.get)
        elif dataset_type == 'scannet200':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_200)}.get)
            self.color_mapper = np.vectorize(SCANNET_COLOR_MAP_200.get)
        else:
            raise NotImplementedError

    def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        # normalize mask features
        mask_features_normalized = mask_features / np.linalg.norm(mask_features, axis=1)[..., None]

        similarity_scores = mask_features_normalized @ self.text_query_embeddings.T  # (177, 20)
        max_class_similarity_scores = np.max(similarity_scores, axis=1)  # does not correspond to class probabilities
        max_ind = np.argmax(similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)
        pred_classes = max_ind_remapped

        return masks, pred_classes, max_class_similarity_scores

    def compute_classes_per_mask_diff_scores(self, masks_path, mask_features_path, keep_first=None):
        pred_masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        keep_mask = np.asarray([True for el in range(pred_masks.shape[1])])
        if keep_first:
            keep_mask[keep_first:] = False

        # normalize mask features
        mask_features_normalized = mask_features / np.linalg.norm(mask_features, axis=1)[..., None]
        mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0

        per_class_similarity_scores = mask_features_normalized @ self.text_query_embeddings.T  # (177, 20)
        max_ind = np.argmax(per_class_similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)

        pred_masks = pred_masks[:, keep_mask]
        pred_classes = max_ind_remapped[keep_mask]
        pred_scores = np.ones(pred_classes.shape)

        return pred_masks, pred_classes, pred_scores

    @staticmethod
    def evaluate_full(preds, scene_gt_dir, dataset, output_file='temp_output.txt', no_class=False):
        # pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))

        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset, no_class=no_class)
        # read .txt file: scene0000_01.txt has three parameters each row: the mask file for the instance, the id of the instance, and the score.

        return inst_AP


def test_pipeline_full_scannet200(mask_features_dir,
                                  gt_dir,
                                  pred_root_dir,
                                  sentence_structure,
                                  feature_file_template,
                                  dataset_type='scannet200',
                                  clip_model_type='ViT-L/14@336px',
                                  keep_first=None,
                                  scene_list_file='evaluation/val_scenes_scannet200.txt',
                                  masks_template='_masks.pt'
                                  ):
    evaluator = InstSegEvaluator(dataset_type, clip_model_type, sentence_structure)
    print('[INFO]', dataset_type, clip_model_type, sentence_structure)

    with open(scene_list_file, 'r') as f:
        scene_names = f.read().splitlines()

    preds = {}

    for scene_name in tqdm.tqdm(scene_names[:]):
        scene_id = scene_name[5:]

        masks_path = os.path.join(pred_root_dir, scene_name + masks_template)
        scene_per_mask_feature_path = os.path.join(mask_features_dir, feature_file_template.format(scene_name))

        if not os.path.exists(scene_per_mask_feature_path):
            print('--- SKIPPING ---', scene_per_mask_feature_path)
            continue
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(masks_path=masks_path,
                                                                                               mask_features_path=scene_per_mask_feature_path,
                                                                                               keep_first=keep_first)

        preds[scene_name] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = InstSegEvaluator.evaluate_full(preds, gt_dir, dataset=dataset_type)


# -------------------------------------------------------------------------------------- #
# entry
# -------------------------------------------------------------------------------------- #
def test_pipeline_full_scannet200_mgl(cfg):
    # gt txt path
    gt_folder = f"datasets/constants/scannet/scannet200_instance_gt/validation"
    gt_paths = sorted(os.listdir(gt_folder))

    # gt_paths = ["scene0011_00", "scene0050_00", "scene0231_00", "scene0378_00", "scene0518_00"]

    # voc feature
    l_vocab_features = np.load(f"./ovgraph/evaluation/voc_features/scannet200_clip_l_14_vild.npy")
    h_vocab_features = np.load(f"./ovgraph/evaluation/voc_features/scannet200_clip_h_14.npy")
    # 200 -> 600 id mapper
    label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_200)}.get)

    preds = {}
    for gt_path in tqdm.tqdm(gt_paths[:]):
        scene_id = gt_path.split('.')[0]
        output_folder, output_proposal_folder, output_vis_folder, output_instance_folder, output_predict_folder, output_result_folder, out_file_prefix = parse_scene_path(cfg, scene_id)

        # graph file
        output_file = f"proposed_fusion_{out_file_prefix}_interval-{cfg.merge.interval}-backprj.pkl"
        scene_graph_path = os.path.join(output_predict_folder, output_file)

        if not os.path.isfile(scene_graph_path):
            print("========================================")
            print(f"{scene_graph_path = } doesn't exist!")
            print("========================================")
            return

        scene_pcd = o3d.io.read_point_cloud(os.path.join(cfg.data.data_root, scene_id, f"{scene_id}_vh_clean_2.ply"))
        with open(scene_graph_path, 'rb') as fp:
            scene_graph = pickle.load(fp)
            scene_graph.graph['scan'] = scene_id
            scene_graph.graph['n_pts'] = len(scene_pcd.points)

        pred_features, pred_masks, pred_caption_features, pred_features_list, pred_caption_features_list = get_predicted_instances_list(scene_graph, cfg.eval.feature_name)

        # -------------------------------------------------------------------------------------- #
        # feature selection
        # -------------------------------------------------------------------------------------- #
        # 2. representive and difference
        pred_features = rep_dif_denoise(pred_features_list)
        # pred_caption_features = rep_dif_denoise(pred_caption_features_list)

        # prediction
        similarities1 = pred_features @ l_vocab_features.T
        similarities2 = pred_caption_features @ l_vocab_features.T
        similarities = torch.softmax(torch.from_numpy(similarities1).float(), dim=-1) * cfg.merge.scale_f + torch.softmax(torch.from_numpy(similarities2).float(), dim=-1) * (1 - cfg.merge.scale_f)

        max_ind = np.argmax(similarities.cpu().numpy(), axis=1)
        max_ind_remapped = label_mapper(max_ind)
        pred_classes = max_ind_remapped
        pred_scores = np.ones(pred_classes.shape)

        preds[scene_id] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = InstSegEvaluator.evaluate_full(preds, gt_folder, dataset='scannet200', no_class=False)


def rep_dif_denoise(pred_features_list, neighbor_rate: float = 0.3):
    num_node = len(pred_features_list)
    # 1. compute the main center for each node
    node_mean_center = []
    node_main_center = []
    for cluster in pred_features_list:
        node_mean_center.append(np.mean(np.array(cluster), axis=0))
        node_main_center.append(feats_denoise_dbscan(cluster, 0.01, 50))
    node_mean_center = np.array(node_mean_center)
    node_main_center = np.array(node_main_center)

    # 2. using the node_mean_center to calculate the adj matrix
    node_adj_sim = - node_mean_center @ node_mean_center.T
    for i in range(num_node):
        node_adj_sim[i, i] = 0.0

    # 3. find the nergibors of each node
    neighbor_num = int(neighbor_rate * num_node)
    neighbor_list = find_top_k_neighbors(node_adj_sim, k=neighbor_num)

    # 4. calculated the optimal feature for each node
    optimal_features = []
    for idx, cluster in enumerate(pred_features_list):
        # own main centers
        own_center = node_main_center[idx]
        top_k_centers = neighbor_list[idx]

        # for each feature of a node
        feature_score = []
        for feature in cluster:
            dist_to_center = cal_distance(feature, own_center)
            sum_dist_to_neighbors = np.mean([cal_distance(feature, node_mean_center[nidx]) for nidx in top_k_centers])

            current_distance = dist_to_center - sum_dist_to_neighbors
            feature_score.append(current_distance)

        feature_score = (torch.tensor(feature_score) * 10).softmax(0).cpu()
        optimal_feature = (feature_score.reshape(-1, 1) * np.array(cluster)).sum(0)
        optimal_features.append(optimal_feature)

    return np.array(optimal_features)


def cal_distance(f1, f2, mode='sim'):
    if mode == 'dis':
        return np.linalg.norm(f1 - f2)
    elif mode == 'sim':
        return np.dot(f1, f2)


def find_top_k_neighbors(distance_matrix, k=3):
    """找到每个cluster的top-k近邻"""
    neighbors = []
    for i in range(distance_matrix.shape[0]):
        sorted_indices = np.argsort(distance_matrix[i])
        neighbors.append(sorted_indices[:k])
    return neighbors


def feats_denoise_dbscan(feats, eps=0.01, min_points=50):
    """
        Denoise the features using DBSCAN
        :param feats: Features to denoise.
        :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param min_points: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: Denoised features.
    """
    # Convert to numpy arrays
    feats = np.array(feats)
    # Create DBSCAN object
    clustering = DBSCAN(eps=eps, min_samples=min_points, metric="cosine").fit(feats)

    # Get the labels
    labels = clustering.labels_

    # Count all labels in the cluster
    counter = Counter(labels)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        # Create mask for points in the largest cluster
        largest_mask = labels == most_common_label
        # Apply mask
        largest_cluster_feats = feats[largest_mask]
        lfeats = largest_cluster_feats

        # take the feature with the highest similarity to the mean of the cluster
        if len(lfeats) > 1:
            mean_feats = np.mean(largest_cluster_feats, axis=0)
            # similarity = np.dot(largest_cluster_feats, mean_feats)
            # max_idx = np.argmax(similarity)
            # feats = feats[max_idx]
            lfeats = mean_feats
    else:
        lfeats = np.mean(feats, axis=0)
    return lfeats


def get_predicted_instances(scene_graph, feature_name="feature"):
    node_visual_features = []
    node_caption_features = []
    node_masks = []
    n_pts = scene_graph.graph['n_pts']

    for node in scene_graph.nodes:
        if feature_name == "feature":
            node_visual_features.append(scene_graph.nodes[node]['feature'])
            node_caption_features.append(scene_graph.nodes[node]['feature'])
        else:
            node_visual_features.append(scene_graph.nodes[node]['back_prj_feat_mean'])
            node_caption_features.append(scene_graph.nodes[node]['back_prj_cap_feat_mean'] if "back_prj_cap_feat_mean" in scene_graph.nodes[node] else scene_graph.nodes[node]['back_prj_feat_mean'])  # back_prj_cap_feat_mean

        node_mask = np.zeros(n_pts, dtype=np.bool_)
        node_mask[scene_graph.nodes[node]["pt_indices"]] = True
        node_masks.append(node_mask)

    node_visual_features = np.vstack(node_visual_features)  # (N, n_dims)
    node_caption_features = np.vstack(node_caption_features)  # (N, n_dims)
    node_masks = np.stack(node_masks)  # (N, n_pts)

    return node_visual_features, node_masks, node_caption_features


def get_predicted_instances_list(scene_graph, feature_name="feature"):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--feature_name", type=str, default="feature", choices=['feature', 'representative_features'])
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.feature_name == "feature":  # i.e. mean feature
        print("========================================")
        print("Evaluate using mean feature for each instance")
        print("========================================")
    else:
        print("========================================")
        print(f"Evaluate using {args.K} representative features for each instance.")
        print("This requires the Detic output files to be available.")
        print("========================================")

    test_pipeline_full_scannet200_mgl(cfg)
