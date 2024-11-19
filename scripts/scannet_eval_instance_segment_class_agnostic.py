import open3d as o3d

import os
import pickle

import tqdm
import argparse

import clip

import torch
import numpy as np
from omegaconf import OmegaConf

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
    def evaluate_full(preds, scene_gt_dir, dataset, output_file='temp_output.txt', no_class=True):
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


def test_pipeline_full_scannet200_mgl(cfg):
    # gt txt path
    gt_folder = f"datasets/constants/scannet/scannet200_instance_gt/validation"
    gt_paths = sorted(os.listdir(gt_folder))

    # gt_paths = ["scene0011_00", "scene0050_00", "scene0231_00", "scene0378_00", "scene0518_00"]

    # voc feature
    vocab_features = np.load(f"./ovgraph/evaluation/voc_features/scannet20_clip_l_14_vild.npy")
    t_vocab_features = np.load(f"./ovgraph/evaluation/voc_features/scannet20_clip_l_14_vild.npy")

    # 200 -> 600 id mapper
    label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_200)}.get)

    preds = {}
    for gt_path in tqdm.tqdm(gt_paths[:]):
        scene_id = gt_path.split('.')[0]
        output_folder, output_proposal_folder, output_vis_folder, output_instance_folder, output_predict_folder, output_result_folder, out_file_prefix = parse_scene_path(cfg, scene_id)

        # graph file
        output_file = f"proposed_fusion_{out_file_prefix}_interval-{cfg.merge.interval}.pkl"
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
        pred_features, pred_masks, pred_caption_features = get_predicted_instances(scene_graph, 'feature')  # cfg.eval.feature_name

        # prediction
        similarities1 = pred_features @ vocab_features.T
        similarities2 = pred_caption_features @ t_vocab_features.T
        similarities = torch.softmax(torch.from_numpy(similarities1).float(), dim=-1) * cfg.merge.scale_f + torch.softmax(torch.from_numpy(similarities2).float(), dim=-1) * (1 - cfg.merge.scale_f)

        max_ind = np.argmax(similarities.cpu().numpy(), axis=1)
        max_ind_remapped = label_mapper(max_ind)
        pred_classes = max_ind_remapped
        pred_scores = np.ones(pred_classes.shape)

        preds[scene_id] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = InstSegEvaluator.evaluate_full(preds, gt_folder, dataset='scannet200', no_class=True)


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
