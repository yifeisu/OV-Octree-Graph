import warnings

import numpy as np

warnings.warn("deprecated", DeprecationWarning)

# Ignore warnings in obj loader
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

import sys

import json
from PIL import Image

from torchmetrics.functional import pairwise_cosine_similarity

from tqdm import tqdm
import networkx as nx
import pycocotools.mask as mask_util

# ----------------------------------------------------------------------------------------- #
# our lib
# ----------------------------------------------------------------------------------------- #
# lib
from .building import build_detector, build_extractor
from ..stream.datasets_common import get_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer
from ..utils.scene_path import parse_scene_path
from ..utils.utils import *
from ..utils.metric import *
from ..utils.merging import *
from ..structure.scene_graph import *

IGNORE_INDEX = -1

def masks_to_rle(instances: Instances):
    """
    https://github.com/facebookresearch/detectron2/issues/347
    """
    pred_masks_rle = []
    for mask in instances.pred_masks:
        rle = mask_util.encode(np.asfortranarray(mask.numpy()))
        rle['counts'] = rle['counts'].decode('utf-8')
        pred_masks_rle.append(rle)
    instances.pred_masks_rle = pred_masks_rle
    return instances


def rle_to_masks(instances: Instances):
    instances.pred_masks = torch.from_numpy(np.stack([mask_util.decode(rle) for rle in instances.pred_masks_rle]))
    return instances


def build_scene_graph2(scene_pcd, anchor_objects, vocabs=None, vocab_features=None):
    N = len(anchor_objects)
    scene_pcd = copy.deepcopy(scene_pcd)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    #
    for i in range(N):
        pt_indices = torch.nonzero(anchor_objects[i].mask_ids).cpu().numpy().astype(np.int32)
        pcd = scene_pcd.select_by_index(pt_indices)
        pts = np.asarray(pcd.points)
        G.nodes[i]['center'] = np.mean(pts, axis=0)
        G.nodes[i]['pt_indices'] = pt_indices
        G.nodes[i]['feature'] = anchor_objects[i].mask_feature.cpu().numpy()
        G.nodes[i]['detections'] = (anchor_objects[i].image_idx, anchor_objects[i].image_path, torch.sum(anchor_objects[i].mask_ids))
        G.nodes[i]['contained'] = anchor_objects[i].contained
        G.nodes[i]['containing'] = anchor_objects[i].containing
        G.nodes[i]['mask_center'] = anchor_objects[i].mask_center
        G.nodes[i]['merge_num'] = anchor_objects[i].merge_num
        if vocabs is not None and vocab_features is not None:
            closest_category_ids = (vocab_features @ anchor_objects[i].mask_feature).topk(5).indices
            top5_closest_categories = [vocabs[i.item()] for i in closest_category_ids]
            G.nodes[i]['top5_vocabs'] = top5_closest_categories

    return G


def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors


class ScanNetSceneGraph:
    def __init__(
            self,
            cfg,
            scene_id,
            process_idx=0,
            # data related
            data_device='cpu',
            model_device='cuda:0',
    ):
        self.extractor = None
        self.detector = None

        self.cfg = cfg
        self.scene_id = scene_id
        self.process_idx = process_idx
        self.model_device = model_device
        self.data_device = data_device

        # ----------------------------------------------------------------------------------------- #
        # obtain the scene data property, rgbd stream, point-cloud, and gt semantic.
        # ----------------------------------------------------------------------------------------- #
        scene_config_path = os.path.join(cfg.data.config_root, f"{scene_id}.yaml")
        # 0. build the rgbd stream dataset
        self.rgbd = get_dataset(
            dataconfig=scene_config_path,
            start=cfg.data.start,
            end=cfg.data.end,
            stride=cfg.data.stride,
            basedir=cfg.data.data_root,
            sequence=scene_id,
            desired_height=cfg.data.desired_height,
            desired_width=cfg.data.desired_width,
            device=data_device,
            dtype=torch.float,
        )

        # 1. read the gt ply
        gt_ply_path = os.path.join(cfg.data.data_root, scene_id, f"{scene_id}_vh_clean_2.ply")
        self.gt_point = o3d.io.read_point_cloud(str(gt_ply_path))
        self.gt_xyz = np.asarray(self.gt_point.points)

        # 2. read the gt semantic
        gt_segs_path = os.path.join(cfg.data.data_root, scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        gt_aggr_path = os.path.join(cfg.data.data_root, scene_id, f"{scene_id}.aggregation.json")
        # 2.1 load segments file, over-segmentation id
        with open(gt_segs_path) as f:
            segments = json.load(f)
            seg_indices = np.array(segments["segIndices"])
        # 2.2 load aggregations file, instance with cat_id
        with open(gt_aggr_path) as f:
            aggregation = json.load(f)
            seg_groups = np.array(aggregation["segGroups"])
        labels_pd = pd.read_csv("datasets/constants/scannet/scannetv2-labels.combined.tsv", sep="\t", header=0)
        self.semantic_gt_200, self.semantic_gt_20, self.gt_cat_ids, self.gt_masks = self.get_gt_semantic(seg_groups, seg_indices, labels_pd)

        # ----------------------------------------------------------------------------------------- #
        # define and create some scene path
        # ----------------------------------------------------------------------------------------- #
        self.output_folder, self.output_proposal_folder, self.output_vis_folder, self.output_instance_folder, self.output_predict_folder, self.output_result_folder, self.out_file_prefix = parse_scene_path(cfg, scene_id)

    def get_gt_semantic(self, seg_groups, seg_indices, labels_pd, ignore_index=(0,)):
        # parse each instance
        gt_cat_ids, gt_masks = [], []
        semantic_gt200 = np.ones((self.gt_xyz.shape[0]), dtype=np.int16) * IGNORE_INDEX
        semantic_gt20 = np.ones((self.gt_xyz.shape[0]), dtype=np.int16) * IGNORE_INDEX
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

    def build_detector(self):
        return build_detector(self.cfg, self.model_device)

    def build_extractor(self):
        return build_extractor(self.cfg, self.model_device)

    def generate_2d_proposals(self):
        # ----------------------------------------------------------------------------------------- #
        # generate 2d proposals;
        # ----------------------------------------------------------------------------------------- #
        # 1. build the extracter
        self.detector = self.build_detector()

        # 2. for each rgb image, generate the 2d proposals
        for idx in tqdm(range(len(self.rgbd)), position=self.process_idx, desc=f"{self.scene_id}-{self.process_idx}", ncols=160):
            frame_id = os.path.basename(self.rgbd.color_paths[idx]).split('.')[0]

            _color = self.rgbd.get_rgb(idx)
            # run inference
            instances = self.detector(_color)

            # save visualization
            if self.cfg.detector.save_vis:  # very slow
                colors = make_colors()
                # Img and prediction
                img = cv2.cvtColor(_color, cv2.COLOR_BGR2RGB)
                #
                instances_vis = copy.deepcopy(instances)
                instances_vis.remove('scores')
                visualizer = Visualizer(img, instance_mode=ColorMode.IMAGE)
                vis_output = visualizer.draw_instance_predictions(predictions=instances_vis)
                color_mask = vis_output.get_image()
                vis_im = color_mask
                output_path = os.path.join(self.output_vis_folder, f"{frame_id}.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))

            # save instance
            masks_to_rle(instances)
            instances.remove('pred_masks')

            output_path = os.path.join(self.output_proposal_folder, f"{frame_id}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(instances, f)

    def compute_2d_feature(self):
        # ----------------------------------------------------------------------------------------- #
        # generate 2d proposals;
        # ----------------------------------------------------------------------------------------- #
        # 1. build the extracter
        self.extractor = self.build_extractor()

        # 2. for each rgb image, generate the 2d proposals
        for idx in tqdm(range(len(self.rgbd)), position=self.process_idx, desc=f"{self.scene_id}-{self.process_idx}"):
            _color = self.rgbd.get_rgb(idx)

            # read detected output
            frame_id = os.path.basename(self.rgbd.color_paths[idx]).split('.')[0]
            detected_instance = read_detectron_instances(os.path.join(self.output_proposal_folder, f'{frame_id}.pkl'), rle_to_mask=True)

            # parse mask and xyxy
            mask = detected_instance.pred_masks.numpy()  # (M, H, W)
            xyxy = detected_instance.pred_xyxys.numpy()  # (M, H, W)
            conf = detected_instance.scores.numpy()  # (M,)

            # run inference
            proposal_feats = self.extractor(_color, mask, xyxy, conf)
            detected_instance.pred_box_features = torch.from_numpy(proposal_feats)

            # save instance
            detected_instance.remove('pred_masks')

            output_path = os.path.join(self.output_instance_folder, f"{frame_id}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(detected_instance, f)

    @staticmethod
    def get_edge_template(w, h, bar=20):
        template_image = np.zeros([h, w], dtype=np.uint8)
        template_image[:bar, :] = 1
        template_image[:, :bar] = 1
        template_image[h - bar:, :] = 1
        template_image[:, w - bar:] = 1

        return template_image

    def obtain_3d_segment(self):
        # ----------------------------------------------------------------------------------------- #
        # 1. build the mask-graph; no merging;
        # ----------------------------------------------------------------------------------------- #
        N = len(self.gt_point.points)
        visibility_count = np.zeros((N,), dtype=np.int32)
        interval = min((len(self.rgbd) // 100 * 100) // 4, len(self.rgbd))

        anchor_objects = InstanceList()
        # for each frame
        for idx in tqdm(range(len(self.rgbd.color_paths)), position=self.process_idx, desc=f"{self.scene_id}-{self.process_idx}", ncols=150):
            frame_id = os.path.basename(self.rgbd.color_paths[idx]).split('.')[0]
            # read detected output.
            detect_result_path = os.path.join(self.output_instance_folder, f'{frame_id}.pkl')
            detect_output = read_detectron_instances(detect_result_path, rle_to_mask=True)
            pred_scores, pred_masks, pred_features = detect_output.scores.numpy(), detect_output.pred_masks.numpy(), detect_output.pred_box_features.to(self.data_device)  # (M,)
            pred_masks = resolve_overlapping_masks(pred_masks, pred_scores, device=self.data_device)
            pred_features = F.normalize(pred_features, dim=1, p=2)

            # filter segment on the edge
            valid_mask = []
            template_image = self.get_edge_template(pred_masks.shape[2], pred_masks.shape[1], 20)
            for ii, pred_mask in enumerate(pred_masks):
                if np.sum(template_image * pred_mask) / np.sum(pred_mask) >= 0.75:
                    continue
                valid_mask.append(ii)
            valid_mask = np.array(valid_mask)
            pred_scores, pred_masks, pred_features = pred_scores[valid_mask], pred_masks[valid_mask], pred_features[valid_mask]
            template_image = self.get_edge_template(pred_masks.shape[2], pred_masks.shape[1], 10)
            pred_mask_center = np.zeros_like(pred_scores, dtype=np.int32)
            for ii, pred_mask in enumerate(pred_masks):
                if np.sum(template_image * pred_mask) / np.sum(pred_mask) <= 0.0:
                    pred_mask_center[ii] = 1

            # backproject the scene gt point
            _, depth_im, cam_intr, pose = self.rgbd[idx]

            cam_intr = cam_intr[:3, :3]
            pcd = copy.deepcopy(self.gt_point).transform(np.linalg.inv(pose))
            scene_pts = np.asarray(pcd.points)
            projected_pts = compute_projected_pts(scene_pts, cam_intr)
            visibility_mask = compute_visibility_mask(scene_pts, projected_pts, depth_im, depth_thresh=self.cfg.merge.depth_thresh)
            visibility_count[visibility_mask] += 1
            #
            masked_pts = compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks)  # (M, N)
            masked_pts = torch.from_numpy(masked_pts).to(self.data_device)
            # filter small object
            mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M, )
            valid_mask = mask_area >= self.cfg.merge.size_thresh
            masked_pts, pred_features, pred_scores, pred_mask_center = masked_pts[valid_mask], pred_features[valid_mask], pred_scores[valid_mask], pred_mask_center[valid_mask]
            if masked_pts.shape[0] <= 0:
                continue

            # * preprocess each segment * #
            # 1. dbscan denoise
            masked_pts = masked_pts.long()
            filter_pt_by_dbscan_noise(copy.deepcopy(self.gt_xyz), masked_pts)
            masked_pts = masked_pts.bool()

            # * regester all detected segments * #
            detect_segments = InstanceList()
            for ii in range(masked_pts.shape[0]):
                segment = Segment(
                    image_idx=idx,
                    image_path=self.rgbd.color_paths[idx],
                    mask_ids=masked_pts[ii],
                    mask_feature=pred_features[ii],
                    mask_score=pred_scores[ii],
                    mask_center=pred_mask_center[ii]
                )
                detect_segments.append(segment)

            # * light merging * #
            if len(anchor_objects) == 0:
                anchor_objects.extend(detect_segments)
            else:
                iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix2(detect_segments, anchor_objects, visibility_mask)  # [K, M]
                semantic_similarity_matrix = pairwise_cosine_similarity(detect_segments.get_stacked_values_torch('mask_feature'), anchor_objects.get_stacked_values_torch('mask_feature'))  # [1, N]
                adjacency_matrix = (iou_matrix >= 0.4) & (semantic_similarity_matrix >= 0.8)
                to_move = []
                for ii in range(adjacency_matrix.shape[0]):
                    if torch.sum(adjacency_matrix[ii]) > 0:
                        assign_idx = torch.where(adjacency_matrix[ii])[0]
                        assign_idx = assign_idx[iou_matrix[ii, assign_idx].argmax()]
                        anchor_objects[assign_idx] = anchor_objects[assign_idx].myadd(detect_segments[ii])
                    else:
                        to_move.append(ii)
                anchor_objects.extend(detect_segments.slice_by_indices(to_move))

            # sub-graph merging
            if (idx == (len(self.rgbd) - 1)) or ((idx % interval == 0) and idx != 0):
                # ===================== filter instances based on visibility and size =====================#
                anchor_objects = filter_pt_by_dbscan_noise2(copy.deepcopy(self.gt_xyz), anchor_objects)
                anchor_objects = filter_pt_by_visibility_count2(anchor_objects, visibility_count, 0.1)
                anchor_objects = filter_by_instance_size_no_media2(anchor_objects, size_thresh=self.cfg.merge.size_thresh, median=True)

                # ===================== undersegment filtering =====================#
                iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix_self2(anchor_objects)
                semantic_similarity_matrix = pairwise_cosine_similarity(anchor_objects.get_stacked_values_torch('mask_feature'), anchor_objects.get_stacked_values_torch('mask_feature'))
                precision_matrix_bool = precision_matrix >= 0.8  # find the segments contained in each instance
                to_delete = []
                for ii in range(precision_matrix_bool.shape[0]):
                    if torch.sum(precision_matrix_bool[ii]) > 2:  # containing at least one segment;
                        precision_matrix_bool[ii][ii] = 0.0
                        contained_segment_ii = torch.where(precision_matrix_bool[ii])[0]
                        if torch.sum(semantic_similarity_matrix[ii][contained_segment_ii] <= 0.9) / contained_segment_ii.shape[0] > 0.6 and anchor_objects[ii].merge_num <= 10:
                            to_delete.append(ii)
                        precision_matrix_bool[ii][ii] = 1.0
                anchor_objects = anchor_objects.delete_index(to_delete)

                # -------------------------------------------------------------------------------------- #
                # liner decay & decouple
                # -------------------------------------------------------------------------------------- #
                final_segments = InstanceList()
                to_move = []
                segment_centers = anchor_objects.get_stacked_values_torch('mask_center')  # [M,1]
                segment_centers_idx = torch.where(segment_centers[:, 0])[0].cpu().numpy().tolist()
                centered_segments = anchor_objects.slice_by_indices(segment_centers_idx)
                iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix3(centered_segments, anchor_objects)  # [K, M]
                semantic_similarity_matrix = pairwise_cosine_similarity(centered_segments.get_stacked_values_torch('mask_feature'), anchor_objects.get_stacked_values_torch('mask_feature'))  # [1, N]
                recall_matrix_bool, precision_matrix_bool, iou_matrix_bool, semantic_similarity_matrix_bool = recall_matrix >= 0.6, precision_matrix >= 0.6, iou_matrix >= 0.3, semantic_similarity_matrix >= 0.9
                adjacency_matrix = (recall_matrix_bool | precision_matrix_bool) & iou_matrix_bool & semantic_similarity_matrix_bool
                to_move.extend(segment_centers_idx)
                for ii in range(adjacency_matrix.shape[0]):
                    to_move.extend(torch.where(adjacency_matrix[ii])[0].cpu().numpy().tolist())
                to_move = list(set(to_move))
                final_segments.extend(anchor_objects.slice_by_indices(to_move))
                anchor_objects = anchor_objects.delete_index(to_move)

                # * aggregate the anchor instance * #
                ii = 0
                recall_t, iou_t, similarity_t = 0.7, 0.35, 0.9
                while True:
                    stop = True
                    iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix_self2(final_segments)
                    semantic_similarity_matrix = pairwise_cosine_similarity(final_segments.get_stacked_values_torch('mask_feature'), final_segments.get_stacked_values_torch('mask_feature'))
                    recall_matrix_bool, iou_matrix_bool, semantic_similarity_matrix_bool = recall_matrix >= max(recall_t, 0.6), iou_matrix >= max(iou_t, 0.25), semantic_similarity_matrix >= max(similarity_t, 0.85)
                    adjacency_matrix = (recall_matrix_bool | iou_matrix_bool) & semantic_similarity_matrix_bool
                    adjacency_matrix = adjacency_matrix | adjacency_matrix.T
                    connected_components = find_connected_components(adjacency_matrix)
                    connected_components = [cluster for cluster in connected_components if len(cluster) > 1]
                    # * aggregate the anchor instance * #
                    if connected_components:
                        stop = False
                        merged_segments = InstanceList()
                        for cluster in connected_components:
                            merged_segment = final_segments.pop_and_aggregate(cluster, False)
                            merged_segments.append(merged_segment)
                        # remove the aggregated segments, post-delete, avoiding the index error
                        to_delete = set()
                        for cluster in connected_components:
                            for segment_idx in cluster:
                                to_delete.add(segment_idx)
                        final_segments = final_segments.delete_index(to_delete)
                        final_segments.extend(merged_segments)
                    recall_t -= 0.02
                    iou_t -= 0.02
                    similarity_t -= 0.01
                    ii += 1
                    if stop and ii >= 5:
                        break

                    final_segments = filter_pt_by_dbscan_noise2(copy.deepcopy(self.gt_xyz), final_segments)
                    final_segments = filter_by_instance_size_no_media2(final_segments, size_thresh=self.cfg.merge.size_thresh, median=True)

                ii = 0
                recall_t, iou_t, similarity_t = 0.7, 0.25, 0.85
                while True:
                    stop = True
                    iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix_self2(anchor_objects)
                    semantic_similarity_matrix = pairwise_cosine_similarity(anchor_objects.get_stacked_values_torch('mask_feature'), anchor_objects.get_stacked_values_torch('mask_feature'))
                    iou_matrix = iou_matrix * (semantic_similarity_matrix + 0.1)
                    recall_matrix = recall_matrix * (semantic_similarity_matrix + 0.1)

                    recall_matrix_bool, iou_matrix_bool, semantic_similarity_matrix_bool = recall_matrix >= max(recall_t, 0.5), iou_matrix >= max(iou_t, 0.15), semantic_similarity_matrix >= max(similarity_t, 0.8)
                    adjacency_matrix = (recall_matrix_bool | iou_matrix_bool) & semantic_similarity_matrix_bool
                    adjacency_matrix = adjacency_matrix | adjacency_matrix.T
                    connected_components = find_connected_components(adjacency_matrix)
                    connected_components = [cluster for cluster in connected_components if len(cluster) > 1]
                    # * aggregate the anchor instance * #
                    if connected_components:
                        stop = False
                        merged_segments = InstanceList()
                        for cluster in connected_components:
                            merged_segment = anchor_objects.pop_and_aggregate(cluster, False)
                            merged_segments.append(merged_segment)
                        # remove the aggregated segments, post-delete, avoiding the index error
                        to_delete = set()
                        for cluster in connected_components:
                            for segment_idx in cluster:
                                to_delete.add(segment_idx)
                        anchor_objects = anchor_objects.delete_index(to_delete)
                        anchor_objects.extend(merged_segments)

                    recall_t -= 0.02
                    iou_t -= 0.01
                    similarity_t -= 0.005
                    ii += 1
                    if stop and ii >= 10:
                        break

                    anchor_objects = filter_pt_by_dbscan_noise2(copy.deepcopy(self.gt_xyz), anchor_objects)
                    anchor_objects = filter_by_instance_size_no_media2(anchor_objects, size_thresh=self.cfg.merge.size_thresh, median=True)

                anchor_objects.extend(final_segments)
                ii = 0
                recall_t, iou_t, similarity_t = 0.7, 0.25, 0.85
                while True:
                    stop = True
                    iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix_self2(anchor_objects)
                    semantic_similarity_matrix = pairwise_cosine_similarity(anchor_objects.get_stacked_values_torch('mask_feature'), anchor_objects.get_stacked_values_torch('mask_feature'))

                    iou_matrix = iou_matrix * (semantic_similarity_matrix + 0.1)
                    recall_matrix = recall_matrix * (semantic_similarity_matrix + 0.1)

                    recall_matrix_bool, iou_matrix_bool, semantic_similarity_matrix_bool = recall_matrix >= max(recall_t, 0.5), iou_matrix >= max(iou_t, 0.15), semantic_similarity_matrix >= max(similarity_t, 0.8)
                    adjacency_matrix = (recall_matrix_bool | iou_matrix_bool) & semantic_similarity_matrix_bool
                    adjacency_matrix = adjacency_matrix | adjacency_matrix.T
                    connected_components = find_connected_components(adjacency_matrix)
                    connected_components = [cluster for cluster in connected_components if len(cluster) > 1]
                    # * aggregate the anchor instance * #
                    if connected_components:
                        stop = False
                        merged_segments = InstanceList()
                        for cluster in connected_components:
                            merged_segment = anchor_objects.pop_and_aggregate(cluster, False)
                            merged_segments.append(merged_segment)
                        # remove the aggregated segments, post-delete, avoiding the index error
                        to_delete = set()
                        for cluster in connected_components:
                            for segment_idx in cluster:
                                to_delete.add(segment_idx)
                        anchor_objects = anchor_objects.delete_index(to_delete)
                        anchor_objects.extend(merged_segments)

                    recall_t -= 0.04
                    iou_t -= 0.02
                    similarity_t -= 0.01
                    ii += 1
                    if stop and ii >= 5:
                        break

                    anchor_objects = filter_pt_by_dbscan_noise2(copy.deepcopy(self.gt_xyz), anchor_objects)
                    anchor_objects = filter_by_instance_size_no_media2(anchor_objects, size_thresh=self.cfg.merge.size_thresh, median=True)

        # -------------------------------------------------------------------------------------- #
        # -1. post-processing
        # -------------------------------------------------------------------------------------- #
        anchor_objects = filter_pt_by_dbscan_noise2(copy.deepcopy(self.gt_xyz), anchor_objects)
        anchor_objects = post_processing2(copy.deepcopy(self.gt_point), anchor_objects)

        scene_graph = build_scene_graph2(self.gt_point, anchor_objects, vocabs=None, vocab_features=None)
        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}.pkl"
        graph_output_folder = os.path.join(self.output_predict_folder, output_file)
        with open(graph_output_folder, 'wb') as fp:
            pickle.dump(scene_graph, fp)
            print(f"Saved to {graph_output_folder}")

    def graph_back_projcection(self):
        # -------------------------------------------------------------------------------------- #
        # back projection
        # -------------------------------------------------------------------------------------- #
        # * build the extracter * #
        self.extractor = self.build_extractor()

        # * load the built graph * #
        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}.pkl"
        scene_graph_path = os.path.join(self.output_predict_folder, output_file)

        if not os.path.isfile(scene_graph_path):
            print("========================================")
            print(f"{scene_graph_path = } doesn't exist!")
            print("========================================")
            return
        with open(scene_graph_path, 'rb') as fp:
            scene_graph = pickle.load(fp)
            scene_graph.graph['scan'] = self.scene_id
            scene_graph.graph['n_pts'] = len(self.gt_xyz)

        # * for each object, compute its corresponding views * #
        to_delete = []
        frame_segment_prompt = {}
        for node in tqdm(scene_graph.nodes, position=self.process_idx, desc=f"{self.scene_id}-{self.process_idx}"):
            # the node's index mask in point clouds.
            segment_mask = np.zeros(scene_graph.graph['n_pts'], dtype=np.bool_)
            segment_mask[scene_graph.nodes[node]["pt_indices"]] = True

            # the corresponding 2d view of a 3d segment
            meta_list = scene_graph.nodes[node]["detections"]
            corres_scan_idx_val, corres_obj_bbox, corres_obj_mask, corres_obj_area = [], [], [], []

            # for each frame, judge whether it contains the segment
            corres_scan_idx = list(meta_list[0])
            for view_index in corres_scan_idx:
                # get the corresponding color, depth and pose
                image_bgr, depth_im, cam_intr, pose = self.rgbd[view_index]

                # 1. transform the 3d points from world coordinates to camera coordinates, and get the instance point cloud;
                segment_pts = np.asarray(copy.deepcopy(self.gt_point).transform(np.linalg.inv(pose)).points)[segment_mask]  # (n, 3)
                # 2. convert the 3d points in to 2d plane
                cam_intr = cam_intr[:3, :3]
                projected_pts = compute_projected_pts(segment_pts, cam_intr)  # (n, 2)
                visibility_mask = compute_visibility_mask(segment_pts, projected_pts, depth_im, depth_thresh=self.cfg.merge.depth_thresh)  # [n,]
                if np.sum(visibility_mask) <= 0:
                    continue

                # 3.1 record the frame id
                corres_scan_idx_val.append(view_index)
                # 3.2 segment's bbox
                padding, (image_height, image_width, _) = 20, image_bgr.shape
                x_min, y_min, x_max, y_max = np.min(projected_pts[visibility_mask, 0]), np.min(projected_pts[visibility_mask, 1]), np.max(projected_pts[visibility_mask, 0]), np.max(projected_pts[visibility_mask, 1])
                left_padding, top_padding, right_padding, bottom_padding = min(padding, x_min), min(padding, y_min), min(padding, image_width - x_max), min(padding, image_height - y_max)
                x_min, y_min, x_max, y_max = x_min - left_padding, y_min - top_padding, x_max + right_padding, y_max + bottom_padding
                corres_obj_bbox.append(np.array([x_min, y_min, x_max, y_max]))
                # corres_obj_area.append((x_max - x_min) * (y_max - y_min))
                corres_obj_area.append(np.sum(visibility_mask))
                # 3.2 segment's mask
                obj_mask = np.zeros_like(depth_im, dtype=np.uint8)
                hull = cv2.convexHull(projected_pts[visibility_mask].astype(np.int32), clockwise=True, returnPoints=True)
                cv2.fillPoly(obj_mask, [hull], [255, ])
                corres_obj_mask.append(obj_mask)

            if not corres_scan_idx_val:
                to_delete.append(node)
                continue

            # * cut the min 20% off * #
            corres_scan_area = np.array(corres_obj_area)
            corres_scan_area_idx = np.argsort(corres_scan_area)[::-1]
            cut_idx = max(int(0.9 * len(corres_scan_area_idx)), 1)  # max(int(0.9 * len(corres_scan_area_idx)), 1)
            corres_scan_area_idx = corres_scan_area_idx[:cut_idx]

            # node property
            scene_graph.nodes[node]["back_prj_feat_list"] = []
            scene_graph.nodes[node]["back_prj_cap_feat_list"] = []
            scene_graph.nodes[node]["back_prj_cap_list"] = []
            scene_graph.nodes[node]["corres_scan_idx_val"] = []

            # * record the segment's bbox of mask in each corresponding frames * #
            for sort_idx in corres_scan_area_idx:
                view_idex = corres_scan_idx_val[sort_idx]
                if view_idex not in frame_segment_prompt:
                    frame_segment_prompt[view_idex] = {
                        'contained_segment_idx': [],
                        'segment_bbox': [],
                        'segment_mask': [],
                        'segment_feature': None,
                        'segment_caption_feat': None,
                        'segment_caption': None,
                    }

                frame_segments = frame_segment_prompt[view_idex]
                # record segment id in the nx graph and segment property
                frame_segments['contained_segment_idx'].append(node)
                frame_segments['segment_bbox'].append(corres_obj_bbox[sort_idx])
                frame_segments['segment_mask'].append(corres_obj_mask[sort_idx])
                scene_graph.nodes[node]["corres_scan_idx_val"].append(view_idex)

        # ----------------------------------------------------------------------------------------- #
        # * for each frame, compute the backprojection segment's feature * #
        # ----------------------------------------------------------------------------------------- #
        # delete the void node
        for node in to_delete:
            scene_graph.remove_node(node)

        with torch.no_grad():
            for frame_idx, frame_segment in tqdm(frame_segment_prompt.items(), position=self.process_idx, desc=f"{self.scene_id}-{self.process_idx}"):
                image_bgr, depth_im, cam_intr, pose = self.rgbd[frame_idx]
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                if self.cfg.extractor.type == 'cliph':
                    segment_bboxes = frame_segment['segment_bbox']
                    cropped_images = []
                    for segment_bbox in segment_bboxes:
                        cropped_images.append(Image.fromarray(image_rgb).crop(segment_bbox.tolist()))
                    crop_feat = self.extractor.forward_region(cropped_images)
                    frame_segment['segment_feature'] = crop_feat
                    frame_segment['segment_cap_feat'] = crop_feat
                    frame_segment['segment_caption'] = ['None'] * crop_feat.shape[0]

                elif self.cfg.extractor.type == 'ovseg':
                    segment_masks = np.array(frame_segment['segment_mask'])
                    region_clip_feat = self.extractor.forward_region(image_rgb, segment_masks)
                    frame_segment['segment_feature'] = region_clip_feat.cpu().numpy()
                    frame_segment['segment_cap_feat'] = region_clip_feat.cpu().numpy()
                    frame_segment['segment_caption'] = ['None'] * region_clip_feat.shape[0]

                elif self.cfg.extractor.type == 'alpha_clip':
                    segment_masks = np.array(frame_segment['segment_mask'])
                    region_clip_feat = self.extractor.forward_region(image_rgb, segment_masks)
                    frame_segment['segment_feature'] = region_clip_feat
                    frame_segment['segment_cap_feat'] = region_clip_feat
                    frame_segment['segment_caption'] = ['None'] * region_clip_feat.shape[0]

                elif self.cfg.extractor.type == 'ovseg_tap':
                    segment_masks = np.array(frame_segment['segment_mask'])
                    region_clip_feat, region_caption_feat, region_caption = self.extractor.forward_region(image_rgb, segment_masks, self.cfg.extractor.back_feat)
                    frame_segment['segment_feature'] = region_clip_feat.cpu().numpy()
                    frame_segment['segment_cap_feat'] = region_caption_feat.cpu().numpy()
                    frame_segment['segment_caption'] = region_caption

                for idx, node in enumerate(frame_segment['contained_segment_idx']):
                    scene_graph.nodes[node]["back_prj_feat_list"].append(frame_segment['segment_feature'][idx])
                    scene_graph.nodes[node]["back_prj_cap_feat_list"].append(frame_segment['segment_cap_feat'][idx])
                    scene_graph.nodes[node]["back_prj_cap_list"].append(frame_segment['segment_caption'][idx])

        # * assign the computed feature into the nx graph * #
        for node in scene_graph.nodes:
            # the node's index mask in point clouds.
            assert len(scene_graph.nodes[node]["corres_scan_idx_val"]) == len(scene_graph.nodes[node]["back_prj_feat_list"])
            assert len(scene_graph.nodes[node]["corres_scan_idx_val"]) == len(scene_graph.nodes[node]["back_prj_cap_feat_list"])
            scene_graph.nodes[node]["back_prj_feat_mean"] = np.stack(scene_graph.nodes[node]["back_prj_feat_list"]).mean(0)
            scene_graph.nodes[node]["back_prj_cap_feat_mean"] = np.stack(scene_graph.nodes[node]["back_prj_cap_feat_list"]).mean(0)

            # integrate all view caption into one description
            # scene_graph.nodes[node]["back_prj_cap"] = gpt_post_process(scene_graph.nodes[node]["back_prj_cap_list"])

        # * save the processed secen graph * #
        output_file = f"proposed_fusion_{self.out_file_prefix}_{self.cfg.extractor.back_feat}_interval-{self.cfg.merge.interval}-backprj.pkl"
        scene_graph_path = os.path.join(self.output_predict_folder, output_file)

        with open(scene_graph_path, 'wb') as fp:
            pickle.dump(scene_graph, fp)
            print(f"Saved to {scene_graph_path}")

    def evaluate_retrieval(self):
        # ----------------------------------------------------------------------------------------- #
        # process the scannet 200, classes
        # ----------------------------------------------------------------------------------------- #
        id_mapping_path = f"datasets/constants/scannet/scannetv2-labels.combined.tsv"
        df = pd.read_csv(id_mapping_path, sep="\t")
        id_mapping = {cat_id: i for i, cat_id in enumerate(VALID_CLASS_IDS_200)}

        CLASS_LABELS = CLASS_LABELS_200
        VALID_CLASS_IDS = VALID_CLASS_IDS_200
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        # uncountable categories: i.e. "wall", "floor" and their subcategories + "ceiling"
        INVALID_IDS = [df['id'][i] for i in range(df.shape[0]) if df['nyuClass'][i] in ("wall", "floor") or df['nyuClass'][i] == "ceiling"]
        VAL_IDS = set(i for i in CLASS_LABELS_200_VALIDATION if i not in INVALID_IDS)

        # ----------------------------------------------------------------------------------------- #
        # class feature
        # ----------------------------------------------------------------------------------------- #
        vocab_features = np.load(f"{self.cfg.model.voc_feature}")
        cat_id_to_feature = {cat_id: vocab_features[id_mapping[cat_id]] for cat_id in VALID_CLASS_IDS}

        print(f"{len(CLASS_LABELS_200_VALIDATION) = }")
        print(f"After filtering uncountable categories, {len(VAL_IDS) = }")

        scene_pcd = copy.deepcopy(self.gt_point)

        output_file = f"proposed_fusion_{self.cfg.model.type}_interval-{self.cfg.merge.interval}.pkl"
        scene_graph_path = os.path.join(self.output_predict_folder, output_file)

        if not os.path.isfile(scene_graph_path):
            print("========================================")
            print(f"{scene_graph_path = } doesn't exist!")
            print("========================================")
            return

        with open(scene_graph_path, 'rb') as fp:
            scene_graph = pickle.load(fp)
            scene_graph.graph['scan'] = self.scene_id
            scene_graph.graph['n_pts'] = len(scene_pcd.points)

        gt_path = f"datasets/constants/scannet/scannet200_instance_gt/validation/{self.scene_id}.txt"
        gt_instance_ids = np.loadtxt(gt_path, dtype=np.int64)
        gt_cat_ids, gt_masks = get_gt_instances(gt_instance_ids, valid_ids=VAL_IDS)
        pred_features, pred_masks, pred_caption_features, _, _ = get_predicted_instances(scene_graph, self.cfg.eval.feature_name)

        output_file = f"proposed_fusion_{self.cfg.model.type}_interval-{self.cfg.merge.interval}-{self.cfg.eval.feature_name}-retrieval.pkl"
        output_path = os.path.join(self.output_result_folder, output_file)
        #
        ap_results = compute_ap_for_each_scan(pred_features, pred_caption_features, pred_masks, gt_cat_ids, gt_masks, cat_id_to_feature, self.cfg.merge.scale_f)
        with open(output_path, 'wb') as fp:
            pickle.dump(ap_results, fp)
        print(f"Processed {self.scene_id = }. Results saved to: {output_path}")

    def evaluate_segmentation_20_opt(self):
        # ----------------------------------------------------------------------------------------- #
        # load the feature of the class name, scannet 200 or replica
        # ----------------------------------------------------------------------------------------- #
        # uncountable categories: i.e. "wall", "floor" and their subcategories + "ceiling"
        ignore = [-1]
        for _id, name in enumerate(CLASS_LABELS_20):
            if "wall" in name or "floor" in name or "ceiling" in name or "door" in name or "window" in name:
                ignore.append(_id)

        # class feature for scannet 200
        v_vocab_features = np.load(f"{self.cfg.extractor.voc_feature}")
        t_vocab_features = np.load(f"./tiger/evaluation/voc_features/scannet20_clip_l_14_vild.npy")

        # ----------------------------------------------------------------------------------------- #
        # prepare the predction results, preds (N,)
        # ----------------------------------------------------------------------------------------- #
        output_file = f"proposed_fusion_{self.out_file_prefix}_{self.cfg.extractor.back_feat}_interval-{self.cfg.merge.interval}-backprj.pkl"
        scene_graph_path = os.path.join(self.output_predict_folder, output_file)
        if not os.path.isfile(scene_graph_path):
            print("========================================")
            print(f"{scene_graph_path = } doesn't exist!")
            print("========================================")
            return

        with open(scene_graph_path, 'rb') as fp:
            scene_graph = pickle.load(fp)
            scene_graph.graph['scan'] = self.scene_id
            scene_graph.graph['n_pts'] = len(self.gt_xyz)

        pred_features, pred_masks, pred_caption_features, pred_features_list, pred_caption_features_list = get_predicted_instances(scene_graph, feature_name=self.cfg.eval.feature_name)

        # 2. representive and difference
        pred_features = rep_dif_denoise(pred_features_list)
        # pred_caption_features = rep_dif_denoise(pred_caption_features_list)

        # compute the class logits
        pred_class_sim1 = pred_features @ v_vocab_features.T  # (num_objects, num_classes)
        pred_class_sim2 = pred_caption_features @ t_vocab_features.T  # (num_objects, num_classes)
        # scale
        pred_class = (torch.softmax(torch.from_numpy(pred_class_sim1), dim=-1) * self.cfg.merge.scale_f + torch.softmax(torch.from_numpy(pred_class_sim2), dim=-1) * (1 - self.cfg.merge.scale_f)).argmax(-1)  # (num_objects,)

        #
        predction_20 = np.ones((self.gt_xyz.shape[0]), dtype=np.int16) * IGNORE_INDEX
        for i in range(len(pred_class)):
            predction_20[pred_masks[i]] = pred_class[i]

        # * simple visualize * #
        class2color = np.array([value for key, value in SCANNET_COLOR_MAP_20.items()]) / 255.0
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(self.gt_xyz)
        # gt
        semantic_gt_20 = copy.deepcopy(self.semantic_gt_20)
        semantic_gt_20[semantic_gt_20 == IGNORE_INDEX] = 20
        pred_pcd.colors = o3d.utility.Vector3dVector(class2color[semantic_gt_20])
        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}-{self.cfg.eval.feature_name}-visualize-gt.ply"
        output_path = os.path.join(self.output_result_folder, output_file)
        o3d.io.write_point_cloud(output_path, pred_pcd)
        # pred
        predction_20_visaul = copy.deepcopy(predction_20)
        predction_20_visaul[predction_20_visaul == IGNORE_INDEX] = 20
        pred_pcd.colors = o3d.utility.Vector3dVector(class2color[predction_20_visaul])
        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}-{self.cfg.eval.feature_name}-visualize.ply"
        output_path = os.path.join(self.output_result_folder, output_file)
        o3d.io.write_point_cloud(output_path, pred_pcd)
        # save the graph with topk
        for idx, node in enumerate(scene_graph.nodes):
            scene_graph.nodes[node]['top1_cls'] = CLASS_LABELS_200[pred_class[idx]]
        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}-backprj-top1.pkl"
        scene_graph_path = os.path.join(self.output_predict_folder, output_file)
        with open(scene_graph_path, 'wb') as fp:
            pickle.dump(scene_graph, fp)
            print(f"Saved to {scene_graph_path}")

        # ----------------------------------------------------------------------------------------- #
        # after obtaining the gt point class (N,) and predict point class (N,), use the evaluate
        # function in concept-graph or openscen to calculate the miou, fmiou and macc.
        # ----------------------------------------------------------------------------------------- #
        # * hovsg evaluation * #
        # print number of unique labels in the ground truth and predicted pointclouds
        print("Number of unique labels in the GT pcd: ", len(np.unique(self.semantic_gt_20)))
        print("Number of unique labels in the pred pcd ", len(np.unique(predction_20)))
        print(ignore)

        # knn interpolation & concat coords and labels for predicied pcd
        # prediction knn input
        coords_labels = np.zeros((len(predction_20), 4))
        coords_labels[:, :3] = self.gt_xyz.copy()
        coords_labels[:, -1] = predction_20.copy()
        coords_labels = coords_labels[coords_labels[:, -1] != -1]
        # gt knn input
        coords_gt = np.zeros((len(self.semantic_gt_20), 4))
        coords_gt[:, :3] = self.gt_xyz.copy()
        coords_gt[:, -1] = self.semantic_gt_20.copy()

        match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
        predction_20 = match_pc[:, -1].reshape(-1, 1).astype(np.int32)

        print("################ {} ################".format(self.scene_id))
        predction_20 = predction_20.reshape(-1, 1)
        semantic_gt_20 = self.semantic_gt_20.reshape(-1, 1)

        ious = per_class_IU(predction_20, semantic_gt_20, ignore=ignore)
        print("per class iou: ", ious)
        miou = mean_IU(predction_20, semantic_gt_20, ignore=ignore, class_id_name=CLASS_LABELS_20)
        print("miou: ", miou)
        fmiou = frequency_weighted_IU(predction_20, semantic_gt_20, ignore=ignore)
        print("fmiou: ", fmiou)
        macc = mean_accuracy(predction_20, semantic_gt_20, ignore=ignore)
        print("macc: ", macc)
        pacc = pixel_accuracy(predction_20, semantic_gt_20, ignore=ignore)
        print("pacc: ", pacc)
        print("#######################################")

        results = {
            "scene": self.scene_id,
            "ious": ious,
            "miou": miou,
            "fmiou": fmiou,
            "macc": macc,
            "pacc": pacc,
        }

        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}-{self.cfg.eval.feature_name}-segment-{self.cfg.extractor.back_feat}.pkl"
        output_path = os.path.join(self.output_result_folder, output_file)
        with open(output_path, 'wb') as fp:
            pickle.dump(results, fp)

        output_file = f"proposed_fusion_{self.out_file_prefix}_interval-{self.cfg.merge.interval}-{self.cfg.eval.feature_name}-segment-{self.cfg.extractor.back_feat}.txt"
        output_path = os.path.join(self.output_result_folder, output_file)
        with open(output_path, 'w') as f:
            f.write(str(results))
        print(f"Processed {self.scene_id = }. Results saved to: {output_path}")

        return 0


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
    node_adj_sim = -node_mean_center @ node_mean_center.T
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

        feature_score = (torch.tensor(feature_score) * 25).softmax(0).cpu().numpy()
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

    # print(counter)
    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        # Create mask for points in the largest cluster
        largest_mask = labels == most_common_label
        print(np.sum(largest_mask), labels.shape[0])

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
