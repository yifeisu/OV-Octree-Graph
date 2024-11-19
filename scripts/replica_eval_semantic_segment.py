import os
import sys

import torch
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from datasets.constants.replica.replica_constants import REPLICA_CLASSES, REPLICA_CLASSES_VALIDATION, REPLICA_COLOR_MAP_101


def compute_metrics(confmatrix, class_names):
    if isinstance(confmatrix, torch.Tensor):
        confmatrix = confmatrix.cpu().numpy()

    num_classes = len(class_names)
    ious = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1score = np.zeros(num_classes)

    for _idx in range(num_classes):
        ious[_idx] = confmatrix[_idx, _idx] / (max(1, confmatrix[_idx, :].sum() + confmatrix[:, _idx].sum() - confmatrix[_idx, _idx], ))
        recall[_idx] = confmatrix[_idx, _idx] / max(1, confmatrix[_idx, :].sum())
        precision[_idx] = confmatrix[_idx, _idx] / max(1, confmatrix[:, _idx].sum())
        f1score[_idx] = (2 * precision[_idx] * recall[_idx] / max(1, precision[_idx] + recall[_idx]))

    fmiou = (ious * confmatrix.sum(1) / max(1, confmatrix.sum())).sum()
    print(fmiou)

    mdict = {}
    mdict["iou"] = ious.tolist()
    mdict["miou"] = ious.mean().item()
    mdict["fmiou"] = fmiou.item()
    mdict["num_classes"] = num_classes
    mdict["acc0.15"] = (ious > 0.15).sum().item()
    mdict["acc0.25"] = (ious > 0.25).sum().item()
    mdict["acc0.50"] = (ious > 0.50).sum().item()
    mdict["acc0.75"] = (ious > 0.75).sum().item()
    mdict["precision"] = precision.tolist()
    mdict["recall"] = recall.tolist()
    mdict["f1score"] = f1score.tolist()

    return mdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--feature_name", type=str, default="feature", choices=['feature', 'representative_features'])
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    gt_paths = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4", ]
    # ----------------------------------------------------------------------------------------- #
    # evaluation: hovsg
    # ----------------------------------------------------------------------------------------- #
    CLASS_LABELS = REPLICA_CLASSES

    results = []
    results_avg = {
        "scene": [],
        "miou": [],
        "fmiou": [],
        "macc": [],
        "pacc": [], }

    out_file_prefix = f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}"
    for gt_path in gt_paths:
        scene_id = gt_path.split('.')[0]
        output_folder = os.path.join(cfg.data.data_root, scene_id, cfg.data.output_folder)
        output_result_folder = os.path.join(str(output_folder), 'results', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}")

        output_file = f"proposed_fusion_{out_file_prefix}_interval-{cfg.merge.interval}-{cfg.eval.feature_name}-segment-ovseg-db.pkl"
        output_path = os.path.join(output_result_folder, output_file)
        with open(output_path, 'rb') as fp:
            eval_results = pickle.load(fp)
            del eval_results['ious']

        results.append(eval_results)
        for k, v in eval_results.items():
            results_avg[k].append(v)

    results_avg["scene"] = "all"
    for k, v in results_avg.items():
        if k != 'scene':
            results_avg[k] = np.array(v).mean()
    results.append(results_avg)

    df_result = pd.DataFrame(results)
    output_path = os.path.join("./results", f"{cfg.data.type}-{out_file_prefix}-{cfg.eval.feature_name}-segment.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_result.to_csv(output_path, index=False)

    # ----------------------------------------------------------------------------------------- #
    # evaludation: concept-graph
    # ----------------------------------------------------------------------------------------- #
    conf_matrices = {}
    conf_matrix_all = None

    for gt_path in gt_paths:
        scene_id = gt_path.split('.')[0]
        output_folder = os.path.join(cfg.data.data_root, scene_id, cfg.data.output_folder)
        output_result_folder = os.path.join(str(output_folder), 'results', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}")
        output_file = f"proposed_fusion_{out_file_prefix}_interval-{cfg.merge.interval}-{cfg.eval.feature_name}-segment-ovseg-db-concept-graph.pkl"

        output_path = os.path.join(output_result_folder, output_file)
        with open(output_path, 'rb') as fp:
            res = pickle.load(fp)
            conf_matrix = res["conf_matrix"]
            keep_index = res["keep_index"]

        conf_matrices[scene_id] = {
            "conf_matrix": conf_matrix,
            "keep_index": keep_index,
        }
        if conf_matrix_all is None:
            conf_matrix_all = conf_matrix
        else:
            conf_matrix_all = conf_matrix_all + conf_matrix

    # Remove the rows and columns that are not in keep_class_index
    conf_matrices["all"] = {
        "conf_matrix": conf_matrix_all,
        "keep_index": conf_matrix_all.sum(axis=1).nonzero().reshape(-1),
    }

    results = []
    for scene_id, res in conf_matrices.items():
        conf_matrix = res["conf_matrix"]
        keep_index = res["keep_index"]
        conf_matrix = conf_matrix[keep_index, :][:, keep_index]
        keep_class_names = [CLASS_LABELS[i] for i in keep_index]

        mdict = compute_metrics(conf_matrix, keep_class_names)
        results.append(
            {
                "scene_id": scene_id,
                "miou": mdict["miou"] * 100.0,
                # "macc": mdict["macc"] * 100.0,
                "mrecall": np.mean(mdict["recall"]) * 100.0,
                "mprecision": np.mean(mdict["precision"]) * 100.0,
                "mf1score": np.mean(mdict["f1score"]) * 100.0,
                "fmiou": mdict["fmiou"] * 100.0,
            })

    df_result = pd.DataFrame(results)
    output_path = os.path.join("./results", f"{cfg.data.type}-{out_file_prefix}-{cfg.eval.feature_name}-mIOU.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_result.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
