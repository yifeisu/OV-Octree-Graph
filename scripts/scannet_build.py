import open3d as o3d
import warnings

warnings.filterwarnings('ignore')

import os
import argparse
import multiprocessing
from time import perf_counter
from omegaconf import OmegaConf

import torch

from ovgraph.scene import ScanNetSceneGraph


def mgl_graph(dataset, video, idx, cfg, available_devices):
    device = available_devices.get()
    print(f"Processing {video = }, {idx = }, {device = } \n")

    # set cuda device
    torch.cuda.set_device(f"cuda:{device}")

    # build a scene graph
    scene_graph = ScanNetSceneGraph(
        cfg,
        scene_id=video,
        process_idx=idx,
        model_device=f"cuda",  # f"cuda:{device}"
    )

    scene_graph.generate_2d_proposals()
    scene_graph.compute_2d_feature()
    scene_graph.obtain_3d_segment()
    scene_graph.graph_back_projcection()
    # scene_graph.evaluate_retrieval()
    # scene_graph.evaluate_segmentation_20_opt()

    torch.cuda.empty_cache()
    available_devices.put(device)


def main():
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--num_gpus", type=int, default=-1, help="-1 means using all the visible gpus")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # 1. test
    if args.test:
        scene_graph = ScanNetSceneGraph(
            cfg,
            scene_id="scene0011_00",
            process_idx=0,
            model_device=f"cuda",  # f"cuda:{device}"
        )

        scene_graph.generate_2d_proposals()
        scene_graph.compute_2d_feature()
        scene_graph.obtain_3d_segment()
        scene_graph.graph_back_projcection()
        # scene_graph.evaluate_retrieval()
        # scene_graph.evaluate_segmentation_20_opt()

    # 2. multi processing
    else:
        dataset = cfg.data.data_root
        # all scenes;
        videos = [i for i in sorted(os.listdir(dataset))[:]]
        # concept-fusion scenes;
        # videos = ["scene0011_00", "scene0050_00", "scene0231_00", "scene0378_00", "scene0518_00"]

        if args.num_gpus == -1:
            max_workers = min(torch.cuda.device_count(), len(videos))
        else:
            max_workers = min(args.num_gpus, torch.cuda.device_count())

        print(f"{dataset = }")
        print(f"{len(videos) = }")
        print(f"{max_workers = }")

        _available_devices = multiprocessing.Manager().Queue()
        for i in range(max_workers):
            _available_devices.put(i)

        tic = perf_counter()
        p = multiprocessing.Pool(max_workers)
        for idx, scene in enumerate(videos):
            p.apply_async(mgl_graph, args=(dataset, scene, idx, cfg, _available_devices,))

        p.close()
        p.join()
        print(f"Process {len(videos)} takes {perf_counter() - tic}s")


if __name__ == "__main__":
    main()
