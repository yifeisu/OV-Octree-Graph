# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import os
import sys

import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

sys.path.insert(0, './3rdparty/Entity/Entityv2/CropFormer/')

from mask2former import add_maskformer2_config
from demo_cropformer.predictor import VisualizationDemo

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


class CropFormerDetector:
    def __init__(
            self,
            config,
            device='cuda',
    ):
        self.device = device

        # 2. Initialize the CropFormer model
        self.cropformer = VisualizationDemo(setup_cfg(config, [f"MODEL.WEIGHTS", "pretrained_weights/Mask2Former_hornet_3x_576d0b.pth"]))

        # 3. meta_data
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = ['other', ]

    def __call__(self, image):
        det_ins = self.cropformer.run_on_image(image)
        pred_masks = det_ins["instances"].pred_masks
        pred_scores = det_ins["instances"].scores

        # select by confidence threshold
        selected_indexes = (pred_scores >= 0.5)
        selected_scores = pred_scores[selected_indexes]
        selected_masks = pred_masks[selected_indexes]

        # detectron2 instance
        instance = Instances(image_size=(image.shape[0], image.shape[0]))
        instance.scores = selected_scores.cpu()
        instance.pred_masks = selected_masks.bool().cpu()

        # compute mask's xyxy
        xyxys = []
        for mask in instance.pred_masks:
            yy, xx = torch.where(mask > 0)
            xyxy = torch.tensor([
                torch.min(xx),
                torch.min(yy),
                torch.max(xx),
                torch.max(yy),
            ])
            xyxys.append(xyxy)
        instance.pred_xyxys = torch.stack(xyxys)

        return instance
