import sys
import cv2
import numpy as np

from PIL import Image

import torch
import ovclip
import open_clip

import supervision as sv
# tap
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack

# tap checkpoint
model_type = "tap_vit_h"
checkpoint = "./pretrained_weights/tap_vit_h_v1_1.pkl"
concept_weights = "./pretrained_weights/merged_2560.pkl"

# ovseg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

sys.path.insert(0, './3rdparty/ovseg')
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


class OVsegExtractor:
    def __init__(
            self,
            cfg,
            ovseg_config,
            sam_variant='sam',
            class_set='none',
            device='cuda',
    ):
        self.device = device
        self.class_set = class_set
        self.sam_variant = sam_variant

        # 1. Initialize the ovseg model
        self.demo = VisualizationDemo(setup_cfg(ovseg_config, [f"MODEL.WEIGHTS", "pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth"]))

    def __call__(self, image, mask, xyxy, conf):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros_like(conf).astype(int),
            mask=mask, )

        # classes = ['item']
        # annotated_image, labels = vis_result_fast(image, detections, classes, instance_random_color=True)

        # obtain the region feature and text feature using ovseg
        image_feats_ovseg = compute_ovseg_features(image_rgb, detections, self.demo, self.device)

        return image_feats_ovseg

    def forward_region(self, image_rgb, region_mask):
        mask_pt = torch.from_numpy(region_mask).to(self.device).float().squeeze()
        region_clip_feat = self.demo.run_on_regions(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask_pt, self.device)

        return region_clip_feat


class OVsegTapExtractor:
    def __init__(
            self,
            ovseg_config,
            device='cuda',
    ):
        self.device = device

        # 1. Initialize the ovseg model
        self.demo = VisualizationDemo(setup_cfg(ovseg_config, [f"MODEL.WEIGHTS", "pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth"]))

        # 2.  Initialize the TAP model
        self.tap = model_registry[model_type](checkpoint=checkpoint).to(device)
        self.tap.concept_projector.reset_weights(concept_weights)
        self.tap.text_decoder.reset_cache(max_batch_size=64)

        # 3. Initialize the clip-h model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def __call__(self, image, mask, xyxy, conf):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros_like(conf).astype(int),
            mask=mask, )

        # obtain the region feature and text feature using ovseg
        image_feats_ovseg = compute_ovseg_features(image_rgb, detections, self.demo, self.device)
        # image_feats_tap = compute_tap_features(image_rgb, detections, self.tap, self.demo, self.device)

        return image_feats_ovseg

    def forward_region(self, image_rgb, region_mask, back_feat='ovseg'):
        if back_feat == 'ovseg':
            # 1. obtain the visual feat from ovseg clip
            mask_pt = torch.from_numpy(region_mask).to(self.device).float().squeeze()
            region_clip_feat = self.demo.run_on_regions(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask_pt, self.device)
        elif back_feat == 'clip':
            # 1.2
            image = Image.fromarray(image_rgb)
            padding = 20
            cropped_image = []
            for mask in region_mask:
                yy, xx = np.where(mask)
                x_min, x_max, y_min, y_max = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
                # Check and adjust padding to avoid going beyond the image borders
                image_width, image_height = image.size
                left_padding = min(padding, x_min)
                top_padding = min(padding, y_min)
                right_padding = min(padding, image_width - x_max)
                bottom_padding = min(padding, image_height - y_max)

                # Apply the adjusted padding
                x_min -= left_padding
                y_min -= top_padding
                x_max += right_padding
                y_max += bottom_padding
                cropped_image.append(image.crop((x_min, y_min, x_max, y_max)))
            # Get the preprocessed image for clip from the crop
            preprocessed_image = torch.cat([self.clip_preprocess(img).unsqueeze(0).to(self.device) for img in cropped_image], dim=0)
            # turn the list of feats into np matrices
            crop_feat = self.clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            region_clip_feat = crop_feat
        else:
            raise NotImplementedError

        # 2. obtain the caption and text feat from tap, image_rgb
        # 2.1 resize the image
        img_list, img_scales = im_rescale(image_rgb, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, image_rgb.shape[:2]

        img_batch = im_vstack(img_list, fill_value=self.tap.pixel_mean_value, size=(1024, 1024))
        inputs = self.tap.get_inputs({"img": img_batch})
        inputs.update(self.tap.get_features(inputs))

        # 2.1 prepare the bbox prompt
        bbox_list = []
        for mask in region_mask:
            yy, xx = np.where(mask)
            x_min, x_max, y_min, y_max = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
            bbox_list.append([[x_min, y_min, 2], [x_max, y_max, 3]])

        inputs["points"] = np.array(bbox_list, "float32")
        inputs["points"][:, :, :2] *= np.array(img_scales, "float32")

        # 2.2 decode outputs for the box prompt.
        outputs = self.tap.get_outputs(inputs)

        # 2.3 select final mask.
        iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
        iou_score[:, 1:] -= 1000.0  # Penalize the score of loose points.
        mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)

        # 2.4 upscale masks to the original image resolution.
        # iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
        # masks = self.tap.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        # masks = masks[..., : input_size[0], : input_size[1]]
        # masks = self.tap.upscale_masks(masks, original_size).gt(0).cpu().numpy()

        # 2.5 predict concepts and generate captions.
        sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
        # concepts, scores = self.tap.predict_concept(sem_embeds[mask_index])
        captions = self.tap.generate_text(sem_tokens[mask_index])

        # 2.6 embedding the caption text
        text_inputs = ovclip.tokenize(captions).to(self.device)

        region_caption_feat = self.demo.predictor.model.clip_adapter.clip_model.encode_text(text_inputs)
        region_caption_feat = region_caption_feat / region_caption_feat.norm(dim=-1, keepdim=True)

        return region_clip_feat, region_caption_feat, captions


@torch.no_grad()
def compute_tap_features(image_rgb, detections, tap, ovseg, device):
    img_list, img_scales = im_rescale(image_rgb, scales=[1024], max_size=1024)
    input_size, original_size = img_list[0].shape, image_rgb.shape[:2]

    img_batch = im_vstack(img_list, fill_value=tap.pixel_mean_value, size=(1024, 1024))
    inputs = tap.get_inputs({"img": img_batch})
    inputs.update(tap.get_features(inputs))

    # 2.1 prepare the bbox prompt
    region_mask = detections.mask
    bbox_list = []
    for mask in region_mask:
        yy, xx = np.where(mask)
        x_min, x_max, y_min, y_max = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
        bbox_list.append([[x_min, y_min, 2], [x_max, y_max, 3]])

    inputs["points"] = np.array(bbox_list, "float32")
    inputs["points"][:, :, :2] *= np.array(img_scales, "float32")

    # 2.2 decode outputs for the box prompt.
    outputs = tap.get_outputs(inputs)

    # 2.3 select final mask.
    iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
    iou_score[:, 1:] -= 1000.0  # Penalize the score of loose points.
    mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)

    # 2.4 upscale masks to the original image resolution.
    # iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
    # masks = self.tap.upscale_masks(masks[:, None], img_batch.shape[1:-1])
    # masks = masks[..., : input_size[0], : input_size[1]]
    # masks = self.tap.upscale_masks(masks, original_size).gt(0).cpu().numpy()

    # 2.5 predict concepts and generate captions.
    sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
    # concepts, scores = self.tap.predict_concept(sem_embeds[mask_index])
    captions = tap.generate_text(sem_tokens[mask_index])

    # 2.6 embedding the caption text
    text_inputs = ovclip.tokenize(captions).to(device)
    region_caption_feat = ovseg.predictor.model.clip_adapter.clip_model.encode_text(text_inputs)
    region_caption_feat = region_caption_feat / region_caption_feat.norm(dim=-1, keepdim=True)

    return region_caption_feat.cpu().numpy()


@torch.no_grad()
def compute_ovseg_features(image, detections, ovseg, device):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask_np = detections.mask
    mask_pt = torch.from_numpy(mask_np).to(device).float()
    region_clip_feat = ovseg.run_on_regions(image_bgr, mask_pt, device)

    # turn the list of feats into np matrices
    image_feats = region_clip_feat.cpu().numpy()

    return image_feats


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub
