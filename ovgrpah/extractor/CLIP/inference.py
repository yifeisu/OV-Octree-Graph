import dataclasses

import cv2
import numpy as np
import open_clip
import supervision as sv
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

from supervision.draw.color import ColorPalette

# tap
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack

# tap checkpoint
model_type = "tap_vit_h"
checkpoint = "./pretrained_weights/tap_vit_h_v1_1.pkl"
concept_weights = "./pretrained_weights/merged_2560.pkl"


class CLIPHExtractor:
    def __init__(
            self,
            cfg,
            device='cuda',
    ):
        self.device = device

        # 1. Initialize the cliph model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def __call__(self, image, mask, xyxy, conf):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            mask=mask,
            xyxy=xyxy,
            class_id=np.zeros_like(conf).astype(int),
        )

        # obtain the region feature and text feature using ovseg
        image_feats = compute_clip_features(image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)

        return image_feats

    def forward_region(self, region):
        if not isinstance(region, list):
            preprocessed_image = self.clip_preprocess(region).unsqueeze(0).to(self.device)
        else:
            preprocessed_image = torch.cat([self.clip_preprocess(ref).unsqueeze(0).to(self.device) for ref in region], dim=0)

        crop_feat = self.clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)

        return crop_feat.cpu().numpy()


class CLIPTapExtractor:
    def __init__(
            self,
            cfg,
            device='cuda',
    ):
        self.device = device

        # 1. Initialize the cliph model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # 2.  Initialize the TAP model
        self.tap = model_registry[model_type](checkpoint=checkpoint).to(device)
        self.tap.concept_projector.reset_weights(concept_weights)
        self.tap.text_decoder.reset_cache(max_batch_size=64)

    def __call__(self, image, mask, xyxy, conf):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            mask=mask,
            xyxy=xyxy,
            class_id=np.zeros_like(conf).astype(int),
        )

        # obtain the region feature and text feature using ovseg
        image_feats = compute_clip_features(image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)

        return image_feats

    def forward_region(self, image_rgb, region_mask):
        # 1. obtain the visual feat from ovseg clip
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
        tokenized_text = self.clip_tokenizer(captions).to(self.device)
        region_caption_feat = self.clip_model.encode_text(tokenized_text)
        region_caption_feat /= region_caption_feat.norm(dim=-1, keepdim=True)

        return region_clip_feat, region_caption_feat, captions


@torch.no_grad()
def compute_clip_features(image, detections, clip_model, clip_preprocess, device):
    image = Image.fromarray(image)

    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    cropped_image = []
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

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
    preprocessed_image = torch.cat([clip_preprocess(img).unsqueeze(0).to(device) for img in cropped_image], dim=0)
    # turn the list of feats into np matrices
    crop_feat = clip_model.encode_image(preprocessed_image)
    crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
    image_feats = crop_feat.cpu().numpy()

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


def vis_result_fast(
        image: np.ndarray,
        detections: sv.Detections,
        classes,
        color=ColorPalette.default(),
        instance_random_color: bool = False,
        draw_bbox: bool = True,
) -> np.ndarray:
    """
    Annotate the image with the detection results.
    This is fast but of the same resolution of the input image, thus can be blurry.
    """
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color=color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color=color
    )

    if hasattr(detections, 'confidence') and hasattr(detections, 'class_id'):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")

    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels
