import sys
import argparse
from typing import List

import clip
import open_clip
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from datasets.constants.scannet.scannet200_constants import CLASS_LABELS_200, CLASS_LABELS_20
from datasets.constants.replica.replica_constants import REPLICA_CLASSES

# ovseg module;
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, './3rdparty/ovseg')
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo


@torch.no_grad()
def get_clip_feature(clip_model, text_label, normalize=True, prompt_fn=lambda x: f"a picture of {x}", device="cpu"):
    print("computing text features...")

    if isinstance(text_label, str):
        text_inputs = clip.tokenize(prompt_fn(text_label)).to(device)
    else:
        text_inputs = torch.cat([clip.tokenize(prompt_fn(c)) for c in text_label]).to(device)

    # in case the vocab is too large
    chunk_size = 100
    chunks = torch.split(text_inputs, chunk_size, dim=0)
    # extract
    text_features = []
    for i, chunk in enumerate(chunks):
        chunk_feature = clip_model.encode_text(chunk).detach()
        text_features.append(chunk_feature)

    text_features = torch.cat(text_features, dim=0)

    if normalize:
        text_features = F.normalize(text_features, dim=-1).detach().cpu().numpy()
    return text_features


@torch.no_grad()
def compute_ovseg_features(image, detections, ovseg, classes, device):
    text_feats = []
    # Get the text feature
    for idx in range(len(detections.xyxy)):
        class_id = detections.class_id[idx]
        text_feat = ovseg.predictor.model.clip_adapter.get_text_features([classes[class_id]])

        text_feat = text_feat.cpu().numpy()
        text_feats.append(text_feat)

    # turn the list of feats into np matrices
    text_feats = np.concatenate(text_feats, axis=0)

    return text_feats


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


# ----------------------------------------------------------------------------------------- #
# text prompt feature extractor
# ----------------------------------------------------------------------------------------- #
class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError()


class PredefinedPromptExtractor(PromptExtractor):
    def __init__(self, templates: List[str]):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)

        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ovseg related
    parser.add_argument(
        "--config-file",
        default="./3rdparty/ovseg/configs/ovseg_swinB_vitL_demo.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth"],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------- #
    # an extracter
    # ----------------------------------------------------------------------------------------- #
    extractor = PredefinedPromptExtractor(VILD_PROMPT)

    with torch.no_grad():
        # ----------------------------------------------------------------------------------------- #
        # build the CLIP ViT-H-14 model
        # ----------------------------------------------------------------------------------------- #
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # #* CLASS_LABELS_200 *# #
        text_feats = []
        for class_name in CLASS_LABELS_200:
            tokenized_text = clip_tokenizer(f"a picture of {class_name}").to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

            text_feat = text_feat.detach().cpu().numpy()
            text_feats.append(text_feat)
        text_feats = np.concatenate(text_feats, axis=0)
        # save
        np.save(f"scannet200_clip_h_14.npy", text_feats)

        # #* CLASS_LABELS_20 *# #
        text_feats = []
        for class_name in CLASS_LABELS_20:
            tokenized_text = clip_tokenizer(f"a picture of {class_name}").to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

            text_feat = text_feat.detach().cpu().numpy()
            text_feats.append(text_feat)
        text_feats = np.concatenate(text_feats, axis=0)
        # save
        np.save(f"scannet20_clip_h_14.npy", text_feats)

        # #* REPLICA_CLASSES *# #
        text_feats = []
        for class_name in REPLICA_CLASSES:
            tokenized_text = clip_tokenizer(f"a picture of {class_name}").to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

            text_feat = text_feat.detach().cpu().numpy()
            text_feats.append(text_feat)
        text_feats = np.concatenate(text_feats, axis=0)

        # save
        np.save(f"replica101_clip_h_14.npy", text_feats)

        # ----------------------------------------------------------------------------------------- #
        # build the CLIP ViT-B-32 model, for Detic
        # ----------------------------------------------------------------------------------------- #
        # 1. simple prompts
        clip_model, _ = clip.load('ViT-B/32', 0)

        # #* scanenet *# #
        vocab_features = get_clip_feature(clip_model=clip_model, text_label=CLASS_LABELS_200, normalize=True, device=f"cuda:{0}")
        np.save(f"scannet200_clip_b_32.npy", vocab_features)

        # #* scanenet *# #
        vocab_features = get_clip_feature(clip_model=clip_model, text_label=CLASS_LABELS_20, normalize=True, device=f"cuda:{0}")
        np.save(f"scannet20_clip_b_32.npy", vocab_features)

        # * replica * #
        vocab_features = get_clip_feature(clip_model=clip_model, text_label=REPLICA_CLASSES, normalize=True, device=f"cuda:{0}")
        np.save(f"replica101_clip_b_32.npy", vocab_features)

        # 2. multiple prompts
        # #* scanenet *# #
        vocab_features = extractor(CLASS_LABELS_200, clip_model).cpu().numpy()
        np.save(f"scannet200_clip_b_32_vild.npy", vocab_features)

        # #* scanenet *# #
        vocab_features = extractor(CLASS_LABELS_20, clip_model).cpu().numpy()
        np.save(f"scannet20_clip_b_32_vild.npy", vocab_features)

        # * replica * #
        vocab_features = extractor(REPLICA_CLASSES, clip_model).cpu().numpy()
        np.save(f"replica101_clip_b_32_vild.npy", vocab_features)

        # ----------------------------------------------------------------------------------------- #
        # build the OVseg text model, for ovseg
        # ----------------------------------------------------------------------------------------- #
        # Initialize the ovseg model
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)

        # * scanenet * #
        class_feats = demo.predictor.model.clip_adapter.get_text_features(CLASS_LABELS_200)
        np.save(f"scannet200_clip_l_14_vild.npy", class_feats.cpu().numpy())

        # * scanenet * #
        class_feats = demo.predictor.model.clip_adapter.get_text_features(CLASS_LABELS_20)
        np.save(f"scannet20_clip_l_14_vild.npy", class_feats.cpu().numpy())

        # * replica * #
        class_feats = demo.predictor.model.clip_adapter.get_text_features(REPLICA_CLASSES)
        np.save(f"replica101_clip_l_14_vild.npy", class_feats.cpu().numpy())
