def build_detector(cfg, device):
    if cfg.detector.type == 'cropformer':
        from ..detector.CropFormer.inference import CropFormerDetector
        return CropFormerDetector(cfg.detector.config_file, device)

    else:
        raise NotImplementedError


def build_extractor(cfg, device):
    # ----------------------------------------------------------------------------------------- #
    # register the muti-modal extracter
    # ----------------------------------------------------------------------------------------- #
    if cfg.extractor.type == 'none':
        return None
    elif cfg.extractor.type == 'ovseg':
        from ..extractor.OVSeg.inference import OVsegExtractor
        return OVsegExtractor(cfg.extractor.config_file, device)
    elif cfg.extractor.type == 'ovseg_tap':
        from ..extractor.OVSeg.inference import OVsegTapExtractor
        return OVsegTapExtractor(cfg.extractor.config_file, device)
    elif cfg.extractor.type == 'cliph':
        from ..extractor.CLIP.inference import CLIPHExtractor
        return CLIPHExtractor(cfg.detector.config_file, device)
    elif cfg.extractor.type == 'clip_tap':
        from ..extractor.CLIP.inference import CLIPTapExtractor
        return CLIPTapExtractor(cfg.extractor.config_file, device)
    else:
        raise NotImplementedError
