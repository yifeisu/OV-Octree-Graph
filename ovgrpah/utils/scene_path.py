import os


def parse_scene_path(cfg, scene_id):
    # scene outputdir
    output_folder = os.path.join(cfg.data.data_root, scene_id, cfg.data.output_folder)

    # scene 2d proposal outputdir
    output_proposal_folder = os.path.join(str(output_folder), 'proposals', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}")

    # scene visualization outputdir
    output_vis_folder = os.path.join(str(output_folder), 'proposals_vis', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}")

    # scene 2d proposal w/ features outputdir
    output_instance_folder = os.path.join(str(output_folder), 'instances', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}")

    # scene graph predction outputdir
    output_predict_folder = os.path.join(str(output_folder), 'graph_prediction', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}")

    # scene graph evaluation outputdir
    output_result_folder = os.path.join(str(output_folder), 'results', f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_proposal_folder, exist_ok=True)
    os.makedirs(output_vis_folder, exist_ok=True)
    os.makedirs(output_instance_folder, exist_ok=True)
    os.makedirs(output_predict_folder, exist_ok=True)
    os.makedirs(output_result_folder, exist_ok=True)

    # graph_output_file
    out_file_prefix = f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}"

    return output_folder, output_proposal_folder, output_vis_folder, output_instance_folder, output_predict_folder, output_result_folder, out_file_prefix
