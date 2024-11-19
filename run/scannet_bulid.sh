# scannet, sam + ovseg + tap + ovseg
OMP_NUM_THREADS=5 python scripts/scannet_build.py --config configs/scannet/scannet_cropformer_ovseg_tap.yaml
#OMP_NUM_THREADS=6 python scripts/scannet_eval_instance_segment_class_agnostic.py --config configs/scannet/scannet_cropformer_ovseg_tap.yaml
#OMP_NUM_THREADS=6 python scripts/scannet_eval_instance_segment.py --config configs/scannet/scannet_cropformer_ovseg_tap.yaml
#OMP_NUM_THREADS=6 python scripts/scannet_eval_semantic_segment.py --config configs/scannet/scannet_cropformer_ovseg_tap.yaml

# scannet, cropformer + ovseg + tap + clip
#OMP_NUM_THREADS=6 python scripts/scannet_build.py --config configs/scannet/scannet_cropformer_ovseg_clip_tap.yaml
#OMP_NUM_THREADS=6 python scripts/scannet_eval_instance_segment_class_agnostic.py --config configs/scannet/scannet_cropformer_ovseg_clip_tap.yaml
#OMP_NUM_THREADS=6 python scripts/scannet_eval_instance_segment.py --config configs/scannet/scannet_cropformer_ovseg_clip_tap.yaml
#OMP_NUM_THREADS=6 python scripts/scannet_eval_semantic_segment.py --config configs/scannet/scannet_cropformer_ovseg_clip_tap.yaml
