# replica segment, cropformer + ovseg + tap + ovseg
OMP_NUM_THREADS=6 python scripts/replica_build.py --config configs/replica/replica_cropformer_ovseg_tap.yaml
#OMP_NUM_THREADS=6 python scripts/replica_eval_semantic_segment.py --config configs/replica/replica_cropformer_ovseg_tap.yaml

# replica segment, cropformer + ovseg + tap + clip
#OMP_NUM_THREADS=6 python scripts/replica_build.py --config configs/replica/replica_cropformer_ovseg_clip_tap.yaml
#OMP_NUM_THREADS=6 python scripts/replica_eval_semantic_segment.py --config configs/replica/replica_cropformer_ovseg_clip_tap.yaml
