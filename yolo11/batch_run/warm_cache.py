"""Force-build the train/val label cache for a dataset once, sequentially,
so concurrent seed-jobs for the same dataset never race on cache creation."""
import copy
import sys

from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.build import build_yolo_dataset

data_yaml = sys.argv[1]
data = check_det_dataset(data_yaml)

cfg = copy.deepcopy(DEFAULT_CFG)
cfg.imgsz = 1280

for mode in ("train", "val"):
    img_path = data[mode]
    build_yolo_dataset(cfg, img_path, batch=16, data=data, mode=mode, rect=(mode == "val"), stride=32)
    print(f"warmed {mode} cache for {data_yaml}")
