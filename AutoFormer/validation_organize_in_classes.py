import os
from pathlib import Path

# assuming imagenet/ is under AutoFormer

with open('imagenet/validation_ground_truth.txt') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    class_name = line = line.strip()

    Path(f'imagenet/val/{class_name}').mkdir(exist_ok=True)
    command = f'mv imagenet/val/ILSVRC2012_val_{i+1:8d}.JPEG imagenet/val/{class_name}/ILSVRC2012_val_{i+1:8d}.JPEG'

    os.system(command)
    