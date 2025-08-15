import os
from datasets import load_dataset
from utils.logging import get_logger
from utils.config import load_yaml, ensure_dirs

logger = get_logger("download")

def save_images(dataset, out_dir, prefix):
    ensure_dirs(out_dir)
    n = 0
    for idx, ex in enumerate(dataset):
        img = ex["image"]
        p = os.path.join(out_dir, f"{prefix}_{idx:06d}.jpg")
        img.save(p, format="JPEG")
        n += 1
    logger.info(f"Saved {n} images → {out_dir}")

def main():
    paths = load_yaml("configs/paths.yaml")
    raw_root = paths["raw_images"]

    # COCO (sample for dev; remove select for full set)
    coco = load_dataset("HuggingFaceM4/COCO", split="train").shuffle(42).select(range(1000))
    save_images(coco, os.path.join(raw_root, "coco"), "coco")

    # Flickr30k
    flickr = load_dataset("nlphuji/flickr30k", split="train").shuffle(42).select(range(1000))
    save_images(flickr, os.path.join(raw_root, "flickr30k"), "flickr")

if __name__ == "__main__":
    main()
# This script downloads sample images from COCO and Flickr30k datasets
# and saves them to the specified directory structure.