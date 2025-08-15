import os, json
import tensorflow as tf
from pathlib import Path
from utils.config import load_yaml, ensure_dirs
from utils.image import load_image_bytes
from utils.text import TextEncoder
from utils.logging import get_logger

logger = get_logger("tfrecords")

def _bytes(x): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
def _int64(x): return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))

def write_shard(examples, out_path):
    with tf.io.TFRecordWriter(out_path) as w:
        for ex in examples:
            feat = {
              "image": _bytes(ex["image"]),
              "ids": _bytes(tf.io.serialize_tensor(tf.constant(ex["ids"], dtype=tf.int32)).numpy()),
              "length": _int64(len(ex["ids"]))
            }
            w.write(tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString())

def main():
    paths = load_yaml("configs/paths.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    ensure_dirs(paths["tfrecords"])
    enc = TextEncoder(paths.get("tokenizer"), vocab_size=model_cfg["decoder"]["vocab_size"])

    for name in ["coco", "flickr30k"]:
        src = Path(paths["hf_jsonl_dir"]) / f"{name}.jsonl"
        out = Path(paths["tfrecords"]) / f"{name}-00000-of-00001.tfrecord"
        examples = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                img_bytes = load_image_bytes(rec["image"], target_size=model_cfg["vision"]["img_size"])
                ids = enc.encode(rec["paragraph"])
                examples.append({"image": img_bytes, "ids": ids})
        write_shard(examples, str(out))
        logger.info(f"Wrote {len(examples)} → {out}")

if __name__ == "__main__":
    main()
# This script converts image-caption pairs from JSONL format to TFRecord format.
# It uses a tokenizer to encode captions and saves images as bytes.