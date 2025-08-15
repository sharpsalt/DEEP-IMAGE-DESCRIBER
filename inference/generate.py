import os, yaml, numpy as np, tensorflow as tf
from PIL import Image
from utils.config import load_yaml
from utils.text import TextEncoder
from models.model import ImageParagraph
from inference.beam_search import generate_ids

def load_image(path, sz):
    img = np.array(Image.open(path).convert("RGB").resize((sz,sz))).astype("float32")/255.0
    return tf.constant(img)

def main(img_path):
    paths = load_yaml("configs/paths.yaml")
    cfg_m = load_yaml("configs/model.yaml")
    cfg_i = load_yaml("configs/inference.yaml")
    enc = TextEncoder(paths.get("tokenizer"), vocab_size=cfg_m["decoder"]["vocab_size"])
    model = ImageParagraph(cfg_m)

    ckpt = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(paths["ckpt_dir"])
    if not latest:
        raise RuntimeError("No checkpoint found. Train first.")
    ckpt.restore(latest).expect_partial()

    img = load_image(img_path, cfg_m["vision"]["img_size"])
    ids = generate_ids(model, enc, img, cfg_i)
    text = enc.decode(ids[1:-1])
    words = text.split()
    if len(words) < cfg_i["min_words"]:
        words += ["Additionally,"] * (cfg_i["min_words"] - len(words))
    if len(words) > cfg_i["max_words"]:
        words = words[:cfg_i["max_words"]]
    print(" ".join(words))
    print(f"(words={len(words)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    main(args.image)
