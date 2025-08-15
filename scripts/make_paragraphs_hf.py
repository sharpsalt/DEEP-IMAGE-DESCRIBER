import os, io, json, glob
from pathlib import Path
from huggingface_hub import InferenceClient
from utils.config import load_yaml, ensure_dirs
from utils.logging import get_logger

logger = get_logger("hf_paragraphs")

PROMPT = ("You are a meticulous vision assistant. In ~250 words, write one coherent paragraph "
          "that describes the scene, key objects with attributes (color, material, count), spatial relations, "
          "actions, and plausible context (use 'likely'/'perhaps' for speculation). Avoid bullets/headings/repetition.")

def iter_images(folder):
    for p in sorted(glob.glob(str(Path(folder) / "*.jpg"))):
        yield p

def run_for_folder(client, in_dir, out_path, min_tokens=220, max_tokens=380):
    with open(out_path, "w", encoding="utf-8") as f:
        for img_path in iter_images(in_dir):
            with open(img_path, "rb") as ib:
                image_bytes = ib.read()
            out = client.chat.completions.create(
                messages=[{"role":"user","content":[
                    {"type":"input_text","text":PROMPT},
                    {"type":"input_image","image":image_bytes}
                ]}],
                temperature=0.7, top_p=0.9, max_tokens=max_tokens, min_tokens=min_tokens
            )
            text = out.choices[0].message.content.strip()
            f.write(json.dumps({"image": img_path, "paragraph": text}, ensure_ascii=False) + "\n")

def main():
    paths = load_yaml("configs/paths.yaml")
    ensure_dirs(paths["hf_jsonl_dir"])
    model = os.environ.get("HF_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Please `export HF_TOKEN=...`")
    client = InferenceClient(model=model, token=token)

    run_for_folder(client, Path(paths["raw_images"]) / "coco", Path(paths["hf_jsonl_dir"]) / "coco.jsonl")
    run_for_folder(client, Path(paths["raw_images"]) / "flickr30k", Path(paths["hf_jsonl_dir"]) / "flickr30k.jsonl")
    logger.info("HF paragraphs complete.")

if __name__ == "__main__":
    main()
# This script generates paragraphs for images in specified folders using a Hugging Face model.
# It saves the results in JSONL format with image paths and generated paragraphs.