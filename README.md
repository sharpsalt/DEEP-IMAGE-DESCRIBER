## Deep Image Describer (TensorFlow)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

1) Download Images
python scripts/download_data.py

2)Generate long paragraphs with HF (no local weights)
export HF_TOKEN=hf_xxx
export HF_MODEL=Qwen/Qwen2-VL-7B-Instruct
python scripts/make_paragraphs_hf.py

3)Build TFRecords
python scripts/build_tfrecords.py

4)Train
TF_XLA_FLAGS="--tf_xla_auto_jit=2" python training/train.py

5)Inference (200–300 words)
python inference/generate.py --image data/raw/coco/coco_000000.jpg

Notes:
Tokenizer is a stub; swap in SentencePiece at utils/text.py when ready.
Metrics: start with scripts/eval_metrics.py (BLEU proxy, diversity). Add CLIP/BERTScore when you can cache models.

