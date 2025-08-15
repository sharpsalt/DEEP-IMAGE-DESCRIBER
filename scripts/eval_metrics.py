import json, math, re
from pathlib import Path
from utils.logging import get_logger

logger = get_logger("eval")

def simple_bleu(reference: str, candidate: str, n=4):
    def ngrams(s, n): 
        toks = s.split()
        return list(zip(*[toks[i:] for i in range(n)])) if len(toks)>=n else []
    score = 0.0; weights = [0.25]*4
    for k in range(4):
        r = ngrams(reference, k+1); c = ngrams(candidate, k+1)
        if not c: return 0.0
        match = sum(1 for g in c if g in set(r))
        score += weights[k] * (match / max(len(c),1))
    bp = math.exp(1 - len(reference.split())/max(1,len(candidate.split()))) if len(candidate.split())<=len(reference.split()) else 1.0
    return bp * math.exp(min(0.0, score-1)) if score>0 else 0.0

def main(pred_jsonl, ref_jsonl=None):
    preds = [json.loads(l) for l in open(pred_jsonl, encoding="utf-8")]
    if ref_jsonl and Path(ref_jsonl).exists():
        refs = [json.loads(l) for l in open(ref_jsonl, encoding="utf-8")]
        refs_map = {Path(r["image"]).name: r["paragraph"] for r in refs}
        scores = []
        for p in preds:
            name = Path(p["image"]).name
            scores.append(simple_bleu(refs_map.get(name, ""), p["paragraph"]))
        logger.info(f"BLEU proxy avg: {sum(scores)/max(1,len(scores)):.4f}")
    # diversity proxy
    uniq = set()
    for p in preds:
        uniq.update(p["paragraph"].lower().split())
    logger.info(f"Vocab size: {len(uniq)} over {len(preds)} samples")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--ref_jsonl", default=None)
    args = ap.parse_args()
    main(args.pred_jsonl, args.ref_jsonl)
# This script evaluates the generated paragraphs against reference captions using a simple BLEU score.
# It also calculates the vocabulary size of the generated paragraphs as a diversity proxy.