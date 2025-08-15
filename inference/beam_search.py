import numpy as np
import tensorflow as tf

def sample_top_p(logits, top_p=0.9, temperature=0.7):
    logits = logits / max(temperature, 1e-8)
    probs = tf.nn.softmax(logits).numpy()
    idx = np.argsort(probs)[::-1]
    probs = probs[idx]
    cdf = np.cumsum(probs)
    cutoff = np.searchsorted(cdf, top_p) + 1
    idx, probs = idx[:cutoff], probs[:cutoff]
    probs = probs / probs.sum()
    return int(np.random.choice(idx, p=probs))

def generate_ids(model, enc, img, cfg):
    start, end = enc.start_id, enc.end_id
    ids = [start]; words = 0
    while True:
        inp = tf.constant([ids], dtype=tf.int32)
        enc_vec = model.vision(img[None,...], training=False)
        logits = model.dec(inp, enc_vec, training=False)[0, -1]  # [V]
        tok = sample_top_p(logits[:enc.vocab_size], cfg["top_p"], cfg["temperature"])
        ids.append(tok)
        if tok == end and words >= cfg["min_words"]: break
        words += 1
        if words >= cfg["max_words"]:
            ids.append(end); break
    return ids
