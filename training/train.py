import os, math, tensorflow as tf
from utils.config import load_yaml, ensure_dirs
from utils.image import parse_tfrecord
from utils.text import TextEncoder
from utils.logging import get_logger
from training.losses import xent_loss, length_regularizer
from training.schedules import WarmupCosine
from training.callbacks import ckpt_manager
from models.model import ImageParagraph

logger = get_logger("train")

def build_dataset(tfrecords, batch_size, img_size, max_len):
    files = tf.data.Dataset.from_tensor_slices(tfrecords)
    ds = files.interleave(lambda f: tf.data.TFRecordDataset(f),
                          cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: parse_tfrecord(x, img_size, max_len), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(2048).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    paths = load_yaml("configs/paths.yaml")
    cfg_m = load_yaml("configs/model.yaml")
    cfg_t = load_yaml("configs/train.yaml")

    if cfg_t["mixed_precision"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    tfrecords = [os.path.join(paths["tfrecords"], f) for f in os.listdir(paths["tfrecords"]) if f.endswith(".tfrecord")]
    ds = build_dataset(tfrecords, cfg_t["batch_size"], cfg_m["vision"]["img_size"], 400)

    model = ImageParagraph(cfg_m)
    steps_per_epoch = sum(1 for _ in ds)
    lr = WarmupCosine(cfg_t["lr"], cfg_t["warmup_steps"], steps_per_epoch * cfg_t["epochs"])
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg_t["weight_decay"])
    ckpt, mgr = ckpt_manager(model, opt, paths["ckpt_dir"])

    @tf.function
    def train_step(images, ids):
        inp, tgt = ids[:, :-1], ids[:, 1:]
        with tf.GradientTape() as tape:
            logits = model(images, inp, training=True)
            loss = xent_loss(logits, tgt)
            lengths = tf.reduce_sum(tf.cast(tgt!=0, tf.int32), axis=1)
            loss += cfg_t["loss"]["length_reg_weight"] * length_regularizer(lengths, 200, 300)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, cfg_t["clip_norm"]) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    step = 0
    for epoch in range(cfg_t["epochs"]):
        for batch in ds:
            loss = train_step(batch["image"], batch["ids"])
            step += 1
            if step % 100 == 0:
                logger.info(f"epoch {epoch+1} step {step} loss {float(loss):.4f}")
        mgr.save()
        logger.info(f"epoch {epoch+1} saved.")

if __name__ == "__main__":
    main()
