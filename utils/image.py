from pathlib import Path
import tensorflow as tf

def load_image_bytes(path, target_size=384):
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (target_size, target_size))
    img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    return tf.io.encode_jpeg(img).numpy()

def parse_tfrecord(rec, img_size=384, max_len=400):
    feat = {
      "image": tf.io.FixedLenFeature([], tf.string),
      "ids": tf.io.FixedLenFeature([], tf.string),
      "length": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(rec, feat)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    ids = tf.io.parse_tensor(ex["ids"], out_type=tf.int32)
    ids = ids[:max_len]
    return {"image": img, "ids": ids}
