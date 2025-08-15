import tensorflow as tf
from utils.image import parse_tfrecord

def test_parse():
    # synthetic record
    feat = {
      "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(tf.zeros([8,8,3], tf.uint8)).numpy()])),
      "ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.constant([1,5,6,2], tf.int32)).numpy()])),
      "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[4]))
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()
    d = parse_tfrecord(ex, 8, 16)
    assert "image" in d and "ids" in d
