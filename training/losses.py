import tensorflow as tf

def xent_loss(logits, labels):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    mask = tf.cast(labels != 0, tf.float32)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)

def length_regularizer(lengths, lo=200, hi=300):
    l = tf.cast(lengths, tf.float32)
    return tf.reduce_mean(tf.nn.relu(lo - l) + tf.nn.relu(l - hi))

def total_loss(logits, labels, lengths):
    loss = xent_loss(logits, labels)
    reg = length_regularizer(lengths)
    return loss + reg