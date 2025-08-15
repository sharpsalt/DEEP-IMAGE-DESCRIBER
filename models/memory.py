import tensorflow as tf

class ParagraphMemory(tf.keras.layers.Layer):
    def __init__(self, dim=512, slots=8):
        super().__init__()
        self.slots = slots
        self.mem = self.add_weight(shape=(slots, dim), initializer="zeros", trainable=True)
        self.read_head = tf.keras.layers.Dense(1)

    def call(self, enc, step_emb, training=False):
        gate = tf.sigmoid(tf.keras.layers.Dense(self.slots)(tf.concat([enc, step_emb], -1)))
        upd = tf.keras.layers.Dense(self.mem.shape[-1], activation="tanh")(enc)
        self.mem.assign(self.mem * (1-gate)[0,:,None] + upd[None,:] * gate[0,:,None])
        att = tf.nn.softmax(self.read_head(self.mem), axis=0)
        read = tf.reduce_sum(att * self.mem, axis=0)
        return read[None, :]
