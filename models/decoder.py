import tensorflow as tf

class MoRDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout, recursions):
        super().__init__()
        self.sa = tf.keras.layers.MultiHeadAttention(n_heads, d_model//n_heads, dropout=dropout)
        self.ca = tf.keras.layers.MultiHeadAttention(n_heads, d_model//n_heads, dropout=dropout)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.recursions = recursions

    def call(self, y, enc, attn_mask, training):
        for _ in range(self.recursions):
            z = self.sa(y, y, attention_mask=attn_mask, training=training)
            y = self.norm1(y + self.drop(z, training=training))
            z = self.ca(y, enc, training=training)
            y = self.norm2(y + self.drop(z, training=training))
            z = self.ff(y, training=training)
            y = self.norm3(y + self.drop(z, training=training))
        return y

class TransformerDecoder(tf.keras.Model):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, recursions, dropout):
        super().__init__()
        self.tok = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos = tf.keras.layers.Embedding(4096, d_model)
        self.blocks = [MoRDecoderBlock(d_model, n_heads, d_ff, dropout, recursions) for _ in range(n_layers)]
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.lm = tf.keras.layers.Dense(vocab_size)

    def call(self, ids, enc, training):
        t = tf.shape(ids)[1]
        x = self.tok(ids) + self.pos(tf.range(t)[None, :])
        mask = tf.linalg.band_part(tf.ones((t,t)), -1, 0)[None, None, ...]
        # enc: [B, D] → expand to [B, T_enc, D] with T_enc=1
        enc = enc[:, None, :]
        for blk in self.blocks:
            x = blk(x, enc, mask, training)
        x = self.ln(x)
        return self.lm(x)
