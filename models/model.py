import tensorflow as tf
from .vision import build_vision_backbone
from .decoder import TransformerDecoder
from .memory import ParagraphMemory

class ImageParagraph(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.vision = build_vision_backbone(cfg["vision"]["img_size"], cfg["vision"]["trainable_layers"])
        self.dec = TransformerDecoder(
            cfg["decoder"]["vocab_size"], cfg["decoder"]["d_model"],
            cfg["decoder"]["n_heads"], cfg["decoder"]["d_ff"],
            cfg["decoder"]["n_layers"], cfg["decoder"]["recursions"],
            cfg["decoder"]["dropout"]
        )
        self.mem = ParagraphMemory(cfg["memory"]["dim"], cfg["memory"]["slots"])
        self.to_d = tf.keras.layers.Dense(cfg["decoder"]["d_model"])

    def call(self, images, ids, training):
        enc = self.vision(images, training=training)                   # [B, 768]
        step_emb = tf.reduce_mean(self.dec.tok(ids), axis=1)           # [B, d]
        mem_read = self.mem(enc, step_emb)
        enc_plus = tf.concat([enc, mem_read], axis=-1)
        enc_proj = self.to_d(enc_plus)                                 # [B, d_model]
        logits = self.dec(ids, enc_proj, training)
        return logits
