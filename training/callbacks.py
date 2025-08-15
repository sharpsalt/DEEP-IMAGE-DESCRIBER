import os, json, tensorflow as tf
from utils.text import count_words

class SamplePrinter(tf.keras.callbacks.Callback):
    def __init__(self, enc, every=500):
        super().__init__(); self.enc=enc; self.every=every; self.step=0
    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.every == 0:
            print(f"[sample] loss={logs.get('loss', 0):.4f}")

def ckpt_manager(model, opt, ckpt_dir):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), opt=opt, model=model)
    mgr = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)
    return ckpt, mgr
def save_ckpt(ckpt, mgr):
    ckpt.step.assign_add(1)
    save_path = mgr.save()
    print(f"[ckpt] saved to {save_path} at step {ckpt.step.numpy()}")
    return save_path