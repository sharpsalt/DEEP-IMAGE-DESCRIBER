import tensorflow as tf

def build_vision_backbone(img_size=384, trainable_layers=80):
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False, input_shape=(img_size,img_size,3), weights="imagenet")
    for l in base.layers[:-trainable_layers]:
        l.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(768, activation="gelu")(x)
    return tf.keras.Model(base.input, x, name="vision_encoder")
