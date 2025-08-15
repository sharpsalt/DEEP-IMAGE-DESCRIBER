import tensorflow as tf

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total = total_steps
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      0.5 * self.base_lr * (1 + tf.cos(3.14159265*(step-self.warmup_steps)/tf.maximum(1.0,self.total-self.warmup_steps))))
        return lr
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total
        }
class WarmupLinear(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total = total_steps
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      self.base_lr * (1 - (step - self.warmup_steps) / tf.maximum(1.0, self.total - self.warmup_steps)))
        return lr
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total
        }
class WarmupExponential(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_rate, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.total = total_steps
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      self.base_lr * tf.pow(self.decay_rate, (step - self.warmup_steps) / tf.maximum(1.0, self.total - self.warmup_steps)))
        return lr
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "decay_rate": self.decay_rate,
            "total_steps": self.total
        }
class WarmupInverseTime(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_rate, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.total = total_steps
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      self.base_lr / (1 + (step - self.warmup_steps) / tf.maximum(1.0, self.total - self.warmup_steps)) ** self.decay_rate)
        return lr
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "decay_rate": self.decay_rate,
            "total_steps": self.total
        }
class WarmupPolynomial(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_steps, power=1.0):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.power = power

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      self.base_lr * ((1 - (step - self.warmup_steps) / tf.maximum(1.0, self.decay_steps)) ** self.power))
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "power": self.power
        }
class WarmupInverseSquareRoot(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      self.base_lr / tf.sqrt(tf.maximum(1.0, step - self.warmup_steps)))
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps
        }
class WarmupConstant(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.where(step < self.warmup_steps,
                      self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
                      self.base_lr)
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps
        }
