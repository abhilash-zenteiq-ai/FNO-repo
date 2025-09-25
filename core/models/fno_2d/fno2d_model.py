import tensorflow as tf
from core.layers.fourier_layer import FourierLayer

class FNO2D(tf.keras.Model):
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(width, 1)
        self.fourier1 = FourierLayer(width, width, modes)
        self.conv1 = tf.keras.layers.Conv2D(width, 1)
        self.fourier2 = FourierLayer(width, width, modes)
        self.conv2 = tf.keras.layers.Conv2D(width, 1)
        self.output_proj = tf.keras.layers.Conv2D(1, 1)

    def call(self, x):
        x = self.input_proj(x)
        x1 = self.fourier1(x)
        x = tf.nn.gelu(self.conv1(x + x1))
        x2 = self.fourier2(x)
        x = tf.nn.gelu(self.conv2(x + x2))
        return self.output_proj(x)
