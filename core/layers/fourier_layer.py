import tensorflow as tf

class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        initializer = tf.keras.initializers.RandomNormal(stddev=1e-2)
        self.weights_real = self.add_weight(
            name="w_real",
            shape=[in_channels, out_channels, modes, modes],
            initializer=initializer,
            trainable=True,
        )
        self.weights_imag = self.add_weight(
            name="w_imag",
            shape=[in_channels, out_channels, modes, modes],
            initializer=initializer,
            trainable=True,
        )

    def call(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        x_ft = x_ft[:, :, :self.modes, :self.modes]

        w = tf.complex(self.weights_real, self.weights_imag)
        out_ft = tf.einsum("bcmn,comn->bomn", x_ft, w)

        B, H, W = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        out_full = tf.concat([
            tf.concat([out_ft, tf.zeros([B, self.out_channels, self.modes, W - self.modes], tf.complex64)], axis=-1),
            tf.zeros([B, self.out_channels, H - self.modes, W], tf.complex64)
        ], axis=-2)

        out = tf.signal.ifft2d(out_full)
        return tf.transpose(tf.math.real(out), [0, 2, 3, 1])
