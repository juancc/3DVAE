"""
3D Convolutional Variational AutoEncoder

JCA
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    def __init__(self, seed=1337, **kwargs):
        super().__init__(**kwargs)
        self.gen = tf.random.Generator.from_seed(seed)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        eps = self.gen.normal(shape=(batch, dim), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def conv_block(x, filters, depth_per_block=2, stride=2):
    x = layers.Conv3D(filters, 3, strides=stride, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    for _ in range(depth_per_block - 1):
        x = layers.Conv3D(filters, 3, strides=1, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
    return x

def upconv_block(x, filters, depth_per_block=2, stride=2):
    x = layers.Conv3DTranspose(filters, 3, strides=stride, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    for _ in range(depth_per_block - 1):
        x = layers.Conv3D(filters, 3, strides=1, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
    return x

class VAE(keras.Model):
    def __init__(self,
                 input_shape=(32, 32, 32, 1),
                 latent_dim=16,
                 alpha=5.0,
                 n_blocks=3,
                 depth_per_block=2,
                 base_filters=32,
                 dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape_ = tuple(input_shape)
        self.spatial_shape = tuple(input_shape[:3])
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.n_blocks = n_blocks
        self.depth_per_block = depth_per_block
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate

        self.encoder = self.get_encoder(self.input_shape_, latent_dim)
        self.decoder = self.get_decoder(latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # self.bce_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
        
        self.bce_fn = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            alpha=0.25,
            gamma=2.0,
            from_logits=False,   # keep False if your decoder ends with sigmoid
            reduction="none"
        )




        print(' -- Model --')
        self.encoder.summary()
        self.decoder.summary()
        

    def get_encoder(self, input_shape, latent_dim):
        inp = keras.Input(shape=input_shape)
        x = layers.Dropout(self.dropout_rate)(inp)

        # x = inp
        filters = self.base_filters
        for _ in range(self.n_blocks):
            x = conv_block(x, filters, self.depth_per_block, stride=2)
            filters *= 2
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(inp, [z_mean, z_log_var, z], name="encoder")

    def get_decoder(self, latent_dim):
        # --- compute starting spatial size (input // 2**n_blocks) ---
        downs = 2 ** self.n_blocks
        if any(s % downs != 0 for s in self.spatial_shape):
            raise ValueError(f"Each spatial dim {self.spatial_shape} must be divisible by 2^{self.n_blocks}={downs}")
        start_dims = tuple(s // downs for s in self.spatial_shape)

        # filters should match the last encoder filters used
        filters = self.base_filters * (2 ** (self.n_blocks - 1))

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(start_dims[0] * start_dims[1] * start_dims[2] * filters, activation="relu")(latent_inputs)
        x = layers.Reshape((*start_dims, filters))(x)

        # apply upconvs: use current filters then halve for next stage
        for _ in range(self.n_blocks):
            x = upconv_block(x, filters, self.depth_per_block, stride=2)
            filters //= 2

        out = layers.Conv3D(1, 3, activation="sigmoid", padding="same")(x)
        return keras.Model(latent_inputs, out, name="decoder")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if len(data.shape) == 4:
            data = tf.expand_dims(data, axis=-1)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            bce = self.bce_fn(data, reconstruction)
            # ensure bce has channel dim
            if bce.shape.rank == 4:
                bce = tf.expand_dims(bce, axis=-1)
            recon_loss_per_sample = tf.reduce_sum(bce, axis=[1, 2, 3, 4])
            reconstruction_loss = tf.reduce_mean(recon_loss_per_sample)

            kl_per_sample = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            kl_loss = tf.reduce_mean(kl_per_sample)

            total_loss = reconstruction_loss + self.alpha * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def sample(self, n_samples=1, temperature=1.0):
        z = tf.random.normal(shape=(n_samples, self.latent_dim)) * temperature
        return self.decoder(z, training=False)

    def call(self, x):
        if len(x.shape) == 4:  # (batch, 32, 32, 32)
            x = tf.expand_dims(x, axis=-1)  # (batch, 32, 32, 32, 1)
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z)

    def reconstruct(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z)