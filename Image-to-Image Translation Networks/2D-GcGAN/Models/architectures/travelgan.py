from functools import partial
from itertools import combinations

import tensorflow as tf

from tensorflow import keras
from tensorflow.experimental import numpy as tnp
from tensorflow.keras import layers, losses, optimizers

from Models.common.layers import conv2D, deconv2D
from Models.common.losses import gan_loss_fn

IMAGE_SHAPE = (128, 128, 3)

SIAMESE_DIM = 1000
LAMBDA_TRAVEL = 10.0
LAMBDA_MARGIN = 10.0
LAMBDA_GAN = 1.0
BATCH_SIZE = 16

disc_gan_loss_fn = losses.BinaryCrossentropy(
    reduction=losses.Reduction.NONE, from_logits=True, label_smoothing=0.1
)


def build_generator(image_shape):
    inputs = layers.Input(shape=image_shape)

    conv1 = conv2D(inputs, filters=64)
    conv2 = conv2D(conv1, filters=128)
    conv3 = conv2D(conv2, filters=256)
    conv4 = conv2D(conv3, filters=256)

    conv5 = conv2D(conv4, filters=256)
    deconv5 = deconv2D(conv5, filters=512)
    merged5 = layers.Concatenate()([deconv5, conv4])

    deconv4 = deconv2D(merged5, filters=512)
    merged4 = layers.Concatenate()([deconv4, conv3])

    deconv3 = deconv2D(merged4, filters=512)
    merged3 = layers.Concatenate()([deconv3, conv2])

    deconv2 = deconv2D(merged3, filters=256)
    merged2 = layers.Concatenate()([deconv2, conv1])

    deconv1 = deconv2D(merged2, filters=128)
    outputs = layers.Conv2DTranspose(
        filters=3, kernel_size=4, strides=1, padding="same", activation="tanh", use_bias=False
    )(deconv1)

    return keras.Model(inputs=inputs, outputs=outputs, name="travelgan_unet_generator")


def build_encoder(inputs):
    x = conv2D(inputs, filters=64, use_batch_norm=False)
    x = conv2D(x, filters=128)
    x = conv2D(x, filters=256)
    x = conv2D(x, filters=512)
    x = conv2D(x, filters=512)

    x = layers.Flatten()(x)
    return x


def build_discriminator(image_shape):
    inputs = layers.Input(shape=image_shape)
    x = build_encoder(inputs)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="travelgan_discriminator")


def build_siamese(image_shape, latent_dim):
    inputs = layers.Input(shape=image_shape)
    x = build_encoder(inputs)
    outputs = layers.Dense(latent_dim)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="travelgan_siamese")


def get_discriminator_loss(real_outputs, fake_outputs):
    real_loss = disc_gan_loss_fn(tf.ones_like(real_outputs), real_outputs)
    fake_loss = disc_gan_loss_fn(tf.zeros_like(fake_outputs), fake_outputs)
    return real_loss + fake_loss


def get_travel_loss(siam_real_outputs, siam_fake_outputs, pairs):
    numpy_data = tnp.array(siam_real_outputs)
    d1 = numpy_data[pairs[:, 0]]
    d2 = numpy_data[pairs[:, 1]]
    v1 = d1 - d2

    numpy_data = tnp.array(siam_fake_outputs)
    d1 = numpy_data[pairs[:, 0]]
    d2 = numpy_data[pairs[:, 1]]
    v2 = d1 - d2

    mse_loss = tf.reduce_mean(losses.mean_squared_error(v1, v2))
    travel_loss = tf.reduce_mean(losses.cosine_similarity(v1, v2))
    return mse_loss - travel_loss


def get_generator_loss(
    fake_outputs,
    siam_real_outputs,
    siam_fake_outputs,
    pairs,
    lambda_gan=LAMBDA_GAN,
    lambda_travel=LAMBDA_TRAVEL,
):
    gan_loss = gan_loss_fn(tf.ones_like(fake_outputs), fake_outputs)
    travel_loss = get_travel_loss(siam_real_outputs, siam_fake_outputs, pairs)
    return gan_loss * lambda_gan + travel_loss * lambda_travel


def get_margin_loss(siam_real_outputs, pairs):
    numpy_data = tnp.array(siam_real_outputs)
    d1 = numpy_data[pairs[:, 0]]
    d2 = numpy_data[pairs[:, 1]]

    v1 = d1 - d2

    margin_loss = tf.reduce_mean(tf.maximum(0., 10.0 - tf.norm(v1, axis=1)))
    return margin_loss


def get_siamese_loss(
    siam_real_outputs,
    siam_fake_outputs,
    pairs,
    lambda_margin=LAMBDA_MARGIN,
    lambda_travel=LAMBDA_TRAVEL,
):
    margin_loss = get_margin_loss(siam_real_outputs, pairs)
    travel_loss = get_travel_loss(siam_real_outputs, siam_fake_outputs, pairs)
    return margin_loss * lambda_margin + travel_loss * lambda_travel


class TraVeLGAN(keras.Model):
    def __init__(
        self,
        siamese_dim=SIAMESE_DIM,
        lambda_travel=LAMBDA_TRAVEL,
        lambda_margin=LAMBDA_MARGIN,
        lambda_gan=LAMBDA_GAN,
        batch_size=BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()

        self.generator = build_generator(IMAGE_SHAPE)
        self.generator.summary()

        self.discriminator = build_discriminator(IMAGE_SHAPE)
        self.discriminator.summary()

        self.siamese = build_siamese(IMAGE_SHAPE, siamese_dim)
        self.siamese.summary()

        pairs = tnp.asarray(list(combinations(list(range(batch_size)), 2)))

        self.get_generator_loss = partial(
            get_generator_loss, pairs=pairs, lambda_gan=lambda_gan, lambda_travel=lambda_travel
        )
        self.get_siamese_loss = partial(
            get_siamese_loss, pairs=pairs, lambda_margin=lambda_margin, lambda_travel=lambda_travel
        )

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.gen_opt = optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
        self.disc_opt = optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
        self.siam_opt = optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)

    def train_step(self, batch_data):
        real_A, real_B = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.generator(real_A, training=True)

            disc_real_outputs = self.discriminator(real_B, training=True)
            disc_fake_outputs = self.discriminator(fake_B, training=True)

            siam_real_outputs = self.siamese(real_A, training=True)
            siam_fake_outputs = self.siamese(fake_B, training=True)

            disc_loss = get_discriminator_loss(disc_real_outputs, disc_fake_outputs)
            gen_loss = self.get_generator_loss(
                disc_fake_outputs, siam_real_outputs, siam_fake_outputs
            )
            siam_loss = self.get_siamese_loss(siam_real_outputs, siam_fake_outputs)

        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        siam_gradients = tape.gradient(siam_loss, self.siamese.trainable_variables)

        self.disc_opt.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        self.gen_opt.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.siam_opt.apply_gradients(zip(siam_gradients, self.siamese.trainable_variables))

        return {"d_loss": disc_loss, "g_loss": gen_loss, "s_loss": siam_loss}
