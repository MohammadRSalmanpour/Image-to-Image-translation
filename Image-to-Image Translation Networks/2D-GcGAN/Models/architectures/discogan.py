from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers, losses

from Models.common.layers import conv2D, deconv2D
from Models.common.losses import gan_loss_fn, get_discriminator_loss

IMG_SHAPE = (128, 128, 1)

LAMBDA_GAN = 0.1
LAMBDA_RECONSTR = 1.0
LAMBDA_FML = 0.9

reconstr_loss_fn = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


def build_generator(image_shape, name):
    inputs = layers.Input(shape=image_shape)

    conv = conv2D(inputs, filters=64, use_batch_norm=False)
    conv = conv2D(conv, filters=128)
    conv = conv2D(conv, filters=256)
    conv = conv2D(conv, filters=512)
    conv = conv2D(conv, filters=100, strides=1)

    deconv = deconv2D(conv, filters=512, strides=1)
    deconv = deconv2D(deconv, filters=256)
    deconv = deconv2D(deconv, filters=128)
    deconv = deconv2D(deconv, filters=64)

    outputs = layers.Conv2DTranspose(
        filters=1, kernel_size=4, strides=2, padding="same", activation="tanh", use_bias=False
    )(deconv)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_discriminator(image_shape, name):
    inputs = layers.Input(shape=image_shape)

    conv = conv2D(inputs, filters=64, use_batch_norm=False)
    conv2 = conv2D(conv, filters=128)
    conv3 = conv2D(conv2, filters=256)
    conv4 = conv2D(conv3, filters=512)

    outputs = layers.Conv2D(
        filters=1, kernel_size=4, strides=1, padding="valid", use_bias=False
    )(conv4)

    return keras.Model(inputs=inputs, outputs=[outputs, [conv2, conv3, conv4]], name=name)


def get_feature_matching_loss(real_features, fake_features):
    result = 0.0
    for real_feat, fake_feat in zip(real_features, fake_features):
        l2_norm_sqd = tf.math.squared_difference(
            tf.reduce_mean(real_feat, axis=0), tf.reduce_mean(fake_feat, axis=0)
        )
        result += tf.reduce_mean(l2_norm_sqd)
    return result


def get_generator_loss(
    fake_outputs,
    real,
    reconstr,
    real_features,
    fake_features,
    lambda_gan=LAMBDA_GAN,
    lambda_reconstr=LAMBDA_RECONSTR,
    lambda_fml=LAMBDA_FML,
):
    gan_loss = gan_loss_fn(tf.ones_like(fake_outputs), fake_outputs)
    reconstr_loss = tf.reduce_mean(reconstr_loss_fn(reconstr, real))
    fml_loss = get_feature_matching_loss(real_features, fake_features)
    return gan_loss * lambda_gan + reconstr_loss * lambda_reconstr + fml_loss * lambda_fml


class DiscoGAN(keras.Model):
    def __init__(
        self,
        generator_A,
        generator_B,
        discriminator_A,
        discriminator_B,
        lambda_gan=LAMBDA_GAN,
        lambda_reconstr=LAMBDA_RECONSTR,
        lambda_fml=LAMBDA_FML,
        **kwargs,
    ):
        super().__init__()

        self.gen_A = generator_A
        self.gen_B = generator_B

        self.disc_A = discriminator_A
        self.disc_B = discriminator_B

        self.get_generator_loss = partial(
            get_generator_loss,
            lambda_gan=lambda_gan,
            lambda_reconstr=lambda_reconstr,
            lambda_fml=lambda_fml
        )

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.gen_A_optimizer = tfa.optimizers.AdamW(
            learning_rate=2e-4, beta_1=0.5, weight_decay=1e-5
        )
        self.gen_B_optimizer = tfa.optimizers.AdamW(
            learning_rate=2e-4, beta_1=0.5, weight_decay=1e-5
        )

        self.disc_A_optimizer = tfa.optimizers.AdamW(
            learning_rate=2e-4, beta_1=0.5, weight_decay=1e-5
        )
        self.disc_B_optimizer = tfa.optimizers.AdamW(
            learning_rate=2e-4, beta_1=0.5, weight_decay=1e-5
        )

    def train_step(self, batch_data):
        real_A, real_B = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.gen_B(real_A, training=True)
            fake_A = self.gen_A(real_B, training=True)

            disc_real_B_outputs, disc_real_B_feats = self.disc_B(real_B, training=True)
            disc_fake_B_outputs, disc_fake_B_feats = self.disc_B(fake_B, training=True)
            disc_real_A_outputs, disc_real_A_feats = self.disc_A(real_A, training=True)
            disc_fake_A_outputs, disc_fake_A_feats = self.disc_A(fake_A, training=True)

            disc_B_loss = get_discriminator_loss(disc_real_B_outputs, disc_fake_B_outputs)
            disc_A_loss = get_discriminator_loss(disc_real_A_outputs, disc_fake_A_outputs)

            reconstr_B = self.gen_B(fake_A, training=True)
            reconstr_A = self.gen_A(fake_B, training=True)

            gen_B_loss = self.get_generator_loss(
                disc_fake_B_outputs, real_B, reconstr_B, disc_real_B_feats, disc_fake_B_feats
            )
            gen_A_loss = self.get_generator_loss(
                disc_fake_A_outputs, real_A, reconstr_A, disc_real_A_feats, disc_fake_A_feats
            )

        disc_A_grads = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)
        disc_B_grads = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)

        self.disc_A_optimizer.apply_gradients(zip(disc_A_grads, self.disc_A.trainable_variables))
        self.disc_B_optimizer.apply_gradients(zip(disc_B_grads, self.disc_B.trainable_variables))

        gen_A_grads = tape.gradient(gen_A_loss, self.gen_A.trainable_variables)
        gen_B_grads = tape.gradient(gen_B_loss, self.gen_B.trainable_variables)

        self.gen_A_optimizer.apply_gradients(zip(gen_A_grads, self.gen_A.trainable_variables))
        self.gen_B_optimizer.apply_gradients(zip(gen_B_grads, self.gen_B.trainable_variables))

        
        return {
            "disc_A_loss": disc_A_loss, "disc_B_loss": disc_B_loss, "gen_A_loss": gen_A_loss, "gen_B_loss": gen_B_loss
        }
