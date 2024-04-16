from functools import partial

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_gan as tfgan

from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers

IMG_SHAPE = (128, 128, 1)
LR = 0.0002

BATCH_SIZE = 1
EPOCHS_DECAY_THRESHOLD = 100

LAMBDA_A2B = 10.0
LAMBDA_GC = 2.0
LAMBDA_GAN = 1.0
LAMBDA_IDENT = 0.3

kernel_init_fn = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init_fn = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

gan_loss_fn = losses.MeanSquaredError(reduction=losses.Reduction.NONE)
geometry_loss_fn = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)
identity_loss_fn = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)


class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init_fn,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init_fn,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init_fn,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init_fn,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init_fn,
    gamma_initializer=gamma_init_fn,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_generator(
    image_shape, name, num_downsampling_blocks=2, num_residual_blocks=6, num_upsample_blocks=2,
):
    filters = 64

    img_input = layers.Input(shape=image_shape)

    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init_fn, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init_fn)(x)
    x = layers.Activation("relu")(x)

    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(1, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


def get_discriminator(image_shape, name):
    filters=64

    img_input = layers.Input(shape=image_shape)

    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_init_fn,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_init_fn
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


def get_discriminator_loss(real, fake):
    real_loss = gan_loss_fn(tf.ones_like(real), real)
    fake_loss = gan_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def get_geometry_loss(fake, fake_rot):
    fake_rot_f_reverse_rot = tf.experimental.numpy.rot90(fake_rot, k=1, axes=(1, 2))
    fake_f_rot = tf.experimental.numpy.rot90(fake, k=3, axes=(1, 2))

    loss = geometry_loss_fn(fake, fake_rot_f_reverse_rot) + geometry_loss_fn(fake_rot, fake_f_rot)

    return loss


def get_generator_loss(
    fake_outputs,
    fake_rot_outputs,
    fake,
    fake_rot,
    identity,
    real,
    identity_rot,
    real_rot,
    lambda_gan=LAMBDA_GAN,
    lambda_A2B=LAMBDA_A2B,
    lambda_ident=LAMBDA_IDENT,
    lambda_gc=LAMBDA_GC,
):
    gen_loss = tf.reduce_mean(gan_loss_fn(tf.ones_like(fake_outputs), fake_outputs)) \
               + tf.reduce_mean(gan_loss_fn(tf.ones_like(fake_rot_outputs), fake_rot_outputs))
    identity_loss = identity_loss_fn(identity, real) + identity_loss_fn(identity_rot, real_rot)
    geometry_loss = get_geometry_loss(fake, fake_rot)
    return gen_loss * lambda_gan \
        + (identity_loss * lambda_ident + geometry_loss * lambda_gc) * lambda_A2B


class LinearDecaySchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, epochs, dataset_size, batch_size):
        super().__init__()

        steps_per_epoch = dataset_size // batch_size

        self.initial_lr = lr
        self.steps_threshold = EPOCHS_DECAY_THRESHOLD * steps_per_epoch
        self.steps_to_decay = (epochs - EPOCHS_DECAY_THRESHOLD) * steps_per_epoch
        self.decay_value = self.initial_lr / self.steps_to_decay

    @tf.function
    def __call__(self, step):
        if step > self.steps_threshold:
            return self.initial_lr - self.decay_value * (step - self.steps_threshold)
        else:
            return self.initial_lr


class GcGAN(keras.Model):
    def __init__(
        self,
        generator_gcgan,
        discriminator_simple,
        discriminator_gcgan,
        lambda_gan=LAMBDA_GAN,
        lambda_A2B=LAMBDA_A2B,
        lambda_gc=LAMBDA_GC,
        lambda_ident=LAMBDA_IDENT,
        **kwargs,
    ):
        super().__init__()

        self.generator = generator_gcgan
        self.discriminator = discriminator_simple
        self.discriminator_gc = discriminator_gcgan

        self.linear_decay_lr = partial(
            LinearDecaySchedule,
            lr=2e-4,
            epochs=150,
            dataset_size=38080,
            batch_size=1,
        )

        self.get_generator_loss = partial(
            get_generator_loss,
            lambda_gan=lambda_gan,
            lambda_A2B=lambda_A2B,
            lambda_ident=lambda_ident,
            lambda_gc=lambda_gc,
        )

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.gen_optimizer = optimizers.Adam(learning_rate=self.linear_decay_lr(), beta_1=0.5)
        self.disc_optimizer = optimizers.Adam(learning_rate=self.linear_decay_lr(), beta_1=0.5)
        self.disc_gc_optimizer = optimizers.Adam(learning_rate=self.linear_decay_lr(), beta_1=0.5)

    def train_step(self, data):
        real_A, real_B = data
        real_A_rot, real_B_rot = tf.image.rot90(real_A, k=3), tf.image.rot90(real_B, k=3)

        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.generator(real_A, training=True)
            fake_B_rot = self.generator(real_A_rot, training=True)

            with tape.stop_recording():
                pooled_fake_B = tfgan.features.tensor_pool(fake_B, name="pool_disc")
                pooled_fake_B_rot = tfgan.features.tensor_pool(fake_B_rot, name="pool_disc_gc")

            real_B_outputs = self.discriminator(real_B, training=True)
            real_B_rot_outputs = self.discriminator_gc(real_B_rot, training=True)

            fake_B_outputs = self.discriminator(pooled_fake_B, training=True)
            fake_B_rot_outputs = self.discriminator_gc(pooled_fake_B_rot, training=True)

            disc_loss = get_discriminator_loss(real_B_outputs, fake_B_outputs)
            disc_gc_loss = get_discriminator_loss(real_B_rot_outputs, fake_B_rot_outputs)

            disc_fake_B_outputs = self.discriminator(fake_B)
            disc_fake_B_rot_outputs = self.discriminator_gc(fake_B_rot)

            fake_B_same = self.generator(real_B, training=True)
            fake_B_rot_same = self.generator(real_B_rot, training=True)

            gen_loss = self.get_generator_loss(
                disc_fake_B_outputs,
                disc_fake_B_rot_outputs,
                fake_B,
                fake_B_rot,
                fake_B_same,
                real_B,
                fake_B_rot_same,
                real_B_rot,
            )

        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        disc_gc_gradients = tape.gradient(disc_gc_loss, self.discriminator_gc.trainable_variables)
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)

        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        self.disc_gc_optimizer.apply_gradients(
            zip(disc_gc_gradients, self.discriminator_gc.trainable_variables)
        )
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss, "disc_gc_loss": disc_gc_loss}
