import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy, Reduction

gan_loss_fn = BinaryCrossentropy(reduction=Reduction.NONE, from_logits=True)


def get_discriminator_loss(real_outputs, fake_outputs):
    real_loss = gan_loss_fn(tf.ones_like(real_outputs), real_outputs)
    fake_loss = gan_loss_fn(tf.zeros_like(fake_outputs), fake_outputs)
    return real_loss + fake_loss
