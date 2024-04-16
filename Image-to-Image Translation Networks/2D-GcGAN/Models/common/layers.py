from tensorflow import keras


def conv2D(inputs, filters, strides=2, use_batch_norm=True):
    x = keras.layers.Conv2D(
        filters=filters, kernel_size=4, strides=strides, padding="same", use_bias=False
    )(inputs)

    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.LeakyReLU(0.2)(x)
    return x


def deconv2D(inputs, filters, strides=2):
    x = keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=4, strides=strides, padding="same", use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    return x
