from functools import partial

import tensorflow as tf


def read_image(image):
    image = tf.io.decode_png(tf.io.read_file(image), channels=3)
    return tf.data.Dataset.from_tensors(image)


def normalize_image(image):
    image = tf.image.resize(image, [128, 128]) / 127.5 - 1
    return image


def augment_images(images, augmentations):
    if "random_flip_left_right" in augmentations:
        images = tf.image.random_flip_left_right(images)
    if "random_brightness" in augmentations:
        images = tf.image.random_brightness(images, 0.15)
    if "random_saturation" in augmentations:
        images = tf.image.random_saturation(images, 0.8, 1.8)
    return images


def build_input_pipeline(domain_files, dataset_size, batch_size, augment=None, cache=True):
    d = tf.data.Dataset.list_files(domain_files).take(dataset_size)
    d = d.interleave(read_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        d = d.map(
            partial(augment_images, augmentations=augment), num_parallel_calls=tf.data.AUTOTUNE
        )

    d = d.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=True)

    if cache:
        d = d.cache()

    d = d.prefetch(tf.data.AUTOTUNE)
    return d
