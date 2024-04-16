import os

from pathlib import Path

from tensorflow import keras


class ImageSampler(keras.callbacks.Callback):

    NUMBER_OF_SAMPLES = 5

    def __init__(self, every_N_epochs, samples_dir):
        super().__init__()

        self.every_N_epochs = every_N_epochs
        self.samples_dir = samples_dir

        Path(samples_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.every_N_epochs == 1 or epoch % self.every_N_epochs == 0:
            generator = self.images_generator()

            for i in range(self.NUMBER_OF_SAMPLES):
                real_input, fake_output = next(generator)

                real_image = keras.preprocessing.image.array_to_img(real_input)
                fake_image = keras.preprocessing.image.array_to_img(fake_output)

                real_image_path = os.path.join(self.samples_dir, f"real_input_{epoch}_{i}.png")
                fake_image_path = os.path.join(self.samples_dir, f"fake_output_{epoch}_{i}.png")
                real_image.save(real_image_path)
                fake_image.save(fake_image_path)

    def images_generator(self):
        raise NotImplementedError()
