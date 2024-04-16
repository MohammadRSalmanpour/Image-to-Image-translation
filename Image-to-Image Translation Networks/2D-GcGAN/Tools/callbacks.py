from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, example_us, example_mri, args):
        self.example_input = np.expand_dims(example_us, axis=0)
        self.example_target = np.expand_dims(example_mri, axis=0)
        self.args = args
        self.epoch_number = 1
    
    def on_train_begin(self, logs={}):
        self.history={'disc_loss': [], 'disc_gc_loss': [], 'gen_loss': []}   

    def on_epoch_end(self, epoch, logs=None):
        # save a sample generated image
        sample_generated_image_path = os.path.join(self.args['sample_generated_images_dir'], f'SampleGenerated_epoch{self.epoch_number}')
        generate_images(self.model.generator, self.example_input, self.example_target, sample_generated_image_path)

        # saved models
        self.model.generator.save_weights(os.path.join(self.args['last_models_dir'], 'Generator.h5'))
        self.model.discriminator.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator.h5'))
        self.model.discriminator_gc.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator_GC.h5')) 
        
        # save models for each epoch
        dst_path = os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        self.model.generator.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Generator.h5'))
        self.model.discriminator.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Discriminator.h5'))
        self.model.discriminator_gc.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Discriminator_GC.h5'))
        
        self.history['gen_loss'].append(np.mean(logs.get('gen_loss')))
        self.history['disc_loss'].append(np.mean(logs.get('disc_loss')))
        self.history['disc_gc_loss'].append(np.mean(logs.get('disc_gc_loss')))

        
        epochs = range(1, len(self.history['gen_loss']) + 1)
        plt.plot(epochs, self.history['gen_loss'], label='gen_loss')
        plt.plot(epochs, self.history['disc_loss'], label='disc_loss')
        plt.plot(epochs, self.history['disc_gc_loss'], label='disc_gc_loss')
        plt.title('Generator and Discriminator loss in Epoch {}'.format(self.epoch_number))
        plt.legend()
        plt.savefig(os.path.join(dst_path, 'loss_graph_in_epoch {}'.format(self.epoch_number)))
        plt.close()
	
        self.epoch_number += 1
      

def generate_images(model, test_input, tar, save_to):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig(save_to)
  plt.close()
