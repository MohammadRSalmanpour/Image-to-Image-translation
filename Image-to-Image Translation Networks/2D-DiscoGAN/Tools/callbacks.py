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
        self.history={'disc_A_loss': [], 'disc_B_loss': [], 'gen_A_loss': [], 'gen_B_loss': []}   

    def on_epoch_end(self, epoch, logs=None):
        # save a sample generated image
        sample_generated_image_path = os.path.join(self.args['sample_generated_images_dir'], f'SampleGenerated_epoch{self.epoch_number}')
        generate_images(self.model.gen_B, self.example_input, self.example_target, sample_generated_image_path)

        # saved models
        self.model.gen_A.save_weights(os.path.join(self.args['last_models_dir'], 'Generator_A.h5'))
        self.model.gen_B.save_weights(os.path.join(self.args['last_models_dir'], 'Generator_B.h5'))
        self.model.disc_A.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator_A.h5'))
        self.model.disc_B.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator_B.h5')) 
        
        # save models for each epoch
        dst_path = os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        
        self.model.gen_A.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Generator_A.h5'))
        self.model.gen_B.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Generator_B.h5'))
        self.model.disc_A.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Discriminator_A.h5'))
        self.model.disc_B.save_weights(os.path.join(self.args['last_models_dir'], 'Epoch{}'.format(self.epoch_number), 'Discriminator_B.h5'))
        
        self.history['gen_A_loss'].append(np.mean(logs.get('gen_A_loss')))
        self.history['gen_B_loss'].append(np.mean(logs.get('gen_B_loss')))
        self.history['disc_A_loss'].append(np.mean(logs.get('disc_A_loss')))
        self.history['disc_B_loss'].append(np.mean(logs.get('disc_B_loss')))

        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX Callback: ')
        print('log: ', logs.get('gen_A_loss'))
        print('history: ', self.history['gen_A_loss'])
        print('history: ', self.history['gen_B_loss'])
        
        epochs = range(1, len(self.history['gen_B_loss']) + 1)
        plt.plot(epochs, self.history['gen_B_loss'], label='gen_B_loss')
        plt.plot(epochs, self.history['gen_A_loss'], label='gen_A_loss')
        plt.plot(epochs, self.history['disc_A_loss'], label='disc_A_loss')
        plt.plot(epochs, self.history['disc_B_loss'], label='disc_B_loss')
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
