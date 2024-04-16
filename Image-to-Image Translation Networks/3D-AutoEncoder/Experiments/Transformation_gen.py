import os
import numpy as np
from Models import EncoderDecoder, UNet
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from Tools import metrics
from Tools import callbacks
import nibabel as nib
import utils
from pathlib import Path


class Experiment_transformation():
    def __init__(self, preprocessed_dataset_path=None, network_structure='UNet', base_project_dir='.', args={}, mode='train'):
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.network_structure = network_structure
        self.base_project_dir = base_project_dir
        self.args = args
        self.task = ''
        self.mode = mode

        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def prepare(self):
        self.train_ds, self.val_ds, self.test_ds, self.file_names = utils.create_dataset(self.preprocessed_dataset_path, self.args['val_ratio'], self.args['test_ratio'], True, self.args['random_seed'])
        
        
        def load_sample(file_path):
            data = np.load(file_path.numpy(), allow_pickle=True)  # convert Tensor to numpy
            return np.expand_dims(data['us'], axis=-1), np.expand_dims(data['mri'], axis=-1)
        
        """
        def make_gen_callable(_gen):
            def gen():
                for x in _gen:
                    data = np.load(file_path.numpy(), allow_pickle=True)  # convert Tensor to numpy
                    yield np.expand_dims(data['us'], axis=-1), np.expand_dims(data['mri'], axis=-1)
                    
                    data_tensor = tf.data.Dataset.from_tensor_slices(list(x))
                    data_tensor = test_data_shape.map(lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32)))
                    yield data_tensor
            return gen"""
    
        def generator_from_disk(images):
            for image in images:
                yield image
        
        self.train_dataset = tf.data.Dataset.from_generator(generator_from_disk, args=[self.train_ds], output_types=(tf.string))
        self.train_dataset = self.train_dataset.map(lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32)))
      
        self.val_dataset = tf.data.Dataset.from_generator(generator_from_disk, args=[self.val_ds], output_types=(tf.string))
        self.val_dataset = self.val_dataset.map(lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32)))
        
        self.test_dataset = tf.data.Dataset.from_generator(generator_from_disk, args=[self.test_ds], output_types=(tf.string))
        self.test_dataset = self.test_dataset.map(lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32)))
    
        # this is useful for 'generate_output' mode where we wnat to save data
        image_path = os.path.join(self.base_project_dir, 'Data', 'US_Images', '0001_US_Image_a.nii.gz')
        nii_img  = nib.load(image_path)
        self.images_metadata = {'affine':nii_img.affine, 'header':nii_img.header}

        train_data_shape = tf.data.Dataset.from_tensor_slices(self.train_ds)
        train_data_shape = train_data_shape.map(lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32)))
        example_us, _ = next(iter(train_data_shape))
        self.args['input_shape'] =  example_us.shape #list(example_us.shape) + [1]
        print('Input Shape:', self.args['input_shape'])
        print('\nTrain size:', len(self.train_ds))
        print('Val size:', len(self.val_ds))
        print('Test size:', len(self.test_ds))
        
        self.task += self.network_structure
        
        output_dir = os.path.join(self.base_project_dir, 'Output', self.task)
        output_models_dir = os.path.join(output_dir, 'TrainedModels')
        self.args['best_models_dir'] = os.path.join(output_models_dir, 'BestModels')
        self.args['last_models_dir']= os.path.join(output_models_dir, 'LastModels')
        self.args['log_dir'] = os.path.join(output_dir, 'Log')
        self.args['sample_generated_images_dir'] = os.path.join(output_dir, 'SampleGeneratedImages')

        if self.network_structure == 'EncoderDecoder':
            self.model = EncoderDecoder.EncoderDecoder_Model(input_shape=self.args['input_shape'])
        elif self.network_structure == 'UNet':
            self.model = UNet.UNet_Model(input_shape=self.args['input_shape'])

        if os.path.exists(os.path.join(self.args['last_models_dir'], self.task + '.h5')):
            print('[INFO] Loading pretrained model weights from: ' + self.args['last_models_dir'])
            pretrained_model_path = os.path.join(self.args['last_models_dir'], self.task + '.h5')
            self.model.load_weights(pretrained_model_path)
        else:
            print('[INFO] Pretrained weights not found. training will be strated using randomly intialized weights.')

        self.model.summary()
        
        Path(self.args['best_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['last_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['sample_generated_images_dir']).mkdir(parents=True, exist_ok=True)

        print("Task: " + self.task)

        
    def train(self):

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        self.args['initial_lr'],
        decay_steps=self.args['decay_steps'],
        decay_rate=self.args['decay_rate'],
        staircase=True
        )

        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(
            optimizer=opt,
            loss=  tf.keras.losses.MeanSquaredError()
        )
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.args['last_models_dir'], self.task + '.h5'),
            verbose=1, save_weights_only=True, save_freq='epoch')

        save_model_callback = callbacks.SaveModel(
            save_best_model_to=os.path.join(self.args['best_models_dir'], self.task + '.h5'),
            save_last_model_to=os.path.join(self.args['last_models_dir'], self.task + '.h5')
        )

        print('\n\n')  

        self.model.save('InitialModel.h5')

        try:
            self.model.fit(
                self.train_dataset.batch(self.args['batch_size'], drop_remainder=True).repeat(),
                steps_per_epoch=len(self.train_ds) // self.args['batch_size'],
                epochs=self.args['max_epochs'],
                callbacks=[save_model_callback],
                validation_data=self.val_dataset.batch(self.args['batch_size']),
                validation_steps = len(self.val_ds) // self.args['batch_size'],
                )
        except KeyboardInterrupt :
            pass

    def generate_output(self):
        output_path = os.path.join(self.base_project_dir, 'Output', self.task, 'Results')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for set_name in ['train', 'val', 'test']:
            for result_dir_name in ['GeneratedImages', 'DifferentialImages', 'ResultedMetrics']:
                result_dir_path = os.path.join(output_path, set_name, result_dir_name)
                Path(result_dir_path).mkdir(parents=True, exist_ok=True)

        print('Loading the best model weights from: ' + self.args['best_models_dir'])
        self.model.load_weights(os.path.join(self.args['best_models_dir'], self.task + '.h5'))
        
        mae_list = []
        mape_list = []
        rmse_list = []
        ssi_list = []
        file_name_list = []

        print('Generating outputs ...')
        for set_name in ['train', 'val', 'test']:
            generated_images_path = os.path.join(output_path, set_name, 'GeneratedImages')
            differential_images_path = os.path.join(output_path, set_name, 'DifferentialImages')
            resulted_metrics_path = os.path.join(output_path, set_name, 'ResultedMetrics')
            
            df_columns_names = ['MAE', 'MAPE', 'RMSE', 'SSI']#, 'PSNR']
            df_metrics = pd.DataFrame(columns=df_columns_names)

            if set_name == 'train':
                dataset = self.train_dataset
            elif set_name == 'val':
                dataset = self.val_dataset
            elif set_name == 'test':
                dataset = self.test_dataset

            for i, (US, gt_MRI) in tqdm(enumerate(dataset)):
                file_name = self.file_names[set_name][i].split(os.sep)[-1].replace('.npz', '.nii.gz')
                US = US.numpy()
                gt_MRI = gt_MRI.numpy()

                pred_MRI = self.model(np.expand_dims(US, axis=0), training=True).numpy()
                pred_MRI = np.squeeze(pred_MRI)
                gt_MRI = np.squeeze(gt_MRI)

                pred_MRI_path = os.path.join(generated_images_path, file_name)
                pred_nii_img = nib.Nifti1Image(pred_MRI, self.images_metadata['affine'], self.images_metadata['header'])
                nib.save(pred_nii_img, pred_MRI_path)

                diff_MRI = gt_MRI - pred_MRI
                diff_MRI_path = os.path.join(differential_images_path, file_name)
                diff_nii_img = nib.Nifti1Image(diff_MRI, self.images_metadata['affine'], self.images_metadata['header'])
                nib.save(diff_nii_img, diff_MRI_path)

                mae = metrics.Mean_absolute_error(gt_MRI, pred_MRI)
                mape = metrics.Mean_absolute_percentage_error(gt_MRI, pred_MRI)
                rmse = metrics.Root_mean_squared_error(gt_MRI, pred_MRI)
                ssi = metrics.Structural_similarity(gt_MRI, pred_MRI)
                # psnr = metrics.Peak_signal_noise_ratio(gt_MRI, pred_MRI)
                # psnr = 0

                
                mae_list.append(mae)
                mape_list.append(mape)
                rmse_list.append(rmse)
                ssi_list.append(ssi)
                file_name_list.append(file_name)
                
                df_metrics.loc[file_name] = [mae, mape, rmse, ssi]#, psnr]
            
            
            dict_metrics = {'Patient ID': file_name_list, 'MAE': mae_list, 'MAPE': mape_list, 'RMSE': rmse_list, 'SSI': ssi_list}
            df_metrics_all = pd.DataFrame(dict_metrics)
            
            df_metrics_path = os.path.join(resulted_metrics_path, 'CalculatedMetrics.csv')
            df_metrics_path_all = os.path.join(resulted_metrics_path, 'Toatal_CalculatedMetrics.csv')
            df_metrics_all.to_csv(df_metrics_path_all)
            df_metrics.to_csv(df_metrics_path)


        




        
