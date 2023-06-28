import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import horovod.tensorflow.keras as hvd
import argparse
import h5py as h5
import utils
from CaloLatent import CaloLatent
import tensorflow_addons as tfa
import gc

if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/FCC', help='Folder containing data and MC files')
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/FCC/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='vae', help='Model to train.')
    parser.add_argument('--config', default='config_dataset2.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=float,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.8, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    flags = parser.parse_args()

    config = utils.LoadJson(flags.config)
    data = []
    layers = []
    energies = []
    for dataset in config['FILES']:
        data_,layer_,energy_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),
            config['SHAPE'],flags.nevts,
            emax = config['EMAX'],emin = config['EMIN'],
            logE=config['logE'],
            rank=hvd.rank(),size=hvd.size(),
        )
        
        data.append(data_)
        energies.append(energy_)
        layers.append(layer_)

    data = np.reshape(data,config['SHAPE'])
    layers = np.concatenate(layers)
        
    # data = utils.CalcPreprocessing(data,"preprocessing_{}_voxel.json".format(config['DATASET']))
    # layers = utils.CalcPreprocessing(layers,"preprocessing_{}_layers.json".format(config['DATASET']))
    
    data = utils.ApplyPreprocessing(data,"preprocessing_{}_voxel.json".format(config['DATASET']))
    layers = utils.ApplyPreprocessing(layers,"preprocessing_{}_layers.json".format(config['DATASET']))

    
    energies = np.reshape(energies,(-1,1))    
    data_size = data.shape[0]

    tf_data = tf.data.Dataset.from_tensor_slices(data)        
    tf_energies = tf.data.Dataset.from_tensor_slices(energies)
    tf_layer = tf.data.Dataset.from_tensor_slices(layers)    
    dataset = tf.data.Dataset.zip((tf_data, tf_layer, tf_energies))
    
    train_data, test_data = utils.split_data(dataset,data_size,flags.frac)
    del dataset, data, tf_data,tf_energies, tf_layer
    gc.collect()
    
    BATCH_SIZE = config['BATCH']
    LR = float(config['LR'])
    NUM_EPOCHS = config['MAXEPOCH']
    EARLY_STOP = config['EARLYSTOP']
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),            
        #EarlyStopping(patience=EARLY_STOP,restore_best_weights=True),
    ]

    if flags.model == 'wgan':
        num_noise=config['NOISE_DIM']
        model = WGAN(config['SHAPE_PAD'][1:],energies.shape[1],config=config,num_noise=num_noise)
        opt_gen = tf.optimizers.RMSprop(learning_rate=LR)
        opt_dis = tf.optimizers.RMSprop(learning_rate=LR)

        opt_gen = hvd.DistributedOptimizer(
            opt_gen, backward_passes_per_step=1,
            average_aggregated_gradients=True)

        opt_dis = hvd.DistributedOptimizer(
            opt_dis, backward_passes_per_step=1,
            average_aggregated_gradients=True)
        
        model.compile(
            d_optimizer=opt_dis,
            g_optimizer=opt_gen,        
        )

    elif flags.model == 'vae':
        model = CaloLatent(config['SHAPE'][1:],energies.shape[1],
                           config=config)
        
        
        
        lr_schedule = tf.keras.experimental.CosineDecay(
            initial_learning_rate=LR*hvd.size(), decay_steps=NUM_EPOCHS*int(data_size*flags.frac/BATCH_SIZE)
        )
        opt_vae = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
        opt_vae = hvd.DistributedOptimizer(
            opt_vae,average_aggregated_gradients=True)        
        opt_sgm = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
        opt_sgm = hvd.DistributedOptimizer(
            opt_sgm,average_aggregated_gradients=True)
        opt_layer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
        opt_layer = hvd.DistributedOptimizer(
            opt_layer,average_aggregated_gradients=True)
        
        model.compile(
            layer_optimizer = opt_layer,
            vae_optimizer=opt_vae,
            sgm_optimizer=opt_sgm,        
        )
        
    if flags.load:
        checkpoint_folder = '../checkpoints_{}_{}'.format(config['CHECKPOINT_NAME'],flags.model)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()

    if hvd.rank()==0:
        checkpoint_folder = '../checkpoints_{}_{}'.format(config['CHECKPOINT_NAME'],flags.model)
        checkpoint = ModelCheckpoint('{}/checkpoint'.format(checkpoint_folder),
                                     save_best_only=False,mode='auto',
                                     period=1,save_weights_only=True)
        callbacks.append(checkpoint)
    
    history = model.fit(
        train_data.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size*flags.frac/BATCH_SIZE),
        validation_data=test_data.batch(BATCH_SIZE),
        validation_steps=int(data_size*(1-flags.frac)/BATCH_SIZE),
        verbose=1 if hvd.rank()==0 else 0,
        callbacks=callbacks
    )


    if hvd.rank()==0:
        checkpoint_folder = '../checkpoints_{}_{}'.format(config['CHECKPOINT_NAME'],flags.model)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        os.system('cp CaloLatent.py {}'.format(checkpoint_folder)) # bkp of model def
        os.system('cp JSON/{} {}'.format(flags.config,checkpoint_folder)) # bkp of config file
