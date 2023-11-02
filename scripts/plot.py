import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os, sys
import utils
import tensorflow as tf
from WGAN import WGAN
from CaloLatent import CaloLatent
import time
import horovod.tensorflow.keras as hvd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
import copy
from utils import train_test_split

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

utils.SetStyle()


parser = argparse.ArgumentParser()

#parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/FCC', help='Folder containing data and MC files')
parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/FCC/', help='Folder containing data and MC files')
#parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/FCC/', help='Folder containing data and MC files')
parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
parser.add_argument('--config', default='config_dataset2.json', help='Training parameters')
parser.add_argument('--nevts', type=int,default=1000, help='Number of events to load')
parser.add_argument('--model', default='vae', help='Model to train')
parser.add_argument('--sample', action='store_true', default=False,help='Sample from learned model')
parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
parser.add_argument('--noise_dims', type=int,default=None, help='Factor to multiply base latent dims by')
parser.add_argument('--coordinates', type=str,default="", help='Which coordinate system is the data in')


flags = parser.parse_args()

nevts = int(flags.nevts)
config = utils.LoadJson(flags.config)

if flags.noise_dims:
    config["NOISE_DIM"] = flags.noise_dims

run_classifier=False
ld_plot=True

if flags.sample:
    checkpoint_folder = '../checkpoints_{}_{}_ld{}'.format(config['CHECKPOINT_NAME'],flags.model, config["NOISE_DIM"])
    print(checkpoint_folder)
    
    energies = []
    if ld_plot:
        layers = []
        data = []
        for dataset in config['EVAL']:
        
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
        
        data = utils.ApplyPreprocessing(data,"preprocessing_{}_voxel{}.json".format(config['DATASET'], flags.coordinates))
        layers = utils.ApplyPreprocessing(layers,"preprocessing_{}_layers{}.json".format(config['DATASET'], flags.coordinates))

        
        energies = np.reshape(energies,(-1,1))    
        data_size = data.shape[0]
        # tf_data = tf.data.Dataset.from_tensor_slices(data)        
        # tf_energies = tf.data.Dataset.from_tensor_slices(energies)
        # tf_layer = tf.data.Dataset.from_tensor_slices(layers)    
        # dataset = tf.data.Dataset.zip((tf_data, tf_layer, tf_energies))
    else:
        for dataset in config['EVAL']:
            energy_ = utils.EnergyLoader(os.path.join(flags.data_folder,dataset),
                                        flags.nevts,
                                        emax = config['EMAX'],emin = config['EMIN'],
                                        logE=config['logE'])
            energies.append(energy_)

    energies = np.reshape(energies,(-1,1))

    if flags.model == 'wgan':
        num_noise=config['NOISE_DIM']
        model = WGAN(config['SHAPE'][1:],energies.shape[1],config=config,num_noise=num_noise)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
        start = time.time()
        generated = model.generate(energies.shape[0],energies)
        end = time.time()
        print(end - start)
        
    elif "vae" in flags.model:
        hvd.init()
        model = CaloLatent(config['SHAPE'][1:],energies.shape[1],
                           config=config, name=flags.model)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint-499')).expect_partial()
        start = time.time()        
        print("start sampling")
        voxels=[]
        layers_ = []
        m_latents = []
        t_latents = []
        nsplit = 100 #100
        split_energy = np.array_split(energies,nsplit)
        split_layer = np.array_split(layers, nsplit)
        split_data = np.array_split(data, nsplit)
        for split_e, split_l, split_d in zip(split_energy, split_layer, split_data):
            voxel,layer, m_latent, t_latent  = model.generate(split_e.shape[0],split_e, sample_encoder=ld_plot, layer=split_l, data=split_d)
            voxels.append(voxel)
            layers_.append(layer)
            m_latents.append(m_latent)
            t_latents.append(t_latent)

        voxels = np.concatenate(voxels)
        layers = np.concatenate(layers_)
        end = time.time()

        # m_latent = np.concatenate(m_latents)
        t_latent = np.concatenate(t_latents)

        print(end - start)

        #plot latent_dims
        if ld_plot:
            dict_ = {
                "vae": "VAE+Diffusion",
                "vae_only": "VAE"
            }
            plt.figure()
            # _ = plt.hist(m_latent, bins=35, label=f"{dict_[flags.model]} Latent", color="orangered", alpha=0.5)
            _ = plt.hist(t_latent, bins=35, label="Ground Truth Latent", color="purple", alpha=0.5)
            plt.legend()
            plt.xlabel("Random Latent Dim")
            plt.ylabel("Entries")
            plt.savefig(f"latent_dims_{config['CHECKPOINT_NAME']}_{flags.model}.png", dpi=200)

    generated,energies = utils.ReverseNorm(voxels,layers,energies[:nevts],
                                           logE=config['logE'],                          
                                           emax = config['EMAX'],emin = config['EMIN'], coordinates=flags.coordinates)
    
    generated[generated<config['ECUT']] = 0 #min from samples

    with h5.File(os.path.join(flags.data_folder,'generated_{}_{}.h5'.format(config['CHECKPOINT_NAME'],flags.model)),"w") as h5f:
        dset = h5f.create_dataset("showers", data=1000*np.reshape(generated,(generated.shape[0],-1)))
        dset = h5f.create_dataset("incident_energies", data=1000*energies)

else:
    def LoadSamples(model):
        generated = []
        energies = []

        if model == "vae_only":
            checkpoint_name = config['CHECKPOINT_NAME'].split("_")[0]
            print("VAE-only checkpoint:",checkpoint_name)
            # checkpoint_name = config['CHECKPOINT_NAME']
        else:
            checkpoint_name = config['CHECKPOINT_NAME']
        
        with h5.File(os.path.join(flags.data_folder,'generated_{}_{}.h5'.format(checkpoint_name,model)),"r") as h5f:
            generated.append(h5f['showers'][:]/1000.)
            energies.append(h5f['incident_energies'][:]/1000.)
            
        energies = np.reshape(energies,(-1,1))
        generated = np.reshape(generated,config['SHAPE'])
        return generated, energies


    if flags.model != 'all':
        models = [flags.model]
        # models = ["vae", "vae_only"]
    else:
        #models = ['VPSDE','subVPSDE','VESDE','wgan','vae']
        models = [flags.model]

        
    energies = []
    data_dict = {}

    if flags.test:
        data = []
        layers = []
        energies = []
        
        for dataset in config['EVAL']:
            data_,layer_,energy_ = utils.DataLoader(
                os.path.join(flags.data_folder,dataset),
                config['SHAPE'],nevts,
                emax = config['EMAX'],emin = config['EMIN'],
                logE=config['logE'],
            )
        
        data.append(data_)
        energies.append(energy_)
        layers.append(layer_)

        data = np.reshape(data,config['SHAPE'])
        layers = np.concatenate(layers)

        data = utils.ApplyPreprocessing(data,"preprocessing_{}_voxel{}.json".format(config['DATASET'], flags.coordinates))
        layers = utils.ApplyPreprocessing(layers,"preprocessing_{}_layers{}.json".format(config['DATASET'], flags.coordinates))
        energies = np.reshape(energies,(-1,1))    
        data,energies = utils.ReverseNorm(data,layers,energies[:nevts],
                                          logE=config['logE'],
                                          emax = config['EMAX'],
                                          emin = config['EMIN'],
                                          coordinates=flags.coordinates
                                          )
    
        data[data<config['ECUT']] = 0 #min from samples
        for model in models:
            data_dict[utils.name_translate[model]]=data
            
    else:        
        for model in models:
            print(f"model: {model}")
            if np.size(energies) == 0:
                data,energies = LoadSamples(model)
                data_dict[utils.name_translate[model]]=data
            else:
                data_dict[utils.name_translate[model]]=LoadSamples(model)[0]

                
    total_evts = energies.shape[0]

    
    data = []
    true_energies = []
    for dataset in config['EVAL']:
        with h5.File(os.path.join(flags.data_folder,dataset),"r") as h5f:
            data.append(h5f['showers'][:total_evts]/1000.)
            true_energies.append(h5f['incident_energies'][:total_evts]/1000.)

    
    data_dict['Geant4']=np.reshape(data,config['SHAPE'])
    data_dict['Geant4'][data_dict['Geant4']<config['ECUT']] = 0 #min from samples
    true_energies = np.reshape(true_energies,(-1,1))
    # print(true_energies.shape[0])
    # input()

    
    #Plot high level distributions and compare with real values
    assert np.allclose(true_energies,energies), 'ERROR: Energies between samples dont match'


    def ScatterESplit(data_dict,true_energies):
        
        def SetFig(xlabel,ylabel):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 1) 
            ax0 = plt.subplot(gs[0])
            ax0.yaxis.set_ticks_position('both')
            ax0.xaxis.set_ticks_position('both')
            ax0.tick_params(direction="in",which="both")    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(xlabel,fontsize=20)
            plt.ylabel(ylabel,fontsize=20)

            ax0.minorticks_on()
            return fig, ax0

        fig,ax = SetFig("Gen. energy [GeV]","Dep. energy [GeV]")
        
        for key in data_dict:
            #print(np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1).shape,true_energies.flatten().shape)
            ax.scatter(
                true_energies.flatten()[:500],
                np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)[:500],
                label=key)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc='best',fontsize=16,ncol=1)
        fig.savefig('{}/FCC_Scatter_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), dpi=600, bbox_inches="tight")


    def AverageShowerWidth(data_dict):
        eta_bins = config['SHAPE'][2]
        eta_binning = np.linspace(-1,1,eta_bins+1)
        eta_coord = [(eta_binning[i] + eta_binning[i+1])/2.0 for i in range(len(eta_binning)-1)]

        def GetMatrix(sizex,sizey,minval=-1,maxval=1):
            nbins = sizex
            binning = np.linspace(minval,maxval,nbins+1)
            coord = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
            matrix = np.repeat(np.expand_dims(coord,-1),sizey,-1)
            return matrix

        
        eta_matrix = GetMatrix(config['SHAPE'][2],config['SHAPE'][3])
        eta_matrix = np.reshape(eta_matrix,(1,1,eta_matrix.shape[0],eta_matrix.shape[1],1))
        
        
        phi_matrix = np.transpose(GetMatrix(config['SHAPE'][3],config['SHAPE'][2]))
        phi_matrix = np.reshape(phi_matrix,(1,1,phi_matrix.shape[0],phi_matrix.shape[1],1))

        def GetCenter(matrix,energies,power=1):
            ec = energies*np.power(matrix,power)
            sum_energies = np.sum(np.reshape(energies,(energies.shape[0],energies.shape[1],-1)),-1)
            ec = np.reshape(ec,(ec.shape[0],ec.shape[1],-1)) #get value per layer
            ec = np.ma.divide(np.sum(ec,-1),sum_energies).filled(0)

            return ec

        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width

        
        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_eta = {}
        feed_dict_eta2 = {}
        
        for key in data_dict:
            feed_dict_phi[key] = GetCenter(phi_matrix,data_dict[key])
            feed_dict_phi2[key] = GetWidth(feed_dict_phi[key],GetCenter(phi_matrix,data_dict[key],2))
            feed_dict_eta[key] = GetCenter(eta_matrix,data_dict[key])
            feed_dict_eta2[key] = GetWidth(feed_dict_eta[key],GetCenter(eta_matrix,data_dict[key],2))
            

        fig,ax0 = utils.PlotRoutine(feed_dict_eta,xlabel='Layer number', ylabel= 'x-center of energy')
        fig.savefig('{}/FCC_EtaEC_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        fig,ax0 = utils.PlotRoutine(feed_dict_phi,xlabel='Layer number', ylabel= 'y-center of energy')
        fig.savefig('{}/FCC_PhiEC_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        fig,ax0 = utils.PlotRoutine(feed_dict_eta2,xlabel='Layer number', ylabel= 'x-width')
        fig.savefig('{}/FCC_EtaW_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        fig,ax0 = utils.PlotRoutine(feed_dict_phi2,xlabel='Layer number', ylabel= 'y-width')
        fig.savefig('{}/FCC_PhiW_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)

        return feed_dict_eta2

    def AverageELayer(data_dict):
        
        def _preprocess(data):
            #print(data.shape,total_evts,config['SHAPE'][1],-1)
            preprocessed = np.reshape(data,(total_evts,config['SHAPE'][1],-1))
            preprocessed = np.sum(preprocessed,-1)
            #preprocessed = np.mean(preprocessed,0)
            return preprocessed
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Mean deposited energy [GeV]')
        fig.savefig('{}/FCC_EnergyZ_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict

    def AverageEX(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,3,1,2,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],config['SHAPE'][3],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed
            
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
    
        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='x-bin', ylabel= 'Mean Energy [GeV]')
        fig.savefig('{}/FCC_EnergyX_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict
        
    def AverageEY(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,2,1,3,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],config['SHAPE'][2],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
    
        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='y-bin', ylabel= 'Mean Energy [GeV]')
        fig.savefig('{}/FCC_EnergyY_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict

    def HistEtot(data_dict):
        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed,-1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

            
        binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),np.quantile(feed_dict['Geant4'],1.0),10)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', ylabel= 'Normalized entries',logy=True,binning=binning)
        ax0.set_xscale("log")
        fig.savefig('{}/FCC_TotalE_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict
        
    def HistNhits(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed>0,-1)
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
            
        # fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Number of hits', ylabel= 'Normalized entries',label_loc='upper left')
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Number of hits', ylabel= 'Normalized entries',label_loc='best')
        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        fig.savefig('{}/FCC_Nhits_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict
    def HistMaxELayer(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],config['SHAPE'][1],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Max. voxel/Dep. energy')
        fig.savefig('{}/FCC_MaxEnergyZ_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict

    def HistMaxE(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(0,1,10)
        fig,ax0 = utils.HistRoutine(feed_dict,ylabel='Normalized entries', xlabel= 'Max. voxel/Dep. energy',binning=binning,logy=True)
        fig.savefig('{}/FCC_MaxEnergy_{}_{}.png'.format(flags.plot_folder,config['CHECKPOINT_NAME'],flags.model), bbox_inches="tight", dpi=600)
        return feed_dict

    def Classifier(data_dict,gen_name='VAE+Diffusion'):
        train = np.concatenate([data_dict["VAE"],data_dict[gen_name]],0)
        labels = np.concatenate([np.zeros((data_dict["VAE"].shape[0],1)),
                                 np.ones((data_dict[gen_name].shape[0],1))],0)
        
        train=train.reshape((train.shape[0],-1))
        train, test, labels, test_labels = train_test_split(train, labels)


        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1,activation='sigmoid')
        ])
        opt = tf.optimizers.Adam(learning_rate=2e-4)
        model.compile(optimizer=opt,
                      loss="binary_crossentropy",
                      metrics=['accuracy'])
        
        model.fit(train, labels,batch_size=100, epochs=3, verbose=2)
        pred = model.predict(test)
        fpr, tpr, _ = roc_curve(test_labels,pred, pos_label=1)    
        print("{} AUC: {}".format(auc(fpr, tpr),gen_name))
    
        geant = data_dict["Geant4"][:20_000]
        geant = geant.reshape((geant.shape[0],-1))
        pred_geant = model.predict(geant)
        
        plt.figure(figsize=(8,6))
        plt.hist(pred[np.where(test_labels==0)], bins=20, color="#51b841", label="VAE", histtype="step", lw=3)
        plt.hist(pred[np.where(test_labels==1)],bins=20, color="#4151b8", label="VAE+Diffusion",  histtype="step", lw=3)
        plt.hist(pred_geant,bins=20, color="black", label="Geant4",  histtype="step", lw=2.5, linestyle="dashed")
        plt.xlabel("Classifier Probability")
        plt.yscale("log")
        plt.ylabel("Entries")
        plt.legend()
        plt.savefig(f"Classifier_preds_{config['CHECKPOINT_NAME']}_models.png", dpi=500, bbox_inches="tight")
        
        
        print(f"Average prediction: {np.mean(pred_geant, axis=0)}")

    def Plot_Shower_2D(data_dict):
        #cmap = plt.get_cmap('PiYG')
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad("white")
        plt.rcParams['pcolor.shading'] ='nearest'
        layer_number = [10,44]
        
        def SetFig(xlabel,ylabel):
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 1) 
            ax0 = plt.subplot(gs[0])
            ax0.yaxis.set_ticks_position('both')
            ax0.xaxis.set_ticks_position('both')
            ax0.tick_params(direction="in",which="both")    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(xlabel,fontsize=20)
            plt.ylabel(ylabel,fontsize=20)

            ax0.minorticks_on()
            return fig, ax0

        for layer in layer_number:
            
            def _preprocess(data):
                preprocessed = data[:,layer,:]
                preprocessed = np.mean(preprocessed,0)
                preprocessed[preprocessed==0]=np.nan
                return preprocessed

            vmin=vmax=0
            for ik,key in enumerate(['Geant4',utils.name_translate[flags.model]]):
                fig,ax = SetFig("x-bin","y-bin")
                average = _preprocess(data_dict[key])
                if vmax==0:
                    vmax = np.nanmax(average[:,:,0])
                    vmin = np.nanmin(average[:,:,0])
                    #print(vmin,vmax)
                im = ax.pcolormesh(range(average.shape[1]), range(average.shape[0]), average[:,:,0], cmap=cmap)

                yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
                yScalarFormatter.set_powerlimits((0,0))
                #cbar.ax.set_major_formatter(yScalarFormatter)

                cbar=fig.colorbar(im, ax=ax,label='Dep. energy [GeV]',format=yScalarFormatter)
                
                
                bar = ax.set_title("{}, layer number {}".format(key,layer),fontsize=15)

                fig.savefig('{}/FCC_{}2D_{}_{}_{}.png'.format(flags.plot_folder,key,layer,config['CHECKPOINT_NAME'],flags.model))
            

    high_level = []
    plot_routines = {
        'Energy per layer':AverageELayer,
        'Energy':HistEtot,
        '2D Energy scatter split':ScatterESplit,
        'Nhits':HistNhits,
    }
    
    if '1' in flags.config:
        pass
        plot_routines['Max voxel']=HistMaxE
    else:
        # pass
        plot_routines['Shower width']=AverageShowerWidth        
        plot_routines['Energy per eta']=AverageEX
        plot_routines['Energy per phi']=AverageEY
        plot_routines['2D average shower']=Plot_Shower_2D
        plot_routines['Max voxel']=HistMaxELayer
        if run_classifier:
            plot_routines['Class']=Classifier

        
    for plot in plot_routines:
        if '2D' in plot and flags.model == 'all':continue #skip scatter plots superimposed
        print(plot)
        if 'split' in plot:
            plot_routines[plot](data_dict,true_energies)
        else:
            high_level.append(plot_routines[plot](data_dict))
            
