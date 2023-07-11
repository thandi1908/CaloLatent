import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, initializers
import horovod.tensorflow.keras as hvd
import utils
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow_probability import distributions
from architectures import Encoder, Decoder, Resnet


def soft_clamp(layer,n=5.0):
    return n*tf.math.tanh(layer/n)
    

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_sig = inputs
        batch = tf.shape(z_mean)[0]
        
        z = distributions.MultivariateNormalDiag(
            loc = z_mean,scale_diag=tf.exp(z_log_sig)).sample()
        
        return z

class CaloLatent(keras.Model):
    """CaloLatent model"""
    def __init__(self, data_shape,num_cond,config,name='vae'):
        super(CaloLatent, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        self.num_cond = num_cond
        self.data_shape = data_shape
        self.config = config
        self.read_config()
        self.sigma2_0 = 3e-5
        self.sigma2_1 = 0.99
        self.beta_0 = 0.1
        self.beta_1 = 20.0

        self.activation = tf.keras.activations.swish
        
        self.kl_steps=500*624//hvd.size() #Number of optimizer steps to take before kl is multiplied by 1
        self.warm_up_steps = 500*624//hvd.size() #number of steps to train the VAE alone
        self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank        

        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
        else:
            self.shape = (-1,1,1,1,1)

        #Time and conditional embedding
        
        inputs_cond = Input((self.num_cond))
        inputs_time = Input((1))


        
        #Time embedding for the diffusion models
        self.projection = self.GaussianFourierProjection(scale = 16)
        latent_time = self.Embedding(inputs_time,self.projection)
        layer_time = self.Embedding(inputs_time,self.projection)


        #2 Diffusion models, one that learns the latent space, conditioned on the energy deposition per layer, and one that learns only the energy depositions per layer


        #Energy per layer model
        cond_layer = self.activation(layers.Dense(self.num_embed,activation=None)(inputs_cond))
        cond_layer = self.activation(layers.Dense(self.num_embed,activation=None)(tf.concat([cond_layer,layer_time],-1)))
        inputs_layer,outputs_layer = self.ScoreModel(self.num_layer,cond_layer)
        self.layer_energy = keras.models.Model([inputs_layer,inputs_time,inputs_cond], outputs_layer, name="score")

        
        cond_embed = self.activation(layers.Dense(self.num_embed,activation=None)(tf.concat([inputs_cond,inputs_layer],-1)))

    

        #Encoder and decoder are conditioned on the initial particle energy and in the eneergy deposited per layer, reuse the conditional embedding from before


        #Encoder model
        inputs_encoder,z_mean, z_log_sig, z = self.Encoder(cond_embed)
        self.encoder = keras.Model([inputs_encoder,inputs_cond,inputs_layer], [z_mean, z_log_sig, z], name="encoder")


        #Latent model
        cond_latent = self.activation(layers.Dense(self.num_embed,activation=None)(tf.concat([cond_embed,latent_time],-1)))
        inputs_latent,outputs_latent = self.ScoreModel(self.latent_dim,cond_latent)
        self.latent_diffusion = keras.models.Model([inputs_latent,inputs_time,inputs_cond,inputs_layer], outputs_latent, name="score")



        
        #Decoder Model
        inputs_decoder,outputs_decoder=self.Decoder(cond_embed)
        self.decoder = keras.models.Model([inputs_decoder,inputs_cond,inputs_layer], outputs_decoder, name="generator")

        #Learn the mixture between normal and non-normal components
        self.mixing_logit = tf.Variable(tf.zeros((self.latent_dim)),trainable=True) 

        # if self.verbose:
        #     print(self.encoder.summary())
        #     print(self.decoder.summary())
        #     print(self.latent_diffusion.summary())


        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="rec_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.score_loss_tracker = keras.metrics.Mean(name="score_loss")
        self.layer_loss_tracker = keras.metrics.Mean(name="layer_loss")


    def read_config(self):
        self.num_embed = self.config['EMBED']
        self.projection_dim = self.config['NOISE_DIM'] #latent space dimensionality
        self.num_steps = self.config['NSTEPS']
        self.num_layer = self.config['NLAYER']
        self.snr=self.config['SNR']

        
    
    def Encoder(self,cond_embed):        
        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
            use_1D = True
            inputs,outputs = Encoder(
                self.data_shape,
                cond_embed,
                input_embedding_dims = 16,
                stride=2,
                kernel=3,
                block_depth = 3,
                widths = [16,32,64,96],
                attentions = [False, True,True, True],
                pad=self.config['PAD'],
                use_1D=use_1D
            )

            
        else:
            self.shape = (-1,1,1,1,1)
            use_1D = False

            inputs,outputs = Encoder(
                self.data_shape,
                cond_embed,
                input_embedding_dims = 32,
                stride=2,
                kernel=3,
                block_depth = 2,
                widths = [32,64,96],
                attentions = [False,False,True],
                pad=self.config['PAD'],
                use_1D=use_1D
            )



            # print("last",layer_encoded)
            
            z_mean = layers.Conv3D(self.projection_dim,kernel_size=1,padding="same",
                                   kernel_initializer=initializers.Zeros(),
                                   bias_initializer=initializers.Zeros(),
                                   strides=1,activation=None,use_bias=True)(outputs)
        
            z_log_sig = layers.Conv3D(self.projection_dim,kernel_size=1,padding="same",
                                      kernel_initializer=initializers.Zeros(),
                                      bias_initializer=initializers.Zeros(),
                                      strides=1,activation=None,use_bias=True)(outputs)
            
            self.init_shape = z_mean.shape[1:]            
            z_mean = layers.Flatten()(z_mean)
            z_mean = soft_clamp(z_mean)

            z_log_sig = layers.Flatten()(z_log_sig)
            z_log_sig = soft_clamp(z_log_sig)
        
        
            z = Sampling()([z_mean, z_log_sig])
            self.latent_dim = z.shape[-1]

            print(f"Model latent dimensions: {self.latent_dim}")

        return  inputs, z_mean, z_log_sig, z


        
    
    def Decoder(self,cond_embed):        
        inputs = Input((self.latent_dim))
        layer_decoded = layers.Reshape(self.init_shape)(inputs)

        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
            use_1D = True
            outputs = Decoder(
                layer_decoded,
                cond_embed,
                stride=2,
                kernel=3,
                block_depth = 3,
                widths = [16,32,64,96],
                attentions = [False, True,True, True],
                pad=self.config['PAD'],
                use_1D=use_1D
            )

            
        else:
            self.shape = (-1,1,1,1,1)
            use_1D = False
            outputs = Decoder(
                layer_decoded,
                cond_embed,
                stride=2,
                kernel=3,
                block_depth = 2,
                widths = [32,64,96],
                attentions = [False,False, True],
                pad=self.config['PAD'],
                use_1D=use_1D
            )

        outputs_mean = layers.Conv3D(1,kernel_size=1,padding="same",
                                     kernel_initializer=initializers.Zeros(),
                                     bias_initializer=initializers.Zeros(),
                                     strides=1,activation=None,use_bias=True)(outputs)
        outputs_sigma = layers.Conv3D(1,kernel_size=1,padding="same",
                                      kernel_initializer=initializers.Zeros(),
                                      bias_initializer=initializers.Zeros(),
                                      strides=1,activation=None,use_bias=True)(outputs)
        #outputs_mean = layers.LeakyReLU(0.01)(outputs_mean)
        #outputs_sigma = soft_clamp(outputs_sigma)
        #outputs_mean = soft_clamp(outputs_mean,1)
        outputs=tf.concat([outputs_mean,outputs_sigma],-1)
        
        return  inputs,outputs



    
    def ScoreModel(self,ndim,time_embed,
                   num_layer=3,mlp_dim=128):
        inputs,outputs = Resnet(ndim,
                                time_embed,
                                num_layer = num_layer,
                                mlp_dim=mlp_dim,
                                )
                                
        return inputs,outputs

    def marginal_prob(self,t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_mean_coeff = tf.reshape(log_mean_coeff,(-1,1))

        mean = tf.exp(log_mean_coeff)
        var = 1 - tf.exp(2. * log_mean_coeff)
        
        mean = tf.reshape(mean,(-1,1))
        var = tf.reshape(var,(-1,1))
        return mean, var

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions)
    
    def inv_var(self,var):
        #Return inverse variance for importance sampling

        c = tf.math.log(1 - var)
        a = self.beta_1 - self.beta_0
        t = (-self.beta_0 + tf.sqrt(tf.square(self.beta_0) - 2 * a * c)) /a 
        return t

    def sde(self, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        beta_t = tf.reshape(beta_t,(-1,1))
        drift = -0.5 * beta_t
        diffusion2 = beta_t
        return drift, diffusion2
        
    def GaussianFourierProjection(self,scale = 30):
        half_dim = self.num_embed // 4
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq

    def Embedding(self,inputs,projection):
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding

        
    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.score_loss_tracker,
            self.layer_loss_tracker
        ]

    def compile(self,vae_optimizer, sgm_optimizer,layer_optimizer):
        super(CaloLatent, self).compile(experimental_run_tf_function=False,
                                        weighted_metrics=[],
                                        #run_eagerly=True
        )
        self.vae_optimizer = vae_optimizer
        self.sgm_optimizer = sgm_optimizer
        self.layer_optimizer = layer_optimizer


    def cross_entropy_const(self,t=1e-5):
        return 0.5 * (1.0 + tf.math.log(2.0 * np.pi * self.marginal_prob(t)[1]))
    
    
    def reset_opt(self,opt):
        for var in opt.variables():
            var.assign(tf.zeros_like(var))
        return tf.constant(10)

    def reconstruction_loss(self,data,mean,log_std,axis):
        rec = tf.square(data-mean)                
        rec = tf.reduce_sum(rec,axis)
        return tf.reduce_mean(rec)

    def get_diffusion_loss(self,inputs,conditionals,batch,model,weighted=True,use_mixing=True,eps=1e-5):

        random_t = tf.random.uniform((batch,1))*(1-eps) + eps 

        if weighted:
            # using importance sampling for t
            ones = tf.ones_like(random_t)
            sigma2_1, sigma2_eps = self.marginal_prob(ones)[1], self.marginal_prob(eps * ones)[1]
            log_sigma2_1, log_sigma2_eps = tf.math.log(sigma2_1), tf.math.log(sigma2_eps)
            var = tf.exp(random_t * log_sigma2_1 + (1 - random_t) * log_sigma2_eps)            
            random_t = self.inv_var(var)
            mean = self.marginal_prob(random_t)[0]
        else: 
            # using uniform sampling for t 
            mean,var = self.marginal_prob(random_t) 
        
        f,g2 = self.sde(random_t)
        
        noise = tf.random.normal((tf.shape(inputs)))            
        perturbed_inputs = inputs*mean + tf.sqrt(var)*noise
        pred_params_q = model([perturbed_inputs,random_t]+conditionals)

        if use_mixing:
            mixing_component = perturbed_inputs*tf.sqrt(var)
            #Trainable mixing
            coeff = tf.math.sigmoid(self.mixing_logit)
            #coeff = 1
            params = (1. - coeff) * mixing_component + coeff * pred_params_q
        else:
            coeff = 1.0
            params = pred_params_q
            
        cross_entropy_per_var = tf.square(params - noise)
        if weighted:
            # cross_entropy_per_var *= g2/(2*var)
            cross_entropy_per_var *= (0.5 * (log_sigma2_1 - log_sigma2_eps) / (1.0 - var))

        return  cross_entropy_per_var, coeff
    
    def train_step(self, inputs):
        voxel,layer,cond = inputs
        batch = tf.shape(voxel)[0]
        
        if len(self.data_shape) == 2:
            axis=(1,2)
        else:
            axis=(1,2,3,4)

        
        with tf.GradientTape() as tape:
            z_mean, z_log_sig, z = self.encoder([voxel,cond,layer])
                        
            vae_neg_entropy = distributions.MultivariateNormalDiag(
                loc=z_mean,allow_nan_stats=False,
                scale_diag=tf.exp(z_log_sig)).log_prob(z)

            vae_logp = distributions.MultivariateNormalDiag(
                loc=tf.zeros_like(z_mean),allow_nan_stats=False,
                scale_diag=tf.ones_like(z_mean)).log_prob(z)
            
            #VAE reconstruction
            rec,log_std= tf.split(self.decoder([z,cond,layer]),num_or_size_splits=2, axis=-1)
            reconstruction_loss = self.reconstruction_loss(voxel,rec,log_std,axis=axis)
            
            cross_entropy_per_var,coeff = self.get_diffusion_loss(z,[cond,layer],batch,model=self.latent_diffusion,weighted=True)
            #add constant entropy term

            #cross_entropy_per_var += self.cross_entropy_const()
            cross_entropy = tf.reduce_sum(cross_entropy_per_var,-1)
            

            kl_loss_joint = cross_entropy + vae_neg_entropy 
            kl_loss_disjoint = vae_neg_entropy - vae_logp
            
            kl_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
                              lambda: kl_loss_joint,lambda:kl_loss_disjoint)
            

            kl_loss = tf.reduce_mean(kl_loss)

            
            #Simple linear scaling
            beta = tf.math.minimum(1.0,tf.cast(self.vae_optimizer.iterations,tf.float32)/self.kl_steps)
            total_loss = beta*kl_loss + reconstruction_loss


        vae_weights = self.encoder.trainable_weights + self.decoder.trainable_weights 
        tf.cond(self.warm_up_steps == self.sgm_optimizer.iterations,
                lambda: self.reset_opt(self.vae_optimizer), lambda: tf.constant(10))
            
                
        grads = tape.gradient(total_loss, vae_weights)
        grads = [tf.clip_by_norm(grad, 1)
                 for grad in grads]
        self.vae_optimizer.apply_gradients(zip(grads, vae_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(beta*kl_loss)

        #Latent Diffusion training

        with tf.GradientTape() as tape:
            score_latent_loss,coeff = self.get_diffusion_loss(z,[cond,layer],batch,model=self.latent_diffusion,weighted=False)
            score_latent_loss = tf.reduce_sum(score_latent_loss, axis=-1)
            score_latent_loss = tf.reduce_mean(score_latent_loss)
            #Only start score training after the warmup
            score_latent_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
                                 lambda: score_latent_loss,lambda:0*score_latent_loss)
            

        grads = tape.gradient(score_latent_loss, self.latent_diffusion.trainable_weights+[self.mixing_logit])
        grads = [tf.clip_by_norm(grad, 1)
                 for grad in grads]

        self.sgm_optimizer.apply_gradients(zip(grads, self.latent_diffusion.trainable_weights+[self.mixing_logit]))
        self.score_loss_tracker.update_state(score_latent_loss)
        
        


        #Energy per Layer Diffusion


        with tf.GradientTape() as tape:
            score_layer_loss,_ = self.get_diffusion_loss(layer,[cond],batch,model = self.layer_energy,weighted=False,use_mixing=False)
            score_layer_loss = tf.reduce_sum(score_layer_loss, axis=-1)
            score_layer_loss = tf.reduce_mean(score_layer_loss)
            #Only start score training after the warmup

        grads = tape.gradient(score_layer_loss, self.layer_energy.trainable_weights)
        grads = [tf.clip_by_norm(grad, 1)
                 for grad in grads]

        self.layer_optimizer.apply_gradients(zip(grads, self.layer_energy.trainable_weights))
        self.layer_loss_tracker.update_state(score_layer_loss)

        
        return {
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "latent_loss":self.score_loss_tracker.result(),
            "layer_loss":self.layer_loss_tracker.result(), 
            "beta":beta,            
            "mixing":tf.reduce_max(coeff)
        }

    def test_step(self, inputs):

        voxel,layer,cond = inputs
        batch = tf.shape(voxel)[0]
        
        if len(self.data_shape) == 2:
            axis=(1,2)
        else:
            axis=(1,2,3,4)

        
            
        z_mean, z_log_sig, z = self.encoder([voxel,cond,layer])
                        
        vae_neg_entropy = distributions.MultivariateNormalDiag(
            loc=z_mean,allow_nan_stats=False,
            scale_diag=tf.exp(z_log_sig)).log_prob(z)
        
        vae_logp = distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(z_mean),allow_nan_stats=False,
            scale_diag=tf.ones_like(z_mean)).log_prob(z)
        
        #VAE reconstruction
        rec,log_std= tf.split(self.decoder([z,cond,layer]),num_or_size_splits=2, axis=-1)
        reconstruction_loss = self.reconstruction_loss(voxel,rec,log_std,axis=axis)
        
        cross_entropy_per_var,coeff = self.get_diffusion_loss(z,[cond,layer],batch,model=self.latent_diffusion,weighted=True)
        #add constant entropy term
        
        #cross_entropy_per_var += self.cross_entropy_const()
        cross_entropy = tf.reduce_sum(cross_entropy_per_var,-1)
        
        
        kl_loss_joint = cross_entropy + vae_neg_entropy 
        kl_loss_disjoint = vae_neg_entropy - vae_logp
        
        kl_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
                          lambda: kl_loss_joint,lambda:kl_loss_disjoint)
        
        kl_loss = tf.reduce_mean(kl_loss)
        
        
        #Simple linear scaling
        beta = tf.math.minimum(1.0,tf.cast(self.vae_optimizer.iterations,tf.float32)/self.kl_steps)
        total_loss = beta*kl_loss + reconstruction_loss

        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(beta*kl_loss)

        #Latent Diffusion training


        score_latent_loss,coeff = self.get_diffusion_loss(z,[cond,layer],batch,model=self.latent_diffusion,weighted=False)
        score_latent_loss = tf.reduce_sum(score_latent_loss, axis=-1)
        score_latent_loss = tf.reduce_mean(score_latent_loss)
        #Only start score training after the warmup
        score_latent_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
                                    lambda: score_latent_loss,lambda:0*score_latent_loss)
            

        self.score_loss_tracker.update_state(score_latent_loss)
        
        


        #Energy per Layer Diffusion


        
        score_layer_loss,_ = self.get_diffusion_loss(layer,[cond],batch,model = self.layer_energy,weighted=False,use_mixing=False)
        score_layer_loss = tf.reduce_sum(score_layer_loss, axis=-1)
        score_layer_loss = tf.reduce_mean(score_layer_loss)
        #Only start score training after the warmup

        self.layer_loss_tracker.update_state(score_layer_loss)

        total_loss = beta*kl_loss + reconstruction_loss
        
        return {
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "latent_loss":score_latent_loss,
            "layer_loss":score_layer_loss, 
            "beta":beta,            
            "mixing":tf.reduce_max(coeff),
            "loss":total_loss,
        }

    
    def generate(self,nevts,cond):

        layer_energies = self.PCSampler([cond], batch_size = cond.shape[0],
                                        ndim=self.num_layer,
                                        model = self.layer_energy,
                                        num_steps=self.num_steps,
                                        snr=self.snr).numpy()
        
        
        random_latent_vectors = tf.random.normal(
            shape=(nevts, self.latent_dim)
        )
        
        random_latent_vectors =self.PCSampler([cond,layer_energies],
                                              batch_size = cond.shape[0],
                                              ndim=self.latent_dim,
                                              model=self.latent_diffusion,
                                              use_mixing=True,
                                              num_steps=self.num_steps,
                                              snr=self.snr)
        
        mean,log_std= tf.split(self.decoder([random_latent_vectors,cond,layer_energies], training=False),num_or_size_splits=2, axis=-1)
                            
        # print(tf.exp(std))
        # input()
        return mean,layer_energies


    @tf.function
    def _solve_ode(self, ode_fn, state, atol=1e-4,**kwargs):
        """Solves the initial value problem defined by `ode_fn`.
        Args:
        ode_fn: `Callable(time, state)` that represents state time derivative.
        state: A `Tensor` representing initial state.
        **kwargs: Additional arguments to pass to ode_solve_fn.
        Returns:
        final_state: `Tensor` of the same shape and dtype as `state` representing
        the solution of ODE defined by `ode_fn` at `self._final_time`.
        """
        self._ode_solver = tfp.math.ode.DormandPrince(
            atol=atol,
            rtol=atol,
        )
        self._ode_solve_fn = self._ode_solver.solve
        
        integration_result = self._ode_solve_fn(
            ode_fn=ode_fn,
            initial_time=self._initial_time,
            initial_state=state,
            solution_times=[self._final_time],
            **kwargs)
        final_state = tf.nest.map_structure(
            lambda x: x[-1], integration_result.states)
        return final_state


    def ODESampler(self,cond,atol=1e-4,eps=1e-5):
        import time
        batch_size = cond.shape[0]
        t = tf.ones((batch_size,1))
        self._initial_time = eps
        self._final_time = 1

        samples = []
        def score_eval_wrapper(sample, time_steps,cond):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = tf.convert_to_tensor(sample, dtype=tf.float32)
            time_steps = tf.convert_to_tensor(time_steps, dtype=tf.float32)
            cond = tf.convert_to_tensor(cond, dtype=tf.float32)
            score = self.score([sample, time_steps,cond])
            mean,var = self.marginal_prob(time_steps)
            mixing_component = sample*tf.sqrt(var)            
            coeff = tf.math.sigmoid(self.mixing_logit)
            params = (1 - coeff) * mixing_component + coeff * score
            score=-params/tf.sqrt(var)
            
            return score

        def augmented_ode_fn(cond,final_time=1):
            def ode_fn(t, aug_state):
                state = aug_state
                """The ODE function for use by the ODE solver."""
                time_steps = np.ones((batch_size)) * t
                f,g2 = self.sde(time_steps)
                var = self.marginal_prob(time_steps)[1]
                return  f*state - 0.5 * g2 * score_eval_wrapper(state, time_steps,cond)

            def reverse_ode_fn(t,x):
                """ solver does not work if the final time is bigger than the initial time """
                return -ode_fn(self._final_time - t,x)

            return reverse_ode_fn

        start = time.time()
        init_x = tf.random.normal((batch_size,self.latent_dim))                
        augmented_x = (init_x)        

        ode_fn = augmented_ode_fn(cond,self._final_time)
        img=self._solve_ode(ode_fn, augmented_x,atol=atol)
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(batch_size,end - start))
        return img


    def PCSampler(self,
                  cond,
                  batch_size,
                  ndim,
                  model,
                  num_steps=200, 
                  snr=0.1,
                  use_mixing=False,
                  ncorrections=1,
                  eps=1e-5):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        import time
        t = tf.ones((batch_size,1))
        data_shape = [batch_size,ndim]
        
        #cond = tf.convert_to_tensor(cond, dtype=tf.float32)
        #cond = tf.reshape(cond,(-1,self.num_cond))
        init_x = self.prior_sde(data_shape)
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]        
        x = init_x

        start = time.time()
        for istep,time_step in enumerate(time_steps):      
            batch_time_step = tf.ones((batch_size,1)) * time_step
            z = tf.random.normal(x.shape)
            mean,var = self.marginal_prob(batch_time_step)
            
            score = model([x, batch_time_step]+cond)
            if use_mixing:            
                mixing_component = x*tf.sqrt(var)
                coeff = tf.math.sigmoid(self.mixing_logit)
                params = (1 - coeff) * mixing_component + coeff * score
                score=-params/tf.sqrt(var)
            else:
                score = -score/tf.sqrt(var)

            for _ in range(ncorrections):
                # Corrector step (Langevin MCMC)
                grad = score
                noise = tf.random.normal(x.shape)
                
                grad_norm = tf.reduce_mean(tf.norm(tf.reshape(grad,(grad.shape[0],-1)),axis=-1,keepdims =True),-1)
                grad_norm = tf.reduce_mean(grad_norm)
                
                noise_norm = tf.reduce_mean(tf.norm(tf.reshape(noise,(noise.shape[0],-1)),axis=-1,keepdims =True),-1)
                noise_norm = tf.reduce_mean(noise_norm)
                
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                langevin_step_size = tf.reshape(langevin_step_size,(-1,1))
                x_mean = x + langevin_step_size * grad
                x =  x_mean + tf.math.sqrt(2 * langevin_step_size) * noise


            
            

            # Predictor step (Euler-Maruyama)

            # score = self.score([x, batch_time_step,cond])
            # mixing_component = x*tf.sqrt(var)            
            # coeff = tf.math.sigmoid(self.mixing_logit)
            # params = (1 - coeff) * mixing_component + coeff * score
            # score=-params/tf.sqrt(var)
            
            drift,diffusion2 = self.sde(batch_time_step)
            drift = drift*x - diffusion2 * score
            x_mean = x - drift * step_size            
            x = x_mean + tf.math.sqrt(diffusion2*step_size) * z

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(batch_size,end - start))
        # The last step does not include any noise
        return x_mean

