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
        
        self.kl_steps=int(150*624//hvd.size()) #Number of optimizer steps to take before kl is multiplied by 1
        self.warm_up_steps = int(10e15*624//hvd.size()) #number of steps to train the VAE alone
        self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank        

        if len(self.data_shape) == 2:
            self.shape = (-1,1,1)
        else:
            self.shape = (-1,1,1,1,1)

        #Time and conditional embedding
        
        inputs_cond = Input((self.num_cond))
        inputs_time = Input((1))

        self.projection = self.GaussianFourierProjection(scale = 16)
        cond_embed =  layers.Dense(2*self.num_embed,activation=None)(inputs_cond)
        cond_embed =  self.activation(layers.Dense(self.num_embed,activation=None)(inputs_cond))


        #Encoder model
        inputs_encoder,z_mean, z_log_sig, z = self.Encoder(cond_embed)
        self.encoder = keras.Model([inputs_encoder,inputs_cond], [z_mean, z_log_sig, z], name="encoder")
        self.mixing_logit = tf.Variable(tf.zeros((self.latent_dim)),trainable=True) #Learn the mixture between normal and non-normal components
        
        #Decoder Model
        inputs_decoder,outputs_decoder=self.Decoder(cond_embed)
        self.decoder = keras.models.Model([inputs_decoder,inputs_cond], outputs_decoder, name="generator")

        time_embed = self.Embedding(inputs_time,self.projection)
        time_embed = layers.Dense(2*self.num_embed)(time_embed)
        time_embed = self.activation(layers.Dense(self.num_embed)(time_embed))
        time_embed = tf.concat([time_embed,cond_embed],-1)        
        
        ## Diffusion Model
        inputs_score,outputs_score = self.ScoreModel(time_embed)
        outputs_score = outputs_score/tf.sqrt(self.marginal_prob(inputs_time)[1])
        self.score = keras.models.Model([inputs_score,inputs_time,inputs_cond], outputs_score, name="score")

        # if self.verbose:
        #     print(self.encoder.summary())
        #     print(self.decoder.summary())
        #     print(self.score.summary())


        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="rec_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.score_loss_tracker = keras.metrics.Mean(name="score_loss")


    def read_config(self):
        self.num_embed = self.config['EMBED']
        self.projection_dim = self.config['NOISE_DIM'] #latent space dimensionality
        self.ld_multiplier = self.config['LD_MULTI']
        self.num_steps = self.config['NSTEPS']
        self.snr=self.config['SNR']

        
    def ResidualBlock(self,input_layer,embed,hidden_size,kernel_size=3,padding="same"):

        cond = keras.activations.swish(embed)
        cond = layers.Dense(hidden_size,activation=None,use_bias=False)(embed)
        
        if len(self.data_shape) == 2:
            residual = tfa.layers.SpectralNormalization(
                layers.Conv1D(hidden_size, kernel_size=1))(input_layer)
        else:            
            residual = tfa.layers.SpectralNormalization(
                layers.Conv3D(hidden_size, kernel_size=1))(input_layer)
            

        x = tfa.layers.GroupNormalization(epsilon=1e-5,groups=1)(input_layer)
        # x = layers.BatchNormalization(center=False, scale=False)(input_layer)
        x = keras.activations.swish(x)
        
        
        
        #x = input_layer
        if len(self.data_shape) == 2:
            x = tfa.layers.SpectralNormalization(
                layers.Conv1D(hidden_size,kernel_size=3,
                              activation=self.activation,padding=padding))(x)
            x = tfa.layers.SpectralNormalization(
                layers.Conv1D(hidden_size,kernel_size=3,padding=padding))(x)
        else:
            x = layers.ZeroPadding3D(1)(x)
            x = tfa.layers.SpectralNormalization(
                layers.Conv3D(hidden_size,kernel_size=kernel_size,
                              activation=None,padding=padding))(x)
            x = layers.Add()([x, cond])
            # x = layers.BatchNormalization(center=False, scale=False)(x)
            x = tfa.layers.GroupNormalization(epsilon=1e-5,groups=1)(x)
            x = keras.activations.swish(x)
            x = layers.ZeroPadding3D(1)(x)
            x = tfa.layers.SpectralNormalization(
                layers.Conv3D(hidden_size,kernel_size=kernel_size,padding=padding))(x)
            
        if use_residual:
            x = layers.Add()([x, residual])

        return x

    
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
                attentions = [False,False, True],
                pad=self.config['PAD'],
                use_1D=use_1D
            )



            # print("last",layer_encoded)
            
            z_mean = layers.Conv3D(self.ld_multiplier,kernel_size=1,padding="same",
                                   kernel_initializer=initializers.Zeros(),
                                   bias_initializer=initializers.Zeros(),
                                   strides=1,activation=None,use_bias=True)(outputs)
        
            z_log_sig = layers.Conv3D(self.ld_multiplier,kernel_size=1,padding="same",
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
            print(f"Model Latent Dimensions: {self.latent_dim}")

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



    
    def ScoreModel(self,time_embed):
        inputs,outputs = Resnet(self.latent_dim,
                                time_embed,
                                num_layer = 5,
                                mlp_dim=256,
                                )
                                
        return inputs,outputs

    def marginal_prob(self,t):
    
        # var = self.sigma2_0 * ((self.sigma2_1 / self.sigma2_0) ** t)        
        # mean = tf.sqrt(1.0 + self.sigma2_0 * (1.0 - (self.sigma2_1 / self.sigma2_0) ** t) / (1.0 - self.sigma2_0))

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
                    
        return tf.math.log(var/self.sigma2_0)/tf.math.log(self.sigma2_1 / self.sigma2_0)


    def sde(self, t):
        #https://github.com/NVlabs/LSGM/blob/main/diffusion_continuous.py
        # sigma2_geom = self.sigma2_0 * ((self.sigma2_1 / self.sigma2_0) ** t)
        # log_term = tf.math.log(self.sigma2_1 / self.sigma2_0)
        # diffusion2= sigma2_geom * log_term / (1.0 - sigma2_geom)
        # diffusion2 = tf.reshape(diffusion2,(-1,1))
        # drift = -0.5*diffusion2


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
        ]

    def compile(self,vae_optimizer, sgm_optimizer):
        super(CaloLatent, self).compile(experimental_run_tf_function=False,
                                        #run_eagerly=True
        )
        self.vae_optimizer = vae_optimizer
        self.sgm_optimizer = sgm_optimizer


    def cross_entropy_const(self,t):
        return 0.5 * (1.0 + tf.math.log(2.0 * np.pi * self.marginal_prob(t)[1]))
    
    def CrossEntropy(self,logit,label):
        epsilon = 1e-5
        
        y_pred = tf.clip_by_value(tf.reshape(logit,(tf.shape(logit)[0],-1)), epsilon, 1. - epsilon)
        y_label = tf.reshape(label,(tf.shape(logit)[0],-1))
        ce = -y_label*tf.math.log(y_pred) - (1-y_label)*tf.math.log(1-y_pred)
        ce = tf.reshape(ce,tf.shape(logit))
        return ce
    
    def reset_opt(self,opt):
        for var in opt.variables():
            var.assign(tf.zeros_like(var))
        return tf.constant(10)

    def reconstruction_loss(self,data,mean,log_std,axis):
        rec = tf.square(data-mean)                
        rec = tf.reduce_sum(rec,axis)
        return tf.reduce_mean(rec)

    def get_diffusion_entropy(self,z,cond,batch,t=None,eps=1e-3,weighted=True):
        if t is None:
            random_t = tf.random.uniform((batch,1))*(1-eps) + eps
        else:
            random_t = t
            
        mean,var = self.marginal_prob(random_t)
        diffusion2 = self.sde(random_t)[1]
        
        if weighted:
            weight_t = diffusion2/(2*var)
        else:
            weight_t = 1.0/2
            
        ones = tf.ones_like(random_t)
        sigma2_1, sigma2_eps = self.marginal_prob(ones)[1], self.marginal_prob(eps * ones)[1]
        log_sigma2_1, log_sigma2_eps = tf.math.log(sigma2_1), tf.math.log(sigma2_eps)
        var = tf.exp(random_t * log_sigma2_1 + (1 - random_t) * log_sigma2_eps)            
        random_t = self.inv_var(var)
        mean = self.marginal_prob(random_t)[0]                        
        weight_t = 0.5 * (log_sigma2_1 - log_sigma2_eps) / (1.0 - var)

        
        noise = tf.random.normal((tf.shape(z)))            
        perturbed_z = z*mean + tf.sqrt(var)*noise
        mixing_component = perturbed_z*tf.sqrt(var)
        pred_params_q = self.score([perturbed_z,random_t,cond])

        #Trainable mixing
        coeff = tf.math.sigmoid(self.mixing_logit)
        #coeff = 1
        params = (1 - coeff) * mixing_component + coeff * pred_params_q
        l2_term_q = tf.square(params - noise)
        cross_entropy_per_var = weight_t * l2_term_q

        return  cross_entropy_per_var, coeff, random_t
    
    def train_step(self, inputs):
        eps=1e-5
        data,cond = inputs
        batch = tf.shape(data)[0]
        
        if len(self.data_shape) == 2:
            axis=(1,2)
        else:
            axis=(1,2,3,4)

        
        with tf.GradientTape() as tape:
            z_mean, z_log_sig, z = self.encoder([data,cond])
                        
            vae_neg_entropy = distributions.MultivariateNormalDiag(
                loc=z_mean,allow_nan_stats=False,
                scale_diag=tf.exp(z_log_sig)).log_prob(z)

            vae_logp = distributions.MultivariateNormalDiag(
                loc=tf.zeros_like(z_mean),allow_nan_stats=False,
                scale_diag=tf.ones_like(z_mean)).log_prob(z)
            
            #VAE reconstruction
            rec,log_std= tf.split(self.decoder([z,cond]),num_or_size_splits=2, axis=-1)
            reconstruction_loss = self.reconstruction_loss(data,rec,log_std,axis=axis)
            
            cross_entropy_per_var,coeff,random_t = self.get_diffusion_entropy(z,cond,batch,eps=eps)
            #add constant entropy term

            #cross_entropy_per_var += self.cross_entropy_const(eps)
            cross_entropy = tf.reduce_sum(cross_entropy_per_var,-1)
            

            kl_loss_joint = cross_entropy + vae_neg_entropy 
            kl_loss_disjoint = vae_neg_entropy - vae_logp
            
            # kl_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
            #                   lambda: kl_loss_joint,lambda:kl_loss_disjoint)
            
            kl_loss = kl_loss_disjoint
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

        #Diffusion training

        with tf.GradientTape() as tape:
            all_neg_log_p,coeff,_ = self.get_diffusion_entropy(z,cond,batch,eps=eps,weighted=False)
            score_loss = tf.reduce_sum(all_neg_log_p, axis=-1)
            score_loss = tf.reduce_mean(score_loss)
            #Only start score training after the warmup
            score_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
                                 lambda: score_loss,lambda:0*score_loss)
            
        #+[self.mixing_logit]
        grads = tape.gradient(score_loss, self.score.trainable_weights+[self.mixing_logit])
        grads = [tf.clip_by_norm(grad, 1)
                 for grad in grads]
        #+[self.mixing_logit]
        self.sgm_optimizer.apply_gradients(zip(grads, self.score.trainable_weights+[self.mixing_logit]))
        self.score_loss_tracker.update_state(score_loss)
        
        return {
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "sgm_loss":cross_entropy, 
            "beta":beta,            
            "mixing":tf.reduce_max(coeff)
        }

    def test_step(self, inputs):
        eps=1e-5
        data,cond = inputs
        batch = tf.shape(data)[0]
        
        if len(self.data_shape) == 2:
            axis=(1,2)
        else:
            axis=(1,2,3,4)

        

        z_mean, z_log_sig, z = self.encoder([data,cond])
                        
        vae_neg_entropy = distributions.MultivariateNormalDiag(
            loc=z_mean,allow_nan_stats=False,
            scale_diag=tf.exp(z_log_sig)).log_prob(z)

        vae_logp = distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(z_mean),allow_nan_stats=False,
            scale_diag=tf.ones_like(z_mean)).log_prob(z)
        
        #VAE reconstruction
        rec,log_std= tf.split(self.decoder([z,cond]),num_or_size_splits=2, axis=-1)
        reconstruction_loss = self.reconstruction_loss(data,rec,log_std,axis=axis)
            
        cross_entropy_per_var,coeff,random_t = self.get_diffusion_entropy(z,cond,batch,eps=eps)
        #add constant entropy term

        cross_entropy_per_var += self.cross_entropy_const(eps)
        cross_entropy = tf.reduce_sum(cross_entropy_per_var,-1)
            

        kl_loss_joint = cross_entropy + vae_neg_entropy 
        kl_loss_disjoint = vae_neg_entropy - vae_logp
            
        # kl_loss = tf.cond(self.warm_up_steps < self.sgm_optimizer.iterations,
        #                   lambda: kl_loss_joint,lambda:kl_loss_disjoint)

            
        kl_loss = tf.reduce_mean(kl_loss_disjoint)

            
        #Simple linear scaling
        beta = tf.math.minimum(1.0,tf.cast(self.vae_optimizer.iterations,tf.float32)/self.kl_steps)
        total_loss = beta*kl_loss + reconstruction_loss

            

        return {
            "rec_loss": reconstruction_loss,
            "kl_loss": beta*kl_loss,
            "sgm_loss":cross_entropy,
            "loss":total_loss,
        }

    
    def generate(self,nevts,cond, sample_enconder=False,data=None,batch_size=1000):
        print(f"Model latent dims at gen: {self.latent_dim}")
        if sample_enconder:
            print("Sampling directly from encoder Z")
            data = data.batch(batch_size)
            random_latent_vectors = None
            for batch in data:
                batch_, cond_ = batch
                _, _, random_latent_vectors = self.encoder([batch_,cond_], training=False)
        else:
            random_latent_vectors = tf.random.normal(
                shape=(nevts, self.latent_dim)
            )
    
        # random_latent_vectors =self.PCSampler(cond,num_steps=self.num_steps,snr=self.snr)
        #random_latent_vectors =self.ODESampler(cond,atol=1e-5)
        
        mean,log_std= tf.split(self.decoder([random_latent_vectors,cond], training=False),num_or_size_splits=2, axis=-1)
                            
        # print(tf.exp(std))
        # input()
        return mean


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
                  # num_steps=900, 
                  # snr=0.165,
                  #num_steps=2000,
                  num_steps=200, 
                  #snr=0.23,
                  snr=0.3,
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
        batch_size = cond.shape[0]
        t = tf.ones((batch_size,1))
        data_shape = [batch_size,self.latent_dim]
        
        cond = tf.convert_to_tensor(cond, dtype=tf.float32)
        cond = tf.reshape(cond,(-1,self.num_cond))
        init_x = self.prior_sde(data_shape)
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]        
        x = init_x

        start = time.time()
        for istep,time_step in enumerate(time_steps):      
            batch_time_step = tf.ones((batch_size,1)) * time_step
            z = tf.random.normal(x.shape)
            mean,var = self.marginal_prob(batch_time_step)
            
            score = self.score([x, batch_time_step,cond])
            mixing_component = x*tf.sqrt(var)
            coeff = tf.math.sigmoid(self.mixing_logit)
            params = (1 - coeff) * mixing_component + coeff * score
            score=-params/tf.sqrt(var)

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

