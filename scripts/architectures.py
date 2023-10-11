import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.layers import TimeDistributed


class EpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch = epoch

def Encoder(
        input_dim,
        time_embedding,
        input_embedding_dims,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        pad=((2,1),(0,0),(4,3)),
        use_1D=False,
        #pad=((1,0),(0,0),(1,0)),
):

    act = tf.keras.activations.swish
    def ResidualBlock(width, attention):
        def forward(x):
            x , n = x
            input_width = x.shape[2] if use_1D else x.shape[4]
            if input_width == width:
                residual = x
            else:
                if use_1D:
                    residual = layers.Conv1D(width, kernel_size=1)(x)
                else:
                    residual = TimeDistributed(layers.Conv2D(width, kernel_size=1))(x)

            n = layers.Dense(width)(n)
            #x = tfa.layers.GroupNormalization(groups=4)(x)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = TimeDistributed(layers.Conv2D(width, kernel_size=kernel, padding="same"))(x)
            x = layers.Add()([x, n])
            # x = tfa.layers.GroupNormalization(groups=4)(x)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = TimeDistributed(layers.Conv2D(width, kernel_size=kernel, padding="same"))(x)
            x = layers.Add()([residual, x])

            if attention:
                residual = x
                if use_1D:                    
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1)
                    )(x, x)
                else:
                    x = tfa.layers.GroupNormalization(groups=4, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1, 2, 3)
                    )(x, x)

                x = layers.Add()([residual, x])
            return x
        return forward
    
    def DownBlock(block_depth, width, attention):
        def forward(x):
            x, n = x
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x,n])

            if use_1D:
                x = layers.AveragePooling1D(pool_size=stride)(x)
            else:
                x = TimeDistributed(layers.AveragePooling2D(pool_size=stride))(x)
            return x

        return forward


    inputs = keras.Input((input_dim))
        
    if use_1D:
        #No padding to 1D model
        x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs)
        n = layers.Reshape((1,time_embedding.shape[-1]))(time_embedding)
    else:
        inputs_padded = layers.ZeroPadding3D(pad)(inputs)
        x = TimeDistributed(layers.Conv2D(input_embedding_dims, kernel_size=1))(inputs_padded)
        n = layers.Reshape((1,1,1,time_embedding.shape[-1]))(time_embedding)
    
    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, n])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])([x,n])
        
    return inputs, x





def Decoder(
        inputs,
        time_embedding,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        pad=((2,1),(0,0),(4,3)),
        use_1D=False,
        #pad=((1,0),(0,0),(1,0)),
):

    act = tf.keras.activations.swish
    def ResidualBlock(width, attention):
        def forward(x):
            x , n = x
            input_width = x.shape[2] if use_1D else x.shape[4]
            if input_width == width:
                residual = x
            else:
                if use_1D:
                    residual = layers.Conv1D(width, kernel_size=1)(x)
                else:
                    residual = TimeDistributed(layers.Conv2D(width, kernel_size=1))(x)

            n = layers.Dense(width)(n)
            # x = tfa.layers.GroupNormalization(groups=4)(x)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = TimeDistributed(layers.Conv2D(width, kernel_size=kernel, padding="same"))(x)
            x = layers.Add()([x, n])
            # x = tfa.layers.GroupNormalization(groups=4)(x)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = TimeDistributed(layers.Conv2D(width, kernel_size=kernel, padding="same"))(x)
            x = layers.Add()([residual, x])

            if attention:
                residual = x
                if use_1D:                    
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1)
                    )(x, x)
                else:
                    # x = tfa.layers.GroupNormalization(groups=4, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1, 2, 3)
                    )(x, x)

                x = layers.Add()([residual, x])
            return x
        return forward
    
    def UpBlock(block_depth, width, attention):
        def forward(x):
            x, n = x
            if use_1D:
                x = layers.UpSampling1D(size=stride)(x)
            else:
                x = TimeDistributed(layers.UpSampling2D(size=stride))(x)
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x,n])
            return x

        return forward

    x = inputs

    if use_1D:
        n = layers.Reshape((1,time_embedding.shape[-1]))(time_embedding)
    else:
        n = layers.Reshape((1,1,1,time_embedding.shape[-1]))(time_embedding)

    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, n])
        
    outputs = layers.Cropping3D(pad)(x)
        
    return outputs




def Resnet(
        input_dim,
        time_embedding,
        num_layer = 3,
        mlp_dim=128,
):

    
    act = layers.LeakyReLU(alpha=0.01)
    #act = tf.keras.activations.swish

    def resnet_dense(input_layer,hidden_size):
        layer,time = input_layer
        residual = layers.Dense(hidden_size)(layer)
        embed =  layers.Dense(hidden_size)(time)
        x = act(layer)
        x = layers.Dense(hidden_size)(x)
        x = act(layers.Add()([x, embed]))
        x = layers.Dense(hidden_size)(x)
        x = layers.Add()([x, residual])
        return x

    inputs = Input((input_dim))
    embed = act(layers.Dense(mlp_dim)(time_embedding))
    
    layer = layers.Dense(mlp_dim)(inputs)
    for _ in range(num_layer-1):
        layer =  resnet_dense([layer,embed],mlp_dim)

    outputs = layers.Dense(input_dim)(layer)
    
    return inputs,outputs

def Discriminator(
        input_dim,
        num_layers = 3,
        mlp_width = 256
        ):
    
    act = layers.LeakyReLU(alpha=0.01)
    inputs = Input((input_dim))
    inputs_ = tf.reshape(inputs, shape=(tf.shape(inputs)[0], 45, 16 * 9 * 1))
    x = layers.Dense(mlp_width)(inputs_)
    
    for _ in range(num_layers-1):
        x = act(layers.Dense(mlp_width)(x))
        
    outputs = layers.Dense(1,activation='sigmoid')(x)
    return inputs, outputs

def PatchDiscriminator(
   data_format = "channels_first",
   pad = [[0,0], [2,1]], 
   initializer = tf.random_normal_initializer(0., 0.02)    
):

  def conv2d(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False, data_format=data_format))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

  inp = tf.keras.layers.Input(shape=[45, 16, 9], name='input_image')
  conv1 = conv2d(64, 3, False)(inp)
  conv2 = conv2d(128, 3)(conv1)

  zero_pad1 = tf.keras.layers.ZeroPadding2D(data_format=data_format)(conv2)  
  conv3 = tf.keras.layers.Conv2D(512, 3, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False, data_format=data_format)(zero_pad1) 

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv3)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D(data_format=data_format)(leaky_relu) 

  last = tf.keras.layers.Conv2D(45, 3, strides=1,
                                kernel_initializer=initializer, data_format=data_format, activation="sigmoid")(zero_pad2)

  return inp, last
