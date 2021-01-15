import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.ops.losses.losses_impl import absolute_difference, Reduction
import tensorflow.keras.backend as K

import numpy as np


def kernel_init(shape, dtype=None):
    srm = np.loadtxt('../srm/rich_model.txt')
    srmkernel = np.float32(srm)
    srmkernel = np.reshape(srmkernel, [30, 1, 5, 5])
    srmkernel = np.transpose(srmkernel, (2, 3, 1, 0))

    kernels = np.zeros([5, 5, 3, 30])
    for k in range(30):
        kernels[:, :, 0, k] = srmkernel[:, :, 0, k]
        kernels[:, :, 1, k] = srmkernel[:, :, 0, k]
        kernels[:, :, 2, k] = srmkernel[:, :, 0, k]

    assert kernels.shape == shape

    return K.variable(kernels, dtype='float32')

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def encoder():
    latent_dim = 512
    encoder_inputs = Input(shape=(32, 32, 90))

    x = Conv2D(128, 5, activation='relu', strides=2, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)

    x = Conv2D(256, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    z_mean = Dropout(0.25)(Dense(latent_dim, activation='relu', name="z_mean")(x))
    z_log_var = Dropout(0.25)(Dense(latent_dim, activation='relu', name="z_log_var")(x))

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def decoder():
    latent_inputs = keras.Input(shape=(512,))
    x = Dropout(0.25)(Dense(8 * 8 * 512, activation='relu')(latent_inputs))
    x = Reshape((8, 8, 512))(x)

    x = Conv2DTranspose(512, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(256, 3, strides=2, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, 3, strides=2, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    decoder_outputs = Conv2DTranspose(90, 3, activation='sigmoid', padding="same")(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def dicriminative_error(error, mask):
    mask1 = 1 - mask
    error1 = tf.math.multiply(error, mask1)
    N1 = tf.reduce_sum(mask1, axis=[1, 2])
    mean = tf.math.divide(tf.reduce_sum(error1, axis=[1, 2]), N1)
    return mean


class vae(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(vae, self).__init__(**kwargs)
        self.srmConv = Conv2D(30, kernel_size=[5, 5], kernel_initializer=kernel_init,
                              strides=1, padding='same', trainable=False)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, **kwargs):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return inputs, reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            mask = data[1]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            L1 = absolute_difference(data, reconstruction, reduction=Reduction.NONE)
            print(L1.shape)
            error = tf.reduce_mean(L1, axis=3)
            if error:
                print("ok")

            reconstruction_loss = dicriminative_error(error, mask)

            reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            kl_loss = -0.5*tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            mask = data[1]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        L1 = absolute_difference(data, reconstruction, reduction=Reduction.NONE)
        error = tf.reduce_mean(L1, axis=3)

        reconstruction_loss = dicriminative_error(error, mask)

        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def train(name_model, dataPath, maskPath):
    data = np.load(dataPath)
    mask = np.load(maskPath)

    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, random_state=42)

    model = vae(encoder(), decoder())
    model.compile(optimizer=Adam(lr=1e-6))

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../models/{}".format(name_model),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
    csv_logger = CSVLogger("{}.csv".format(name_model), append=True)

    callbacks_list = [checkpoint, csv_logger]

    model.fit(train_data, train_mask, epochs=250, batch_size=128,
              validation_data=(test_data, test_mask),
              callbacks=callbacks_list)


if __name__ == '__main__':
    print('yo')