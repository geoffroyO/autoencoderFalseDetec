import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, \
    Conv2DTranspose, Reshape, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_math_ops import squared_difference
import tensorflow.keras.backend as K

import numpy as np


def _build_SRM_kernel():
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = np.einsum('klij->ijlk', filters)
    filters = filters.flatten()
    initializer_srm = tf.constant_initializer(filters)

    return initializer_srm


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
    latent_dim = 128
    encoder_inputs = Input(shape=(32, 32, 30))

    x = Conv2D(32, 5, activation='relu', strides=2, padding="same")(encoder_inputs)
    x = BatchNormalization()(x)

    x = Conv2D(64, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, 5, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    z_mean = Dropout(0.25)(Dense(latent_dim, activation='relu', name="z_mean")(x))
    z_log_var = Dropout(0.25)(Dense(latent_dim, activation='relu', name="z_log_var")(x))

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def decoder():
    latent_inputs = keras.Input(shape=(128,))
    x = Dropout(0.25)(Dense(8 * 8 * 128, activation='relu')(latent_inputs))
    x = Reshape((8, 8, 128))(x)

    x = Conv2DTranspose(128, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)

    decoder_outputs = Conv2DTranspose(30, 3, activation='sigmoid', padding="same")(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def dicriminative_error(error, mask):
    mask1 = 1 - mask
    error1 = tf.math.multiply(error, mask1)

    N1 = tf.reduce_sum(mask1, axis=[1, 2])

    mean = tf.math.divide_no_nan(tf.reduce_sum(error1, axis=[1, 2]), N1)
    return mean


class vae(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(vae, self).__init__(**kwargs)
        self.srmConv = Conv2D(30, kernel_size=[5, 5], kernel_initializer=kernel_init,
                              strides=1, padding='same', trainable=False, activation='sigmoid')
        self.norm = BatchNormalization()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, **kwargs):
        inputsSrm = self.srmConv(inputs)

        _, _, z = self.encoder(inputsSrm)

        reconstruction = self.decoder(z)

        error = squared_difference(inputsSrm, reconstruction)
        error = tf.reduce_sum(error, axis=-1)

        return error, reconstruction, inputsSrm

    def train_step(self, data):
        if isinstance(data, tuple):
            mask = data[1]
            data = data[0]

        with tf.GradientTape() as tape:
            dataSrm = self.srmConv(data)

            z_mean, z_log_var, z = self.encoder(dataSrm)
            reconstruction = self.decoder(z)

            L2 = squared_difference(dataSrm, reconstruction)
            error = tf.reduce_mean(L2, axis=-1)

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
            mask = data[1]
            data = data[0]

        dataSrm = self.srmConv(data)

        z_mean, z_log_var, z = self.encoder(dataSrm)
        reconstruction = self.decoder(z)

        L2 = squared_difference(dataSrm, reconstruction)
        error = tf.reduce_mean(L2, axis=-1)

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
    print("... Loading data")
    data = np.load(dataPath)
    mask = np.load(maskPath)

    print("... Spliting")
    train_data, test_data, train_mask, test_mask = train_test_split(data, mask, random_state=42)

    model = vae(encoder(), decoder())
    model.compile(optimizer=Adam(lr=1e-6), run_eagerly=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint("../models/{}.hdf5".format(name_model),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min')
    csv_logger = CSVLogger("{}.csv".format(name_model), append=True)

    callbacks_list = [checkpoint, csv_logger]

    model.fit(train_data, train_mask, epochs=250, batch_size=128,
              validation_data=(test_data, test_mask),
              callbacks=callbacks_list)


if __name__ == '__main__':
    dataPath = "../data/CASIA.numpy/all_to_train.npy"
    maskPath = "../data/CASIA.numpy/all_to_train_msk.npy"

    train("vae_250", dataPath, maskPath)