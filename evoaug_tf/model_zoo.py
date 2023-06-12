import tensoflow as tf
from tensorflow import keras


def DeepSTARR(input_shape):

    # body
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(256, kernel_size=7, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(60, kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(60, kernel_size=5, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv1D(120, kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    #x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(2, activation='linear')(x)

    return inputs, outputs
