import tensorflow as tf

from hyperparameters import HyperParameters

def create_keras_model(input_shape: tuple) -> tf.keras.models.Sequential:
    hp = HyperParameters()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hp.layer1, activation=hp.activation, input_shape=input_shape),
        tf.keras.layers.Dropout(hp.dropout1),
        tf.keras.layers.Dense(hp.layer2, activation=hp.activation),
        tf.keras.layers.Dropout(hp.dropout2),
        tf.keras.layers.Dense(hp.layer3, activation=hp.activation),
        tf.keras.layers.Dropout(hp.dropout3),
        tf.keras.layers.Dense(hp.output_layer, activation=hp.output_activation)
    ])
    return model

