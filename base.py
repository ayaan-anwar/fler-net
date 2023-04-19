import numpy as np
import tensorflow as tf

from hyperparameters import HyperParameters
from model import create_keras_model
from sklearn.preprocessing import MinMaxScaler

class BaseModel:
    def __init__(self, label: str, path_to_features: str) -> None:
        self.label = label
        self.subjects = list(range(1, 33))
        self.path_to_features = path_to_features

    def load_preprocess_data(self):
        X_train = np.concatenate([
            np.load(self.path_to_features + "/S%02d_X_train.npy" % sub, allow_pickle=True)
            for sub in self.subjects], axis=0)
        Y_train = np.concatenate([
            np.load(self.path_to_features + f"/S%02d_Y_train_{self.label}.npy" % sub, allow_pickle=True)
            for sub in self.subjects], axis=0)
        X_test = np.concatenate([
            np.load(self.path_to_features + "/S%02d_X_test.npy" % sub, allow_pickle=True)
            for sub in self.subjects], axis=0)
        Y_test = np.concatenate([
            np.load(self.path_to_features + f"/S%02d_Y_test_{self.label}.npy" % sub, allow_pickle=True)
            for sub in self.subjects], axis=0)

        Y_train = np.reshape(Y_train, (-1, 1))
        Y_test = np.reshape(Y_test, (-1, 1))

        X_train = np.asarray(X_train).astype('float32')
        X_test = np.asarray(X_test).astype('float32')
        Y_train = np.asarray(Y_train).astype('float32')
        Y_test = np.asarray(Y_test).astype('float32')
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(X_train)
        X_train_new = scaler.transform(X_train)
        X_test_new = scaler.transform(X_test)

        return X_train_new, Y_train, X_test_new, Y_test
    
    def train(self):
        X_train, Y_train, X_test, Y_test = self.load_preprocess_data()
        model = create_keras_model(X_train.shape[1:])
        hp = HyperParameters()
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.SGD(learning_rate=hp.base_loss))
        history = model.fit(x=X_train, y=Y_train, epochs=hp.epochs, batch_size=hp.batch_size, validation_data=(X_test, Y_test))
        return model, history

    
    
