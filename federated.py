from collections import OrderedDict
import numpy as np
import random
import tensorflow as tf
import tensorflow_federated as tff

import constants

from hyperparameters import HyperParameters
from model import create_keras_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

class FederatedModel:
    def __init__(self, label: str, num_clients: int, path_to_features: str) -> None:
        self.label = label
        self.subjects = list(range(1, 33))
        random.Random(constants.SEED1).shuffle(self.subjects)
        self.num_clients = num_clients
        self.path_to_features = path_to_features

    def load_preprocess_data(self):
        self.client_ids = list(range(self.num_clients))
        client = []
        subjects_per_client = len(self.subjects) // self.num_clients
        for i in range(self.num_clients):
            client.append(self.subjects[i * subjects_per_client : (i + 1) * subjects_per_client])
        
        self.clientwise_data = dict()
        self.clientwise_target = dict()

        for C in range(self.num_clients):
            self.clientwise_data[C] = np.concatenate([
                                    np.load(self.path_to_features + "/S%02d_X_train.npy" % sub, allow_pickle=True)
                                    for sub in client[C]], axis=0)
            self.clientwise_target[C] = np.concatenate([
                                    np.load(self.path_to_features + f"/S%02d_Y_train_{self.label}.npy" % sub, allow_pickle=True)
                                    for sub in client[C]], axis=0)
            # convert 0-rank array
            self.clientwise_target[C] = np.reshape(self.clientwise_target[C], (-1,1))
            # convert to float-32
            self.clientwise_data[C] = np.asarray(self.clientwise_data[C]).astype('float32')
            self.clientwise_target[C] = np.asarray(self.clientwise_target[C]).astype('float32')

        self.X_test = np.concatenate([
                        np.load(self.path_to_features + "/S%02d_X_test.npy" % sub, allow_pickle=True)
                        for sub in self.subjects], axis=0)
        print("X_test: ", self.X_test.shape)

        self.Y_test = np.concatenate([
                        np.load(self.path_to_features + f"/S%02d_Y_test_{self.label}.npy" % sub, allow_pickle=True)
                        for sub in self.subjects], axis=0)
        print("Y_test: ", self.Y_test.shape)
        # convert 0-rank array
        self.Y_test = np.reshape(self.Y_test, (-1, 1))
        # convert to float-32
        self.X_test = np.asarray(self.X_test).astype('float32')
        self.Y_test = np.asarray(self.Y_test).astype('float32')

        self.input_shape = self.X_test.shape[1:]


    def train(self):
        hp = HyperParameters()
        self.load_preprocess_data()

        federated_train_data = self.make_federated_data(self.client_ids)

        self.example_data = self.example_preprocess(0)
        self.example_data = tf.data.Dataset.from_tensor_slices(self.example_data)
        self.example_data = self.example_data.repeat(hp.epochs_per_round).shuffle(hp.shuffle).batch(hp.batch_size)

        def model_fn():
            keras_model = create_keras_model(self.input_shape)
            return tff.learning.models.from_keras_model(
                keras_model,
                input_spec=self.example_data.element_spec,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])


        iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=hp.fed_loss_client),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=hp.fed_loss_server))
        print(iterative_process.initialize.type_signature.formatted_representation())

        gpu_devices = tf.config.list_logical_devices('GPU')
        print(gpu_devices)
        # tff.backends.native.set_local_python_execution_context(
        #         client_tf_devices=[gpu_devices[0]], clients_per_thread=4)
        #tff.backends.native.set_sync_local_cpp_execution_context(
        #        max_concurrent_computation_calls=4)
        NUM_ROUNDS = hp.fed_rounds
        # The state of the FL server, containing the model and optimization state.
        state = iterative_process.initialize()
        # Global model
        keras_model = create_keras_model(self.input_shape)
        # Load our pre-trained Keras model weights into the global model state.
        trained_weights = tff.learning.models.ModelWeights(
            trainable=[v.numpy() for v in keras_model.trainable_weights],
            non_trainable=[
                v.numpy() for v in keras_model.non_trainable_weights
            ])
        state = iterative_process.set_model_weights(state, trained_weights)
        def keras_evaluate(state, round_num):
            keras_model = create_keras_model(self.input_shape)
            keras_model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])
            model_weights = iterative_process.get_model_weights(state)
            model_weights.assign_weights_to(keras_model)
            loss, accuracy = keras_model.evaluate(x=self.X_test, y=self.Y_test, verbose=0)
            print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
            if round_num % 50 == 0:
                Y_pred = keras_model.predict(self.X_test, verbose=0)
                Y_pred = [round(x[0]) for x in Y_pred]
                print(confusion_matrix(self.Y_test, Y_pred))
            return loss, accuracy

        results = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for round_num in range(NUM_ROUNDS):
            print('Round {r}'.format(r=round_num))
            val_loss, val_accuracy = keras_evaluate(state, round_num)
            result = iterative_process.next(state, federated_train_data)
            state = result.state
            train_metrics = result.metrics['client_work']['train']
            print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
                l=train_metrics['loss'], a=train_metrics['binary_accuracy']))
            results['loss'].append(train_metrics['loss'])
            results['accuracy'].append(train_metrics['binary_accuracy'])
            results['val_loss'].append(val_loss)
            results['val_accuracy'].append(val_accuracy)

        print('Final evaluation')
        keras_evaluate(state, NUM_ROUNDS + 1)
        return results

    def example_preprocess(self, client_id):
        return OrderedDict(
            x = self.clientwise_data[client_id],
            y = self.clientwise_target[client_id]
        )

    def preprocess(self, client_id):
        return OrderedDict(
            x = self.clientwise_data[client_id],
            y = self.clientwise_target[client_id]
        )

    def make_federated_data(self, client_ids):
        hp = HyperParameters()
        return [
            tf.data.Dataset.from_tensor_slices(self.preprocess(x)).repeat(hp.epochs_per_round).shuffle(hp.shuffle).batch(hp.batch_size)
            for x in client_ids
        ]

    
