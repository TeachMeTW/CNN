import os
from datetime import datetime

# optuna optimization
import optuna
import pandas as pd
# plotting/tensorboard
import seaborn as sns
# auto encoder
import tensorflow as tf
# from flask import current_app as app
import keras
from keras.layers import Input, Dense, Dropout
from optuna.samplers import TPESampler
from tensorflow.keras import regularizers
# preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from src.shared.base_logger import log
import logging
import warnings

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
warnings.filterwarnings('ignore')
sns.set(style='whitegrid', context='notebook')


class AutoEncoder(object):
    def __init__(self, data, features, model_name, log_dir, **kwargs):
        self.data = data
        self.features = features
        self.input_dim = len(self.features)
        self.model_name = model_name
        self.log_dir = log_dir
        # kwargs
        self.random_state = kwargs.get('random_state', 7)
        self.n_trials = kwargs.get('n_trials', 10)
        self.train_percentage = kwargs.get('train_size', 0.7)
        self.early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 0.0001)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 25)
        self.activation_function = kwargs.get('activation_function','silu')
        self.machine_type = kwargs.get('machine_type','mac')
        self.engineer_data()

    def engineer_data(self):
        df = self.data
        features = self.features
        train_percentage = self.train_percentage
        # preprocessing
        train_, val_ = train_test_split(df[features],
                                        test_size=1 - train_percentage,
                                        random_state=self.random_state)
        logging.info('Train shape: {}; Validation shape: {};'.format(train_.shape,val_.shape))
        self.train = train_
        self.validation = val_

    # Build auto encoder architecture
    def buildAutoEncoder(self, latent_dim, reduction_modulo, dropout_percentage, learning_rate):
        input_dim = self.input_dim
        autoencoder = tf.keras.models.Sequential()
        encoder_neuron = list(range(latent_dim, input_dim + 1))
        encoder_neuron.sort(reverse=True)
        activation_function = self.activation_function
        for n in encoder_neuron:
            if n == input_dim:
                # input layer
                input_layer = Input(shape=(n,))
                autoencoder.add(input_layer)
                autoencoder.add(Dense(n,
                                      activation=activation_function,
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                                      bias_regularizer=regularizers.L2(1e-5),
                                      activity_regularizer=regularizers.L2(1e-5)
                                      )
                                )
            elif n == latent_dim:
                # latent space
                autoencoder.add(Dropout(dropout_percentage))
                autoencoder.add(Dense(n,
                                      activation=activation_function,
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                                      bias_regularizer=regularizers.L2(1e-5),
                                      activity_regularizer=regularizers.L2(1e-5)
                                      )
                                )
            elif n % reduction_modulo == 0:
                # encoder layers
                autoencoder.add(Dropout(dropout_percentage))
                autoencoder.add(Dense(n,
                                      activation=activation_function,
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                                      bias_regularizer=regularizers.L2(1e-5),
                                      activity_regularizer=regularizers.L2(1e-5)
                                      )
                                )
            else:
                pass

        decoder_neurons = encoder_neuron[:-1]
        decoder_neurons.sort()
        for n in decoder_neurons:
            if n == input_dim:
                # output layer
                autoencoder.add(Dropout(dropout_percentage))
                autoencoder.add(Dense(n,
                                      activation='linear',
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                                      bias_regularizer=regularizers.L2(1e-5),
                                      activity_regularizer=regularizers.L2(1e-5)
                                      )
                                )
            elif n % reduction_modulo == 0:
                autoencoder.add(Dropout(dropout_percentage))
                # decoder layers
                autoencoder.add(Dense(n,
                                      activation=activation_function,
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                      kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                                      bias_regularizer=regularizers.L2(1e-5),
                                      activity_regularizer=regularizers.L2(1e-5)
                                      )
                                )
            else:
                pass
        try:
            if self.machine_type == 'mac':
                optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            else:
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

            autoencoder.compile(optimizer=optimizer,
                                loss="mse",
                                metrics=["acc"])
        except Exception as e:
            logging.info(f'Failed to Compile:\n{e}')
            raise e
        return autoencoder

    def loggingAutoEncoder(self, autoencoder, batch_size):
        yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')
        # new folder for new run
        log_subdir = f'{yyyymmddHHMM}_batch{batch_size}_layers{len(autoencoder.layers)}'
        # define early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=self.early_stopping_min_delta,
            patience=self.early_stopping_patience,
            verbose=0,
            mode='min',
            restore_best_weights=True
        )
        save_model = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_name,
            save_best_only=True,
            monitor='val_loss',
            verbose=0,
            mode='min'
        )
        tensorboard = tf.keras.callbacks.TensorBoard(
            f'{self.log_dir}/{log_subdir}',
            update_freq='batch'
        )
        # callbacks argument only takes a list
        callbacks = [early_stop, save_model, tensorboard]
        return callbacks

    def fitAutoEncoder(self, model, epochs, batch_size, callbacks):
        train = self.train.values
        validation = self.validation.values
        history = model.fit(train, train,
                            shuffle=True,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            validation_data=(validation, validation),
                            verbose=0
                            )
        return pd.DataFrame(history.history)

    def scoreAutoEncoder(self, history):
        if len(history[history['val_loss'] > history['loss']]) > 1:
            model_loss = history['loss'].max()
        else:
            model_loss = history['loss'].min()
        return float(model_loss)

    def objective(self, trial):
        max_latent_dim = int(int(self.input_dim) / 3)
        if max_latent_dim > 10:
            max_latent_dim = 10
        elif max_latent_dim < 3:
            max_latent_dim = 3
        else:
            pass

        latent_dim = trial.suggest_int('latent_dim', 2, max_latent_dim, step=1)
        reduction_modulo = trial.suggest_int('reduction_modulo', 1, int(self.input_dim/2), step=1)
        dropout_percentage = trial.suggest_float('dropout_percentage', 0.01, 0.9, step=0.01)
        epochs = trial.suggest_categorical('epochs', [1000, 1500, 2000])
        batch_size = trial.suggest_categorical('batch_size', [1000, 1500, 2000])
        learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001, 0.0001, 0.00001])
        try:
            autoencoder = self.buildAutoEncoder(latent_dim=latent_dim,
                                                reduction_modulo=reduction_modulo,
                                                dropout_percentage=dropout_percentage,
                                                learning_rate=learning_rate)
            callbacks = self.loggingAutoEncoder(autoencoder=autoencoder,
                                                batch_size=batch_size)
            history = self.fitAutoEncoder(model=autoencoder,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          callbacks=callbacks)
            model_score = self.scoreAutoEncoder(history)
        except Exception as e:
            logging.info('Architecture Failed\n{}'.format(e))
            model_score = 9**999
            pass
        return model_score

    def search(self):
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler,
                                    direction='minimize')
        try:
            study.optimize(lambda trial: self.objective(trial),
                           n_trials=self.n_trials)
        except Exception as e:
            logging.info('Architecture Failed\n{}'.format(e))
            pass
        trial = study.best_trial
        best_params = trial.params
        logging.info(f'Model Loss: {trial.value}\nOptimized Hyper-Parameters: {trial.params}')
        return study, best_params

    def get_optimized_params(self):
        self.study, self.best_params = self.search()

    def build_optimized(self):
        try:
            self.optimized_autoEncoder = self.buildAutoEncoder(latent_dim=self.best_params['latent_dim'],
                                                               reduction_modulo=self.best_params['reduction_modulo'],
                                                               dropout_percentage=self.best_params['dropout_percentage'],
                                                               learning_rate=self.best_params['learning_rate']
                                                               )
            callbacks = self.loggingAutoEncoder(autoencoder=self.optimized_autoEncoder,
                                                batch_size=self.best_params['batch_size'])
            self.history = self.fitAutoEncoder(model=self.optimized_autoEncoder,
                                               epochs=self.best_params['epochs'],
                                               batch_size=self.best_params['batch_size'],
                                               callbacks=callbacks)
            self.model_score = self.scoreAutoEncoder(self.history)
        except Exception as e:
            logging.info('Architecture Failed\n{}'.format(e))
            pass

    def fit_pipeline(self):
        self.get_optimized_params()
        self.build_optimized()
        hfp = {'autoencoder': self.optimized_autoEncoder,
               'history': self.history,
               'best_params': self.best_params,
               'model_loss': self.study.best_value}
        return hfp
