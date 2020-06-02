import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import keras.backend as K

# n_out = (1 + 2*M) * KMIX
n_batch = 64
n_epochs = 200
lr = 1e-4

sigma_min = 1e-3
sigma_max = 5.0
epsilon = 1e-6

class CMDN:
    def __init__(self, img_size=(90, 320, 3), M=4, KMIX=3):
        self.img_size = img_size
        self.M = M
        self.KMIX = KMIX
        self.n_out = (1 + 2*M) * KMIX

    def build_model(self):
        inputs = Input(shape=self.img_size)
        conv1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(inputs)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(128, (4, 4), strides=(2, 2), activation='relu')(conv2)
        conv4 = Conv2D(256, (4, 4), strides=(2, 2), activation='relu')(conv3)
        flat = Flatten()(conv4)
        hidden1 = Dense(128, activation='relu')(flat)
        outputs = Dense(self.n_out, activation='linear')(hidden1)
        self.CMDN_model = Model(inputs=inputs, outputs=outputs)
        self.CMDN_model.summary()
        self.CMDN_model.compile(optimizer=Adam(lr=lr),
                                loss=self.CMDN_loss,
                                metrics=[self.metric_0]) #, metric_1])

    def train_model(self, x_train, y_train, x_val, y_val):
        self.dname = os.path.join('./trained_model', datetime.datetime.now().strftime('%Y%m%d_%H%M'))
        os.makedirs(self.dname)

        log_dir = os.path.join(self.dname, 'logs')
        os.makedirs(log_dir)
        self.training_history = self.CMDN_model.fit(x=x_train,
                                                    y=y_train,
                                                    epochs=n_epochs,
                                                    batch_size=n_batch,
                                                    validation_data=(x_val, y_val),
                                                    verbose=2,
                                                    callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=1)])

    def report_training_results(self):
        ### Plot training loss and validation loss to epochs
        plt.rcParams['figure.figsize'] = [18, 12]
        self.fig_train_hist, self.ax_train_hist = plt.figure(), plt.axes()

        loss_train = self.training_history.history['loss']
        loss_val = self.training_history.history['val_loss']

        epochs = np.arange(len(loss_train))

        self.ax_train_hist.plot(epochs, loss_train, 'b', label="Training")
        self.ax_train_hist.plot(epochs, loss_val, 'r', label="Validation")

        self.ax_train_hist.set_title("Training and validation loss")
        self.ax_train_hist.set_xlabel("Epochs")
        self.ax_train_hist.set_ylabel("Loss")
        self.ax_train_hist.legend()
        self.ax_train_hist.grid(True)

        plt.show()

    def save_training_results(self):
        ### Save the model
        path_save_model = os.path.join(self.dname, 'model.h5')
        self.CMDN_model.save(path_save_model)

        ### Save the related variables
        path_save_vars = os.path.join(self.dname, 'vars.pickle')
        self.save_vars(path_save_vars, mode='wb', M=self.M, KMIX=self.KMIX, n_out=self.n_out, lr=lr,)

        ### Save the training history
        path_save_hist = os.path.join(self.dname, 'training_hist.pickle')
        with open(path_save_hist, 'wb') as training_hist:
            pickle.dump(self.training_history.history, training_hist)    ###
    
        # ### Save the figure of training history
        # path_save_hist_fig = os.path.join(self.dname, 'training_hist.png')
        # self.fig_train_hist.savefig(path_save_hist_fig)

    def save_vars(self, fname, mode='wb', **vars):
        result = {}
        for key in vars.keys():
            exec('result[key]=vars.get(key)')
        with open(fname, mode) as f:
            pickle.dump(result, f)

        return result

    def load_model(self, model_path):
        self.CMDN_model = load_model(os.path.join(model_path, "model.h5"),
                                     custom_objects={'CMDN_loss':self.CMDN_loss,
                                                     'metric_0':self.metric_0,
                                                     'metric_1':self.metric_1,})
    
    def get_estimation(self, x): # Input should be already preprocessed
        y_pred = self.CMDN_model.predict(x)
        pi, mu, sigma = self.get_GMM_params(y_pred)
        # pi: (N, K)
        # mu: (N, K, M)
        # sigma: (N, K, M)

        total_expectation = self.get_total_expectation(pi, mu) # (N, M)
        
        aleatoric_uncertainty = self.get_aleatoric_uncertainty(pi, sigma)
        epistemic_uncertainty = self.get_epistemic_uncertainty(pi, mu, total_expectation)
        uncertainty = aleatoric_uncertainty + epistemic_uncertainty # (N, M, M)
        
        # y_L = total_expectation[:, 0] # (N,)
        # eps_L = total_expectation[:, 1] # (N,)
        
        # unc_y_L = uncertainty[:, 0, 0] # (N,)
        # unc_eps_L = uncertainty[:, 1, 1] # (N,)

        return total_expectation, uncertainty # (N, M), (N, M, M)

    def CMDN_loss(self, y_true, y_pred):
        pi, mu, sigma = self.get_GMM_params(y_pred)
        loss = self.get_loss(y_true, pi, mu, sigma)
        
        return loss

    def get_GMM_params(self, y_pred): # y_pred: (N, (1+2*M)*K)
        pi_hat = y_pred[:, 0:self.KMIX] # (N, K)
        mu_hat = tf.reshape(y_pred[:, self.KMIX:self.KMIX*(1+self.M)], [-1, self.KMIX, self.M]) # (N, K, M)
        sigma_hat = tf.reshape(y_pred[:, self.KMIX*(1+self.M):], [-1, self.KMIX, self.M]) # (N, K, M)

        pi_hat_max = tf.reduce_max(pi_hat, axis=1, keepdims=True) # (N, 1)
        exp_pi_hat = tf.exp(pi_hat - pi_hat_max) # (N, K)
        sum_exp_pi_hat = tf.reduce_sum(exp_pi_hat, axis=1, keepdims=True) # (N, 1)
        pi = exp_pi_hat / sum_exp_pi_hat # (N, K)

        mu = mu_hat # (N, K, M)

        sigmoid_sigma_hat = tf.sigmoid(sigma_hat) # (N, K, M)
        sigma = sigmoid_sigma_hat * sigma_max # (N, K, M)
        sigma = tf.clip_by_value(sigma, clip_value_min=sigma_min, clip_value_max=sigma_max) # (N, K, M)

        return pi, mu, sigma

    def get_loss(self, y_true, pi, mu, sigma):
        # y_true: (N, M)
        # pi: (N, K)
        # mu: (N, K, M)
        # sigma: (N, K, M)
        y_true = tf.stack([y_true]*self.KMIX, axis=1) # (N, K, M)
        probs = self.pdf(y_true, mu, sigma) # (N, K)
        weighted_probs = pi * probs # (N, K)
        likelihood = tf.reduce_sum(weighted_probs, axis=1) # (N,)
        loglikelihood = tf.math.log(likelihood + epsilon) # (N,)
        loss = -tf.reduce_mean(loglikelihood, axis=0)
        
        return loss

    def pdf(self, x, mu, sigma): # p(x) ~ N(mu, sigma)
        scale_diag = tf.pow(sigma, 0.5) # standard deviation
        pdf = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=scale_diag)
        probability = pdf.prob(x)

        return probability

    def metric_0(self, y_true, y_pred):
        # y_true: (N, M)
        # y_pred: (N, M)
        pi, mu, _ = self.get_GMM_params(y_pred)
        total_expectation = self.get_total_expectation(pi, mu) # (N, M)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - total_expectation), axis=0)) # (M,)
        rmse_y_L = rmse[0]

        return rmse_y_L

    def metric_1(self, y_true, y_pred):
        # y_true: (N, M)
        # y_pred: (N, M)
        pi, mu, _ = self.get_GMM_params(y_pred)
        total_expectation = self.get_total_expectation(pi, mu) # (N, M)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - total_expectation), axis=0)) # (M,)
        rmse_eps_L = rmse[1]

        return rmse_eps_L

    def get_total_expectation(self, pi, mu):
        # pi: (N, K)
        # mu: (N, K, M)
        pi = tf.reshape(pi, (-1, self.KMIX, 1)) # (N, K, 1)
        total_expectation = tf.reduce_sum(pi * mu, axis=1) # (N, M)

        return total_expectation # (N, M)

    def get_aleatoric_uncertainty(self, pi, sigma):
        # pi: (N, K)
        # sigma: (N, K, M)
        sigma = tf.linalg.diag(sigma) # (N, K, M, M)
        pi = tf.reshape(pi, [-1, self.KMIX, 1, 1]) # (N, K, 1, 1)
        weighted_sigma = pi * sigma # (N, K, M, M)
        aleatoric_uncertainty = tf.reduce_sum(weighted_sigma, axis=1) # (N, M, M)

        return aleatoric_uncertainty

    def get_epistemic_uncertainty(self, pi, mu, total_expectation):
        # pi: (N, K)
        # mu: (N, K, M)
        # total_expectation: (N, M)
        deviation = mu - tf.stack([total_expectation]*self.KMIX, axis=1) # (N, K, M)
        deviation = tf.reshape(deviation, [-1, self.KMIX, self.M, 1]) # (N, K, M, 1)
        deviation_transpose = tf.reshape(deviation, [-1, self.KMIX, 1, self.M]) # (N, K, 1, M)
        covariance = tf.linalg.matmul(deviation, deviation_transpose) # (N, K, M, M)
        pi = tf.reshape(pi, [-1, self.KMIX, 1, 1]) # (N, K, 1, 1)
        weighted_covariance = pi * covariance # (N, K, M, M)
        epistemic_uncertainty = tf.reduce_sum(weighted_covariance, axis=1) # (N, M, M)

        return epistemic_uncertainty