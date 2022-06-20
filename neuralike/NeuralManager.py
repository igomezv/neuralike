"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Jun 2022
"""
from .NeuralNet import NeuralNet
from .RandomSampling import RandomSampling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import os


class NeuralManager:
    """
        Manager for neural networks to learn likelihood function over a grid
        Parameters
        -----------
        loglikelihood
        pars_bounds
        rootname
        ndivsgrid
        hidden_layers_neurons
        epochs
        plot
    """
    def __init__(self, loglikelihood, samples, likes, rootname='neural',
                 neuralike_settings=None):
        self.loglikelihood_fn = loglikelihood
        self.valid = False
        self.fig_path = '{}.png'.format(rootname)
        self.samples = samples
        self.likes = likes
        if neuralike_settings:
            self.learning_rate = neuralike_settings['learning_rate']
            self.batch_size = neuralike_settings['batch_size']
            self.epochs = neuralike_settings['epochs']
            self.patience = neuralike_settings['patience']
            self.psplit = neuralike_settings['psplit']
            self.hidden_layers_neurons = neuralike_settings['hidden_layers_neurons']
            self.plot = neuralike_settings['plot']
            self.valid_loss = neuralike_settings['valid_loss']
            self.nrand = neuralike_settings['nrand']
            # self.rmse_criterion = neuralike_settings['rmse_criterion']
        else:
            self.learning_rate = 5e-4
            self.batch_size = 32
            self.epochs = 100
            self.patience = self.epochs//2
            self.psplit = 0.8
            self.hidden_layers_neurons = [100, 100, 100]
            self.plot = True
            self.valid_loss = 0.5
            self.nrand = 5

        _, self.dims = np.shape(self.samples)
        self.topology = [self.dims] + self.hidden_layers_neurons + [1]
        self.mse_criterion = self.valid_loss


        # self.model_path = '{}.h5'.format(rootname)
        # if not self.modelChecker():
        #     self.training()
        # else:
        #     self.neural_model = self.load()

    def training(self, map_fn=map):
        ml_idx = np.argmax(self.likes)
        means = self.samples[ml_idx, :]
        print("\nGenerating training set...")
        rsampling = RandomSampling(self.loglikelihood_fn, samples=self.samples, means=means, mins=np.max(self.samples, axis=0),
                                   maxs=np.max(self.samples, axis=0),
                                   nrand=self.nrand)
        rsamples, rlikes = rsampling.make_dataset(map_fn=map_fn)
        print("Training dataset created!")
        likes = np.append(rlikes, self.likes)
        samples = np.append(rsamples, self.samples, axis=0)

        ## scale params
        self.samples_scaler = MinMaxScaler(feature_range=(0.1, 1))
        # self.samples_scaler = StandardScaler(with_mean=False)
        self.samples_scaler.fit(samples)
        sc_samples = self.samples_scaler.transform(samples)
        # print(sc_samples)
        # # create scaler
        # self.likes_scaler = StandardScaler(with_mean=False)
        self.likes_scaler = MinMaxScaler(feature_range=(0.1, 1))
        self.likes_scaler.fit(likes.reshape(-1, 1))
        # # # apply transforms
        sc_likes = self.likes_scaler.transform(likes.reshape(-1, 1))

        self.neural_model = NeuralNet(X=sc_samples, Y=sc_likes, topology=self.topology,
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      learrning_rate=self.learning_rate,
                                      patience=self.patience,
                                      minsample=np.min(np.abs(samples)))

        self.neural_model.train()
        # neural_model.save_model('{}'.format(self.model_path))
        if self.plot:
            self.neural_model.plot(save=True, figname='{}'.format(self.fig_path), show=False)
        lastval = self.neural_model.loss_val[-1]
        lasttrain = self.neural_model.loss_train[-1]
        ## preparing for testing
        test_mse = self.neural_model.test_mse()
        print("Test MSE: {:.4f} | Criterion: {:.4f}".format(test_mse, self.mse_criterion))
        if test_mse < self.mse_criterion:
            self.valid = True
            print("\nValid Neural net | Train loss: {:.4f} | "
                  "Val loss: {:.4f}\n".format(lasttrain, lastval))
        else:
            self.valid = False
            print("\nNOT valid neural net | Train loss:{:.4f} | Val loss: {:.4f}\n".format(lasttrain, lastval))

    def load(self):
        neural_model = NeuralNet(load=True, model_path=self.model_path)
        return neural_model

    def modelChecker(self):
        """
        This method checks if the name of the model.h5 exists, if it already exists use it, otherwise train a
        new neural net in order to generate a new model.
        """
        if os.path.isfile('{}'.format(self.model_path)):
            return True
        else:
            return False

    def neuralike(self, params):
        # loglikelihood only can work if trainning was executed
        sc_params = self.samples_scaler.transform(np.array(params).reshape(1, self.dims))
        # pred = self.neural_model.predict(sc_params)
        pred = self.neural_model.predict(sc_params)
        likes = self.likes_scaler.inverse_transform(pred.reshape(-1, 1))
        likes = np.array(likes)
        return likes