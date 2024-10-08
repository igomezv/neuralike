"""neuralike management object.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: Jun 2022
"""
import numpy as np
from simplemc.analyzers.neuralike.NeuralManager import NeuralManager

class NeuraLike:
    """
        Manager for neural networks to learn likelihood function over a grid
        Parameters
        -----------
        loglikelihood
        rootname
    """
    def __init__(self, loglikelihood_control, rootname='neural',
                 neuralike_settings=None):
        # simplemc: reading neuralike settings
        self.nstart_samples = neuralike_settings['nstart_samples']
        self.nstart_stop_criterion = neuralike_settings['nstart_stop_criterion']
        self.ncalls_excess = neuralike_settings['ncalls_excess']
        self.updInt = neuralike_settings['updInt']

        self.loglikelihood_control = loglikelihood_control
        # Counters
        self.ncalls_neural = 0
        self.n_neuralikes = 0
        self.train_counter = 0
        self.originalike_counter = 0
        self.trained_net = False
        self.net = None
        self.rootname = rootname
        self.neuralike_settings = neuralike_settings

    def run(self, delta_logz, it, nc, samples, likes,
            map_fn=None, perctest=0.1, logl_tolerance=0.05):
        if self.training_flag(delta_logz, it):
            self.train(samples, likes, map_fn=map_fn)
        if self.trained_net:
            self.neural_switch(nc, samples, likes,
                               map_fn=map_fn,
                               perctest=perctest,
                               logl_tolerance=logl_tolerance)

        info = "\nneural calls: {} | neuralikes: {} | "\
               "neural trains: {} | Using: ".format(self.ncalls_neural, self.n_neuralikes,
                                                    self.train_counter)
        if self.trained_net:
            print(info+'Neuralike')
        else:
            if self.train_counter > 0:
                self.originalike_counter += 1
            print(info+'Original_fn {}-aft'.format(self.originalike_counter))
        return None

    def training_flag(self, delta_logz, it):
        start_it = (it >= self.nstart_samples)
        startlogz = (delta_logz <= self.nstart_stop_criterion)
        if start_it or startlogz:
            # setting the conditions to train or retrain
            retrain = (self.originalike_counter >= self.updInt)
            first = (self.train_counter == 0)
            # if first or retrain:
            if retrain or first:
                self.last_train_it = it
                return True
            else:
                return False
        else:
            return False

    def train(self, samples, likes, map_fn=map):
        self.net = NeuralManager(loglikelihood=self.loglikelihood_control,
                                 samples=samples,
                                 likes=likes,
                                 rootname=self.rootname,
                                 neuralike_settings=self.neuralike_settings)
        self.net.training(map_fn=map_fn)
        self.train_counter += 1
        self.trained_net = self.net.valid
        self.originalike_counter = 0
        return None

    def neural_switch(self, nc, samples, likes, map_fn=map,
                      perctest=0.1, logl_tolerance=0.05):
        self.n_neuralikes += 1
        self.ncalls_neural += nc
        if nc > 1000:
            self.trained_net = False
            print("\nExcesive number of calls, neuralike disabled")
        elif self.n_neuralikes % (self.updInt // 2) == 0:
            samples_test = samples[-self.updInt:, :]
            neuralikes_test = likes[-self.updInt:]

            # real_logl = np.array(list(map_fn(self.loglikelihood_control,
            #                                  samples_test)))

            pred_test = self.test_predictions(samples_test, neuralikes_test,
                                             perctest=perctest,
                                             logl_tolerance=logl_tolerance, map_fn=map_fn)
            if pred_test:
                self.trained_net = True
            else:
                self.trained_net = False
        return None

    def likelihood(self, params):
        if self.trained_net:
            return self.net.neuralike(params)
        else:
            return self.loglikelihood_control(params)

    # @staticmethod
    def test_predictions(self, samples_test, y_pred, perctest=0.1, logl_tolerance=0.5, fractrues=0.8, map_fn=map):
        print("\n\nTesting neuralike predictions...")
        nlen = len(y_pred)
        nsize = int(perctest*nlen)
        print("Using {} points to testing".format(nsize))
        y_pred = y_pred.reshape(nlen, 1)
        idx_shuffle = np.random.permutation(nlen)
        idx_red = idx_shuffle[:nsize]
        y_pred = y_pred[idx_red]
        samples_test = samples_test[    idx_red]
        y_real = np.array(list(map_fn(self.loglikelihood_control, samples_test)))
        y_real = y_real.reshape(-1, 1)
        absdiff = np.abs(y_real - y_pred)
        absdiff_criterion = np.abs(logl_tolerance * y_real)
        comparison = absdiff <= absdiff_criterion
        count_trues = np.count_nonzero(comparison)
        criterion = int(fractrues*nsize)
        print("Good predictions {}/{}, requiriment: {}".format(count_trues, nsize, criterion))
        if count_trues >= criterion:
            print("Nice neuralike predictions!")
            return True
        else:
            print("Bad neuralike predictions!")
            return False