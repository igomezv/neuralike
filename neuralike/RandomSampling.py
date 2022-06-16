import os
import numpy as np
import time


class RandomSampling:
    # def __init__(self, like, means, cov, nrand=10, pool=None, files_path='randomsampling'):
    def __init__(self, like, means, mins, maxs, nrand=10, pool=None, files_path='randomsampling'):
        """
        Create a random samples in the parameter space and evaluate the likelihood in them.
        This is used to generate the training set for a neural network.

        Parameters
        ----------
        like: likelihood object
        pars: list of Parameter objects
        nrand: number of random points in the parameter space. Default is 500
        """
        self.like = like
        self.means = means
        # self.dims = len(means)
        self.mins = mins
        self.maxs = maxs
        self.dims = len(mins)
        # self.cov = cov
        self.nrand = nrand
        self.pool = pool
        self.files_path = files_path
        if pool:
            self.M = pool.map
        else:
            self.M = map
        print("\nGenerating a random sample of points in the parameter space...")

    def make_sample(self):
        # if not self.filesChecker():
        # samples = np.random.multivariate_normal(self.means, 5e-3*self.cov, size=(self.nrand,))
        # else:
        #     print('Loading existing random_samples and likelihoods: {}'.format(self.files_path))
        #     samples = np.load('{}_random_samples.npy'.format(self.files_path))
        # print("cov, means, samples", np.shape(self.cov), np.shape(self.means), np.shape(samples))
        # samples = np.zeros((self.nrand, self.dims))
        # for i in range(self.dims):
        d1 = np.abs(self.means-self.mins)
        d2 = np.abs(self.means-self.maxs)
        std = np.abs(d2-d1)/4
        samples = np.random.normal(loc=self.means, scale=std, size=(self.nrand, self.dims))
        print("Random samples in the parameter space generated!")
        return samples


    def make_dataset(self):
        """
        Evaluate the Likelihood function on the grid
        Returns
        -------
        Random samples in the parameter space and their respectives likelihoods.
        """
        samples = self.make_sample()
        t1 = time.time()
        # if not self.filesChecker():
        print("Evaluating likelihoods...")
        likes = np.array(list(self.M(self.like, samples)))
        tf = time.time() - t1
        print("Time of {} likelihood evaluations {:.4f} min".format(len(likes), tf/60))
        if self.pool:
            self.pool.close()
        # print("Time of evaluating {} likelihoods with apply_along_axis: {:.4} s".format(len(likes), tf))

        return samples, likes

    # def filesChecker(self):
    #     """
    #     This method checks if the name of the random_samples.npy and likes.npy exists, if it already exists use it
    #     """
    #     if os.path.isfile('{}_random_samples.npy'.format(self.files_path)):
    #         if os.path.isfile('{}_likes.npy'.format(self.files_path)):
    #             return True
    #     else:
    #         return False

