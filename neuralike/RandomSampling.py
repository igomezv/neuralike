import os
import numpy as np
import time


class RandomSampling:
    def __init__(self, like_fn, samples, means, mins, maxs, nrand=10, files_path='randomsampling'):
        """
        Create a random samples in the parameter space and evaluate the likelihood in them.
        This is used to generate the training set for a neural network.

        Parameters
        ----------
        like: likelihood object
        pars: list of Parameter objects
        nrand: number of random points in the parameter space. Default is 500
        """
        self.like = like_fn
        self.samples = samples
        self.means = means
        # self.dims = len(means)
        self.mins = mins
        self.maxs = maxs
        self.dims = len(mins)
        # self.cov = cov
        self.nrand = nrand
        self.files_path = files_path
        print("\nGenerating a random sample of points in the parameter space...")

    def make_sample(self):
        std = np.std(self.samples, axis=0)/4
        samples = np.random.normal(loc=self.means, scale=std, size=(self.nrand, self.dims))
        print("Random samples in the parameter space generated!")
        return samples

    def make_dataset(self, map_fn=map):
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
        likes = np.array(list(map_fn(self.like, samples)))
        tf = time.time() - t1
        print("Time of {} likelihood evaluations {:.4f} min".format(len(likes), tf/60))
        return samples, likes
