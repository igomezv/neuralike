# neuralike

## Deep Learning and Genetic Algorithms for Cosmological Bayesian Inference Speed-up

## Abstract

In this paper, we present a novel approach to accelerate the Bayesian inference process, focusing specifically on the nested sampling algorithms. Bayesian inference plays a crucial role in cosmological parameter estimation, providing a robust framework for extracting theoretical insights from observational data. However, its computational demands can be substantial, primarily due to the need for numerous likelihood function evaluations. Our proposed method utilizes the power of deep learning, employing feedforward neural networks to approximate the likelihood function dynamically during the Bayesian inference process. Unlike traditional approaches, our method trains neural networks on-the-fly using the current set of live points as training data, without the need for pre-training. This flexibility enables adaptation to various theoretical models and datasets. We perform simple hyperparameter optimization using genetic algorithms to suggest initial neural network architectures for learning each likelihood function. Once sufficient accuracy is achieved, the neural network replaces the original likelihood function. The implementation integrates with nested sampling algorithms and has been thoroughly evaluated using both simple cosmological dark energy models and diverse observational datasets. Additionally, we explore the potential of genetic algorithms for generating initial live points within nested sampling inference, opening up new avenues for enhancing the efficiency and effectiveness of Bayesian inference methods.

## Repository Structure

- neuralike/
    - **NeuraLike.py**.- Main class, gathers all other classes.
    - **NeuralManager.py**.-  API class, Manager for neural networks to learn likelihood function over a grid.
    - **NeuralNet.py**.- Class with neural net architecture in PyTorch.
    - **RandomSampling.py**.- Creates random samples in the parameter space and evaluates the likelihood in them. This is used to generate the training set for a neural network.
    - **pytorchtools.py**.- Methods and utilities for PyTorch.



