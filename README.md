<a href="https://arxiv.org/abs/2402.18124">
  <img src="https://img.shields.io/badge/arXiv-2402.18124-b31b1b.svg" alt="arXiv:2402.18124">
</a>

# neuralike

## Deep Learning and Genetic Algorithms for Cosmological Bayesian Inference Speed-up

Code of our paper *Deep Learning and genetic algorithms for cosmological Bayesian inference speed-up*, preprint available in https://arxiv.org/abs/2405.03293.

### Abstract

In this paper, we present a novel approach to accelerate the Bayesian inference process, focusing specifically on the nested sampling algorithms. Bayesian inference plays a crucial role in cosmological parameter estimation, providing a robust framework for extracting theoretical insights from observational data. However, its computational demands can be substantial, primarily due to the need for numerous likelihood function evaluations. Our proposed method utilizes the power of deep learning, employing feedforward neural networks to approximate the likelihood function dynamically during the Bayesian inference process. Unlike traditional approaches, our method trains neural networks on-the-fly using the current set of live points as training data, without the need for pre-training. This flexibility enables adaptation to various theoretical models and datasets. We perform simple hyperparameter optimization using genetic algorithms to suggest initial neural network architectures for learning each likelihood function. Once sufficient accuracy is achieved, the neural network replaces the original likelihood function. The implementation integrates with nested sampling algorithms and has been thoroughly evaluated using both simple cosmological dark energy models and diverse observational datasets. Additionally, we explore the potential of genetic algorithms for generating initial live points within nested sampling inference, opening up new avenues for enhancing the efficiency and effectiveness of Bayesian inference methods.

## Repository Structure

- neuralike/
    - **NeuraLike.py**.- Main class, gathers all other classes.
    - **NeuralManager.py**.-  API class, Manager for neural networks to learn likelihood function over a grid.
    - **NeuralNet.py**.- Class with neural net architecture in PyTorch.
    - **RandomSampling.py**.- Creates random samples in the parameter space and evaluates the likelihood in them. This is used to generate the training set for a neural network.
    - **pytorchtools.py**.- Methods and utilities for PyTorch.


## Usage

In the branch **neuralike** of the repository https://github.com/igomezv/simplemc_tests/tree/neuralike it is available neuralike integrated within the dynesty library for nested sampling. 

## Acknowledgments

We based or inspired our work on the following external codes:

- https://dynesty.readthedocs.io/en/stable
- https://github.com/DarkMachines/pyBAMBI
- https://arxiv.org/abs/1110.2997
- https://deap.readthedocs.io/en/master
- https://pytorch.org
- https://igomezv.github.io/SimpleMC

## Citation

If you use this work in your research, please cite:

```bibtex
@article{neuralike,
  title={Deep Learning and genetic algorithms for cosmological Bayesian inference speed-up},
  author={G{\'o}mez-Vargas, Isidro and V{\'a}zquez, J Alberto},
  journal={arXiv preprint arXiv:2405.03293},
  year={2024}
}
```

If you find useful our [`nnogada`](https://github.com/igomezv/Nnogada) framework:

```bibtex
@article{nnogada,
  title={Neural networks optimized by genetic algorithms in cosmology},
  author={Gómez-Vargas, I. and Andrade, J. B. and Vázquez, J. A.},
  journal={Physical Review D},
  volume={107},
  number={4},
  pages={043509},
  year={2023},
  publisher={American Physical Society},
  doi={https://doi.org/10.1103/PhysRevD.107.043509},
  url={https://doi.org/10.48550/arXiv.2209.02685}
}
```
