<a href="https://arxiv.org/abs/2405.03293">
  <img src="https://img.shields.io/badge/arXiv-2405.03293-b31b1b.svg" alt="arXiv:2405.03293">
</a>

# neuralike

## Deep Learning and Genetic Algorithms for Cosmological Bayesian Inference Speed-up

Code of our paper *Deep Learning and genetic algorithms for cosmological Bayesian inference speed-up*, preprint available in https://arxiv.org/abs/2405.03293.

## Repository Structure

- neuralike/
    - **NeuraLike.py**.- Main class, gathers all other classes.
    - **NeuralManager.py**.-  API class, Manager for neural networks to learn likelihood function over a grid.
    - **NeuralNet.py**.- Class with neural net architecture in PyTorch.
    - **RandomSampling.py**.- Creates random samples in the parameter space and evaluates the likelihood in them. This is used to generate the training set for a neural network.
    - **pytorchtools.py**.- Methods and utilities for PyTorch.


## Usage

In the branch **neuralike** of the repository https://github.com/igomezv/simplemc_tests/tree/neuralike it is available neuralike integrated within the dynesty library for nested sampling within the SimpleMC cosmological parameter estimation code (https://igomezv.github.io/SimpleMC/).

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
