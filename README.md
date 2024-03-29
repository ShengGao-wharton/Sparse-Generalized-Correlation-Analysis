# Sparse GCA and Thresholded Gradient Descent
[![Generic badge](https://img.shields.io/badge/MATLAB-R2020a-BLUE.svg)](https://shields.io/)
![R](https://img.shields.io/badge/R-CRAN-orange)


## Overview of Sparse GCA

Generalized correlation analysis (GCA) is concerned with uncovering linear relationships across multiple datasets. It generalizes canonical correlation analysis that is designed for two datasets. We study sparse GCA when there are potentially multiple generalized correlation tuples in data and the loading matrix has a small number of nonzero rows. It includes sparse CCA and sparse PCA of correlation matrices as special cases. We first formulate sparse GCA as generalized eigenvalue problems at both population and sample levels via a careful choice of normalization constraints. Based on a Lagrangian form of the sample optimization problem, we propose a thresholded gradient descent algorithm for estimating GCA loading vectors and matrices in high dimensions. We derive tight estimation error bounds for estimators generated by the algorithm with proper initialization. We also demonstrate the prowess of the algorithm on a number of synthetic datasets.

This repository is the official implementation of our algorithm for sparse GCA in MATLAB and R.

For details of the algorithm, please check at https://www.jmlr.org/papers/v24/21-0745.html.

## Algorithm

Our algorithm for solving sparse GCA is based on thresholded gradient descent (TGD) with generalized Fantope projection as the initialization. 

## Implementation

The algorithm is implemented in both MATLAB and R. For details of the implementation, please find in `src/MATLAB Code`  and `src/R Code`. Examples of using our algorithm can be found in `tutorials/MATLAB examples`  and `tutorials/R examples`.

## Citation 

If you find our work interesting or useful for your research. Please cite our work

### Reference
```
@article{JMLR:v24:21-0745,
  author  = {Sheng Gao and Zongming Ma},
  title   = {Sparse GCA and Thresholded Gradient Descent},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {135},
  pages   = {1--61},
  url     = {http://jmlr.org/papers/v24/21-0745.html}
}
```
