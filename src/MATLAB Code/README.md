# MATLAB Code for Sparse GCA

We provide the code of our paper: Sparse GCA and Thresholded Gradient Descent.

## Prerequisites
MATLAB TFOCS
avaiable at: http://cvxr.com/tfocs/download/

## Main scripts

### 1. Intialization using generalized Fantope Projection
`sgca_init.m`: Performs initialization using convex relaxation for generalized correlation analysis

### 2. Thresholded Gradient Descent
`sgca_tgd.m`: Performs thresholded gradient descent on given initialization, output the final estimator for generalized correlation matrix

### 3. Utility Functions
-`gca_to_cca.m` Convert GCA estimator of A to CCA estimators of U and V\
-`hard_thre.m` Function to perform hard thresholding\
-`linop_crosprod.m`, `cap_soft_th.m` used in initialization\
-`subdistance.m` compute the matrix distance defined in the paper

### 4. Experiment Examples:
-`sgca_example.m` Simulation for Sparse GCA on 3 high dimensional datasets\
-`scca_example.m` Simulation for Sparse CCA using TGD\
-`spca_example.m` Simulation for Sparse PCA on correlation matrices

Details of experimental settings can be found in the paper. 
