# R Code for Sparse GCA

We provide the R code implementation of our paper: Sparse GCA and Thresholded Gradient Descent.

## Prerequisites
R package, need to install \
library(MASS)
library(stats)
library(geigen)
library(pracma)

## Main scripts

### 1. Intialization using generalized Fantope Projection
`sgca_init.R`: Performs initialization using convex relaxation for generalized correlation analysis

### 2. Thresholded Gradient Descent
`sgca_tgd.R`: Performs thresholded gradient descent on given initialization, \
output the final estimator for generalized correlation matrix

### 3. Utility Functions
-`gca_to_cca.R` Convert GCA estimator of A to CCA estimators of U and V\
-`init_process.R` Function to convert Fantope estimation to initial estimator used in TGD\
-`utils.R` include functions used in initialization and hard thresholding\
-`subdistance.R` compute the matrix distance defined in the paper

### 4. Experiment Examples:
-`sgca_example.m` Simulation for Sparse GCA on 3 high dimensional datasets\
-`scca_example.m` Simulation for Sparse CCA using TGD

Details of experimental settings can be found in the paper. 

### 5. Running Pipelines
#### (1) Initialization
Use `init_res <- (A=S, B=sigma0hat, rho = sqrt(log(p)/n),K = r,nu=1, epsilon=5e-3,maxiter=1000,trace=FALSE)` \
to get initial estimate of generalized canonical correlation matrix
#### (2) Post-process of Initialization
Use `ainit <- init_process(init_res$Pi, r)` to get input for TGD
#### (3) Perform TGD
Use `final <- sgca_tgd(A=S, B=sigma0hat,r,ainit,k,lambda = 0.01, eta=0.001,convergence=1e-6,maxiter=15000, plot = FALSE)`  
to get final estimate of leading generalized eigenspace
#### (4) Deal with Sparse CCA
If computing estimators for Sparse CCA, use 
```
init <- gca_to_cca(ainit, S, pp)
final <- gca_to_cca(final, S, pp)
initu <- init$u
initv <- init$v
finalu <- final$u
finalv <- final$v
```
to obtain the final estimate. 