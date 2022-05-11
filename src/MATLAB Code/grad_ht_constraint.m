function final_estimate = grad_ht_constraint(X ,a, S ,sigma0hat ,ainit ,r ,k ,eta ,lambda , mu, tol , max_iter)
% Function for Thresholded Gradient Descent (TGD)
% Inputs:
% =======
% X:       data sets, n rows correspond to samples and p columns to variables
% S:       Cov(X), i.e, sigmahat
% a:       vector to be normalized with respect to 
% sigma0hat: Masked version of S (estimator of Sigma_0)
% ainit:   the initial estimator obtained by fantope projection (or any
%          other initialization), need to be k-sparse
% r:       latent variable dimension
% k:       number of non-zeros columns to keep at each iteration
% eta:     stepsize in gradient descent
% lambda:  penalty term in gradient descent
% tol:     tolerance level for convergence in gradient descent
% max_iter:maximum number of iterations in GD
% 
% Outputs:
% ========
% final_estimate:  final estimator of A, the generalized eigenspace for
%                  matrix pair Sigma and Sigma_0
 
ainit = ainit*(eye(r)+ainit'*S*ainit/lambda)^0.5;
ut=ainit;
for i = 1:max_iter
    grad = -S*ut + lambda*sigma0hat*ut*(ut'*sigma0hat*ut-eye(r)) + mu * sigma0hat*a*a'*sigma0hat*ut;
    vt = ut - 2*eta*grad;
    ut = hard_thre(vt,k);
end
final_estimate = ut*(ut'*sigma0hat*ut)^(-1/2);
end