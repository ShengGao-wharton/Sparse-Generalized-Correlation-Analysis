function [u_estimate, v_estimate] = gca_to_cca(a_estimate, S, pp)

% Given GCA estimation for A, convert to CCA estimation of U,V by
% renormalization

% Inputs:
% =======
% a_estimate: estimation for A
% S:          covariance matrix for data cov(X)
% pp:         a length 2 vector giving the sizes of each variable subset
% 
% Outputs:
% ========
% u_estimate:  final estimator of U
% v_estimate:  final estimator of V

p1 = pp(1);
p2 = pp(2);
p = p1 + p2;
sigmaxhat = S(1:p1,1:p1);
sigmayhat = S(p1+1:p,p1+1:p);
u_estimate = a_estimate(1:p1,:) * (a_estimate(1:p1,:)' * sigmaxhat *a_estimate(1:p1,:))^(-1/2);
v_estimate = a_estimate(p1+1:p,:) * (a_estimate(p1+1:p,:)' * sigmayhat *a_estimate(p1+1:p,:))^(-1/2);
end