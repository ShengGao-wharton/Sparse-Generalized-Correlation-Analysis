% Example of Sparse Generalized Correlation Analysis
% k = 3 high dimensional datasets, r = 3 (can be specified to other integer
% values), n = 500, p1 = 500, p2 = p3 = 200, k = 20, eta = 0.001, lambda =
% 0.01, T = 15000
%% data generation
disp('  ');
disp('--------------------------------------');
disp('SGCA Example: 3 high dimentional datasets with Toeplitz covariance matrix'); 
disp('n = 500, p1 = 500, p2 = p3 = 200, s1 = s2 = s3 = 5');
 

pp = [100, 50, 50];
s  = [1, 6, 11,24];
r = 1;
n = 300;
k = 20;
eta = 0.001;
lambda = 0.01;
max_iter = 15000;

u1 = zeros(pp(1),r);
u2 = zeros(pp(2),r);
u3 = zeros(pp(3),r);


Sigma = eye(sum(pp(:)));



T1 = toeplitz(0.5.^(0:1:(pp(1)-1)));
T2 = toeplitz(0.7.^(0:1:(pp(2)-1)));
T3 = toeplitz(0.9.^(0:1:(pp(3)-1)));
u1(s,1:r) = randn(length(s),r);
u1 = u1 / sqrtm(u1(s,1:r)' * T1(s,s) * u1(s,1:r));
u2(s,1:r) = randn(length(s),r);
u2 = u2 / sqrtm(u2(s,1:r)' * T2(s,s) * u2(s,1:r));
u3(s,1:r) = randn(length(s),r);
u3 = u3 / sqrtm(u3(s,1:r)' * T3(s,s) * u3(s,1:r));

idx1 = 1:pp(1);
idx2 = pp(1)+1:pp(1)+pp(2);
idx3 = pp(1)+pp(2)+1:pp(1)+pp(2)+pp(3);
Sigma(idx1, idx1) = T1;
Sigma(idx2, idx2) = T2;
Sigma(idx3, idx3) = T3;
SigmaD = Sigma;

Sigma(idx1, idx2) = T1 * u1 * u2' * T2;
Sigma(idx1, idx3) = T1 * u1 * u3' * T3;
Sigma(idx2, idx3) = T2 * u2 * u3' * T3;
Sigma = Sigma + Sigma' - SigmaD;


X = mvnrnd(zeros(sum(pp(:)),1),Sigma,n);
S = cov(X);
Mask = zeros(size(Sigma));
Mask(idx1,idx1) = ones(pp(1),pp(1));
Mask(idx2,idx2) = ones(pp(2),pp(2));
Mask(idx3,idx3) = ones(pp(3),pp(3));

Sigma0 = Sigma .* Mask;
[V,D,W] = eig(Sigma,Sigma0);
[d,ind] = sort(diag(D),'descend');
[V1,D1,W1] = eig(S,S.*Mask);
[d1,ind1] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);
a = Vs(:,1:r);
disp('Data generated.');
disp('--------------------------------------');


tic;
%% Fantope projection 
[solx, solx1, SSS] = sgca_init(X, pp, r, 0.5, 1e-6, 30);
[aest, sest] = svd(solx);
aest = aest(:,1:r)*sest(1:r,1:r)^0.5;
toc;

%% Now we begin the gradient hard thresholding step
ainit = hard_thre(aest, k);
a_true = a;
tol = 1e-6;
sigma0hat = S .* Mask;
final = sgca_tgd(X,S,sigma0hat,ainit,r,k,eta,lambda,tol,max_iter);
final_a = final;

%% Print the error of initialization and final estimator 
disp('initial error of A is')
init_error = subdistance(aest, a_true)
disp('The final error of A is')
final_error =  subdistance(final_a, a_true)

