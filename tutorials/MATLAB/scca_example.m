%% Example of TGD on Sparse CCA
% n = 500, p1 = p2 = 200, s_u = s_v = 5
% k = 20, eta = 0.001, lambda =
% 0.01, T = 15000

ss = RandStream('mt19937ar', 'Seed', 7);

disp('  ');
disp('--------------------------------------');
disp('SCCA Example: Toeplitz covariance matrix'); 
disp('n = 500, p1 = p2 = 200, s_u = s_v = 5');
disp('lambda1 = 0.9, lambda2 = 0.8');
 
n  = 500;
p1 = 100;
p2 = 100;
p = p1 + p2;
pp = [p1,p2];
s  = randsample(p1,5);
lambda = diag( [0.9,  0.8] );
r = 2;
 
rr = 2;
 
disp('--------------------------------------');
disp('Generating data ...');
 
a = 0.3;
Sigma = eye(p1+p2);

% generate covariance matrix for X and Y
T1 = toeplitz(a.^(0:1:(p1-1)));
Sigma(1:p1, 1:p1) = T1;
Tss = T1(s, s);
u = zeros(p1,r);
u(s,(1:r)) = randi(ss, [-2, 2], size(u(s,(1:r))));
u = u / sqrtm(u(s,1:r)' * Tss * u(s,1:r));
 
 
T2 = toeplitz(a.^(0:1:(p2-1)));
Sigma((p1+1):(p1+p2),(p1+1):(p1+p2)) = T2;
Tss = T2(s, s);
v = zeros(p2,r);
v(s,(1:r)) = randi(ss, [-2, 2], size(v(s,(1:r))));
v = v / sqrtm(v(s,1:r)' * Tss * v(s,1:r));
 
 
Sigma((p1+1):(p1+p2), 1:p1) = T2 *  v  *lambda* u' * T1;
Sigma(1:p1, (p1+1):(p1+p2)) = Sigma((p1+1):(p1+p2), 1:p1)';
 
%sss = rand(p1+p2,1) + 2.5;
%Sigma = diag(sss) * Sigma * diag(sss);
 
[u_n, ~, ~] = svd(u, 'econ');
[v_n, ~, ~] = svd(v, 'econ');
 
Data = mvnrnd(zeros(p1+p2,1), Sigma, n);
 
X = Data(:,1:p1);
Y = Data(:,(p1+1):(p1+p2));
 
 
disp('Data generated.');
disp('--------------------------------------');

idx1 =1:p1;
idx2 = (p1+1):(p1+p2);
Mask = zeros(size(Sigma));
Mask(idx1,idx1)=ones(p1,p1);
Mask(idx2,idx2)=ones(p2,p2);

Sigma0 = Sigma .* Mask;
S=cov(Data);
sigma0hat = S .* Mask;


[V,D,W] = eig(Sigma,Sigma0);
[d,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);

d=diag(Ds);
lambdar=diag(d(1:r));
Sigmax = Sigma(1:p1,1:p1);
Sigmay = Sigma(p1+1:p,p1+1:p);


tic;
%% Fantope Projection
[solx, solx1, SSS] = sgca_init(Data, pp, r, 0.55, 1e-6, 30);
[aest, dest] = svd(solx);
aest = aest(:,1:r)*dest(1:r,1:r)^0.5;
toc;


%% Now we begin the gradient hard thresholding step
lambda = 0.01;
k=20;
ainit = hard_thre(aest,k);
a_true = Vs(:,1:r);
eta = 0.001;
tol = 1e-6;
max_iter = 10000; 
final_a = sgca_tgd(Data, S, sigma0hat, ainit,r, k ,eta,lambda, tol, max_iter);

%% Finally, we convert back to estimation of U,V by a renormalizing funtion gca_to_cca

[u_init_estimate, v_init_estimate] = gca_to_cca(aest, S, pp);
[u_final_estimate, v_final_estimate] = gca_to_cca(final_a, S, pp);


disp('Initial prediction error of U is')
disp( subdistance(Sigmax^(1/2)*u_init_estimate, Sigmax^(1/2)*u)^2)
disp('Initial prediction error of V is')
disp( subdistance(Sigmay^(1/2)*v_init_estimate,Sigmay^(1/2)* v)^2)

disp('The final prediction error on U is')
disp( subdistance(Sigmax^(1/2)*u_final_estimate,Sigmax^(1/2)* u)^2)

disp('The final prediction error on V is')
disp( subdistance(Sigmay^(1/2)*v_final_estimate,Sigmay^(1/2)* v)^2)




