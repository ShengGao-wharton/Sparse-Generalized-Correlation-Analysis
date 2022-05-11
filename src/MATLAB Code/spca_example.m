%% Example pf Sparse PCA using TGD
% r=3, s=20
% we first generate an orthogonal matrix of size 20*3 whose
% row l2 norm are same for all rows, let it be A1
% to generate A1, we notice A0 = [1,1,-1;1,-1,-1;1,1,1;1,-1,1] has same row norm
% We use A0 to get one block for A1. 
% Then we generate 5 blocks in this way and concantenate them to get A1

disp('  ');
disp('--------------------------------------');
disp('SPCA Example: Toeplitz covariance matrix'); 
disp('n = 500, p = 500, s=20, leading eigenvalues are 7, 5, 3.');
 

n = 500;
p = 500;
k = 40;
type = 2;
%type I eigenvalues are 555, type II eigenvalues are 753
s=20;
r=3;

A0=[1,1,-1;1,-1,-1;1,1,1;1,-1,1];
A1=[A0;A0;A0;A0;A0];

% when generating upper_block, we make sure that we add small pertubation
% such that phi is non-degenerate
if type==1 % this is for eigenvalue 5,5,5
    f =ones(r,1);
    A1 = sqrt(12/17)*diag(vecnorm(A1,2,2))^(-1)*A1;
    upper_block = A1*diag(f)*A1'+ 5/17*eye(s);
    phi = [upper_block, zeros(s,p-s);zeros(p-s,s),eye(p-s)];

elseif type == 2 % this is for 7,5,3
    f =[60/17+3/2,60/17,60/17-3/2];
    A1 = sqrt(1/5)*diag(vecnorm(A1,2,2))^(-1)*A1;
    upper_block = A1*diag(f)*A1'+ 5/17*eye(s);
    phi = [upper_block, zeros(s,p-s);zeros(p-s,s),eye(p-s)];
end


%generate sigma_0 with random numbers in [0.1,1] on the diagonal 
sigma0 = diag(rand(p,1)*0.9+0.1);
sigma = sigma0^0.5*phi*sigma0^0.5;

[V,D,W] = eig(sigma,sigma0);
[d,ind] = sort(diag(D),'descend');
Ds = D(ind,ind);
Vs = V(:,ind);

%generate data X from MVN and sigmahat, sigma0hat
X = mvnrnd(zeros(p,1),sigma,n);
S = cov(X);
Mask = eye(p);
sigma0hat = Mask .* S;
pp=ones(p,1);
disp('Data Generated')

%% Now we consider the Fantope projection step 

[solx, solx1, SSS] = sgca_init(X, pp, r, 0.5, 1e-6, 30);
[aest, sest] = svd(solx);
aest = aest(:,1:r)*sest(1:r,1:r)^0.5;



%% Perform Thresholded Gradient Descent
ainit = hard_thre(aest,k);
lambda = 0.01;

eta=0.001;
total=15000;
tol=1e-6;

final_a = sgca_tgd(X, S, sigma0hat, ainit,r, k ,eta,lambda, tol, total);

%% Final normalization
% our interest is in E, the eigenspace of phi
% so we multiply estimate by sigma0hat^(1/2)

afinal = sigma0hat^(1/2) * final_a;
corre_true = [A1*(A1'*A1)^(-1/2);zeros(p-s,3)];
init_error = subdistance(sigma0hat^(1/2)* aest,corre_true);
final_error = subdistance(afinal, corre_true);

disp('The initial error is')
disp(init_error)
disp('The final error is')
disp(final_error)
disp('The leading 4 eigenvalues are')
disp(d(1:4))